import argparse
import cv2
import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.utils.data
import os
import shutil
import numpy as np
import random
from skimage import measure
import logging
from torch.utils.tensorboard import SummaryWriter
from torch import nn

from DeepLabv3_plus import DeepLabv3_plus
from DefEDNetmain.DefEDNet import DefED_Net
from MyNet import MyNet
from MyUnet import MyUnet
from Pspnet import Pspnet
from SegNet import *
from VM_Unet import VMUNet
import utils
from data_folder import DataFolder
from hausdorff_loss import HausdorffERLoss
from models import DeepLab
from network import deeplabv3plus_resnet101, deeplabv3_resnet101
from options import Options
from my_transforms import get_transforms
from loss import LossVariance, dice_loss

from resunet import ResUNet
from MedicalTransformer.axialnet import MedT
from smatunetmodels.SmaAt_UNet import SmaAt_UNet
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg


def main():
    global opt, best_iou, num_iter, tb_writer, logger, logger_results
    torch.backends.cudnn.enabled = False
    best_iou = 0
    opt = Options(isTrain=True)
    opt.parse()
    opt.save_options()

    tb_writer = SummaryWriter('{:s}/tb_logs'.format(opt.train['save_dir']))
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.train['gpu'])
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # set up logger
    logger, logger_results = setup_logging(opt)
    opt.print_options(logger)

    model = VMUNet(3, 3, [2,2,2,2],[2,2,2,1], 0.2, './vmamba_tiny_e292.pth')


    model = nn.DataParallel(model)
    model = model.cuda()
    torch.backends.cudnn.benchmark = True

    # ----- define optimizer ----- #
    optimizer = torch.optim.Adam(model.parameters(), opt.train['lr'], betas=(0.9, 0.99),
                                 weight_decay=opt.train['weight_decay'])

    # ----- define criterion ----- #
    # mseloss = torch.nn.MSELoss(reduction='none').cuda()
    criterion = torch.nn.NLLLoss(reduction='none').cuda()
    global criterion_hau
    criterion_hau = HausdorffERLoss()
    if opt.train['alpha'] > 0:
        logger.info('=> Using variance term in loss...')
        global criterion_var
        criterion_var = LossVariance()

    data_transforms = {'train': get_transforms(opt.transform['train']),
                       'valA': get_transforms(opt.transform['val']),
                       'valB': get_transforms(opt.transform['val'])}

    # ----- load data ----- #
    dsets = {}
    for x in ['train', 'valA', 'valB']:
        img_dir = '{:s}/{:s}'.format(opt.train['img_dir'], x)
        target_dir = '{:s}/{:s}'.format(opt.train['label_dir'], x)
        weight_map_dir = '{:s}/{:s}'.format(opt.train['weight_map_dir'], x)
        dir_list = [img_dir, weight_map_dir,  target_dir]
        if opt.dataset == 'MultiOrgan':
            post_fix = ['weight.png', 'label.png']
        else:
            post_fix = ['anno_weight.png', 'anno_label.png']
        num_channels = [3, 1, 3]
        dsets[x] = DataFolder(dir_list, post_fix, num_channels, data_transforms[x])

    # 加载DeepLabv3_plus时，由于batchnorm层需要大于一个样本去计算其中的参数，
    # 解决方法是将dataloader的一个丢弃参数设置为true
    train_loader = DataLoader(dsets['train'], batch_size=opt.train['batch_size'], shuffle=True,
                              num_workers=opt.train['workers'], drop_last=True)

    val_loader = DataLoader(dsets['valB'], batch_size=1, shuffle=False,
                            num_workers=opt.train['workers'], drop_last=True)
    val_loader1 = DataLoader(dsets['valA'], batch_size=1, shuffle=False,
                             num_workers=opt.train['workers'], drop_last=True)
    # ----- optionally load from a checkpoint for validation or resuming training ----- #
    if opt.train['checkpoint']:
        if os.path.isfile(opt.train['checkpoint']):
            logger.info("=> loading checkpoint '{}'".format(opt.train['checkpoint']))
            checkpoint = torch.load(opt.train['checkpoint'])
            opt.train['start_epoch'] = checkpoint['epoch']
            best_iou = checkpoint['best_iou']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(opt.train['checkpoint'], checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(opt.train['checkpoint']))

    # ----- training and validation ----- #
    for epoch in range(opt.train['start_epoch'], opt.train['num_epochs']):
        # train for one epoch or len(train_loader) iterations
        logger.info('Epoch: [{:d}/{:d}]'.format(epoch + 1, opt.train['num_epochs']))
        train_results = train(train_loader, model, optimizer, criterion, epoch)
        train_loss, train_loss_ce, train_loss_var, train_pixel_acc, train_iou = train_results

        # evaluate on validation set
        with torch.no_grad():
            val_loss, val_pixel_acc, val_iou = validate(val_loader, model, criterion)
            val_loss1, val_pixel_acc1, val_iou1 = validate(val_loader1, model, criterion)

        # check if it is the best accuracy
        is_best = val_iou > best_iou
        best_iou = max(val_iou, best_iou)

        # is_second = val_iou >= 0.84
        if (val_iou >= 0.80):
            # val_loss1, val_pixel_acc1, val_iou1 = validate(val_loader1, model, criterion)
            is_second = val_iou1 >= 0.80
            cp_flag = (epoch + 1) % opt.train['checkpoint_freq'] == 0
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_iou': best_iou,
                'optimizer': optimizer.state_dict(),
            }, epoch, is_best, opt.train['save_dir'], cp_flag, is_second)

        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'state_dict': model.state_dict(),
        #     'best_iou': best_iou,
        #     'optimizer' : optimizer.state_dict(),
        # }, epoch, is_best, opt.train['save_dir'], cp_flag,is_second)

        # save the training results to txt files
        logger_results.info('{:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
                            .format(epoch + 1, train_loss, train_loss_ce, train_loss_var, train_pixel_acc,
                                    train_iou, val_loss, val_pixel_acc, val_iou))
        # tensorboard logs
        tb_writer.add_scalars('epoch_losses',
                              {'train_loss': train_loss, 'train_loss_ce': train_loss_ce,
                               'train_loss_var': train_loss_var, 'val_loss': val_loss}, epoch)
        tb_writer.add_scalars('epoch_accuracies',
                              {'train_pixel_acc': train_pixel_acc, 'train_iou': train_iou,
                               'val_pixel_accB': val_pixel_acc, 'val_iouB': val_iou,
                               'val_pixel_accA': val_pixel_acc1, 'val_iouA': val_iou1}, epoch)
    tb_writer.close()


def train(train_loader, model, optimizer, criterion, epoch):
    # list to store the average loss and iou for this epoch
    results = utils.AverageMeter(5)

    # switch to train mode
    model.train()
    for i, sample in enumerate(train_loader):
        input, weight_map, target = sample
        weight_map = weight_map.float().div(20)
        if weight_map.dim() == 4:
            weight_map = weight_map.squeeze(1)
        weight_map_var = weight_map.cuda()
        if torch.max(target) == 255:
            target = target / 255
        if target.dim() == 4:
            target1 = target.squeeze(1)
        target = F.one_hot(target, num_classes=3)

        target_one_hot0 = target[:, :, :, :, 0]
        target_one_hot1 = target[:, :, :, :, 1]
        target_one_hot2 = target[:, :, :, :, 2]

        input_var = input.cuda()
        target_var = target1.cuda()

        # compute output

        output = model(input_var)


        output1 = F.softmax(output, dim=1)

        # print(target1.shape)

        loss_haus1 = criterion_hau.forward(output1[:, 0:1, :, :], target_one_hot0)
        loss_haus2 = criterion_hau.forward(output1[:, 1:2, :, :], target_one_hot1)
        loss_haus3 = criterion_hau.forward(output1[:, 2:3, :, :], target_one_hot2)
        loss_haus = loss_haus1 + loss_haus2 + loss_haus3



        log_prob_maps = F.log_softmax(output, dim=1)
        loss_map = criterion(log_prob_maps, target_var)
        loss_map *= weight_map_var
        loss_CE = loss_map.mean()


        if opt.train['alpha'] != 0:
            prob_maps = F.softmax(output, dim=1)

            # label instances in target
            target_labeled = torch.zeros(target1.size()).long()
            for k in range(target1.size(0)):
                target_labeled[k] = torch.from_numpy(measure.label(target1[k].numpy() == 1))
                # utils.show_figures((target[k].numpy(), target[k].numpy()==1, target_labeled[k].numpy()))
            loss_var = criterion_var(prob_maps, target_labeled.cuda())
            loss = loss_CE + opt.train['alpha'] * loss_var + 1e-6 * loss_haus
        else:
            loss_var = torch.ones(1) * -1
            loss = loss_CE

        # measure accuracy and record loss
        pred = np.argmax(log_prob_maps.data.cpu().numpy(), axis=1)
        metrics = utils.accuracy_pixel_level(pred, target1.numpy())
        pixel_accu, iou = metrics[0], metrics[1]

        result = [loss, loss_CE, loss_var, pixel_accu, iou]
        results.update(result, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del input_var, output, target_var, log_prob_maps, loss

        if i % opt.train['log_interval'] == 0:
            logger.info('\tIteration: [{:d}/{:d}]'
                        '\tLoss {r[0]:.4f}'
                        '\tLoss_CE {r[1]:.4f}'
                        '\tLoss_var {r[2]:.4f}'
                        '\tPixel_Accu {r[3]:.4f}'
                        '\tIoU {r[4]:.4f}'.format(i, len(train_loader), r=results.avg))

    logger.info('\t=> Train Avg: Loss {r[0]:.4f}'
                '\tLoss_CE {r[1]:.4f}'
                '\tLoss_var {r[2]:.4f}'
                '\tPixel_Accu {r[3]:.4f}'
                '\tIoU {r[4]:.4f}'.format(epoch, opt.train['num_epochs'], r=results.avg))

    return results.avg


def validate(val_loader, model, criterion):
    # list to store the losses and accuracies: [loss, pixel_acc, iou ]
    results = utils.AverageMeter(3)

    # switch to evaluate mode
    model.eval()

    for i, sample in enumerate(val_loader):
        input, weight_map, target = sample
        weight_map = weight_map.float().div(20)
        if weight_map.dim() == 4:
            weight_map = weight_map.squeeze(1)
        weight_map_var = weight_map.cuda()



        if torch.max(target) == 255:
            target = target / 255
        if target.dim() == 4:
            target2 = target.squeeze(1)

        target_var = target2.cuda()

        size = opt.train['input_size'][0]
        overlap = opt.train['val_overlap']
        output = utils.split_forward(model, input, size, overlap, opt.model['out_c'])

        target = F.one_hot(target, num_classes=3)
        target_one_hot0 = target[:, :, :, :, 0]
        target_one_hot1 = target[:, :, :, :, 1]
        target_one_hot2 = target[:, :, :, :, 2]

        output1 = F.softmax(output, dim=1)
        # print(target1.shape)
        loss_haus1 = criterion_hau.forward(output1[:, 0:1, :, :], target_one_hot0)
        loss_haus2 = criterion_hau.forward(output1[:, 1:2, :, :], target_one_hot1)
        loss_haus3 = criterion_hau.forward(output1[:, 2:3, :, :], target_one_hot2)
        loss_haus = loss_haus1 + loss_haus2 + loss_haus3

        log_prob_maps = F.log_softmax(output, dim=1)

        loss_map = criterion(log_prob_maps, target_var)
        loss_map *= weight_map_var
        loss_CE = loss_map.mean()

        if opt.train['alpha'] != 0:
            prob_maps = F.softmax(output, dim=1)

            target_labeled = torch.zeros(target2.size()).long()
            for k in range(target2.size(0)):
                target_labeled[k] = torch.from_numpy(measure.label(target2[k].numpy() == 1))
                # utils.show_figures((target[k].numpy(), target[k].numpy()==1, target_labeled[k].numpy()))
            loss_var = criterion_var(prob_maps, target_labeled.cuda())
            loss = loss_CE + opt.train['alpha'] * loss_var + 1e-6 * loss_haus
        else:
            loss = loss_CE

        # measure accuracy and record loss
        pred = np.argmax(log_prob_maps.data.cpu().numpy(), axis=1)
        metrics = utils.accuracy_pixel_level(pred, target2.numpy())
        pixel_accu = metrics[0]
        iou = metrics[1]

        results.update([loss.item(), pixel_accu, iou])

        del output, target_var, log_prob_maps, loss

    logger.info('\t=> Val Avg:   Loss {r[0]:.4f}\tPixel_Acc {r[1]:.4f}'
                '\tIoU {r[2]:.4f}'.format(r=results.avg))

    return results.avg


def consistency_loss(logits_w1, logits_w2):
    # logits_w2 = logits_w2.detach()
    # assert logits_w1.size() == logits_w2.size()
    return F.mse_loss(logits_w1, logits_w2, reduction='mean')


def compute_sdf(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1,1]
    """
    img_gt = img_gt.detach().numpy().astype(np.uint8)
    normalized_sdf = np.zeros(out_shape)
    for b in range(out_shape[0]):  # batch size
        posmask = img_gt[b].astype(np.bool)
        if posmask.any():
            negmask = ~posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            # 归一化，分割区域内部为负，外部为正
            sdf = (negdis - np.min(negdis)) / (np.max(negdis) - np.min(negdis)) - (posdis - np.min(posdis)) / (
                        np.max(posdis) - np.min(posdis))
            # 边界置零
            sdf[boundary == 1] = 0
            normalized_sdf[b] = sdf
            # assert np.min(sdf) == -1.0, print(np.min(posdis), np.max(posdis), np.min(negdis), np.max(negdis))
            # assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))
    return normalized_sdf


def save_checkpoint(state, epoch, is_best, save_dir, cp_flag, is_second):
    cp_dir = '{:s}/checkpoints'.format(save_dir)
    if not os.path.exists(cp_dir):
        os.mkdir(cp_dir)
    filename = '{:s}/checkpoint.pth.tar'.format(cp_dir)
    torch.save(state, filename)
    if cp_flag:
        shutil.copyfile(filename, '{:s}/checkpoint_{:d}.pth.tar'.format(cp_dir, epoch + 1))
    if is_best:
        shutil.copyfile(filename, '{:s}/checkpoint_best.pth.tar'.format(cp_dir))
    if is_second:
        shutil.copyfile(filename, '{:s}/checkpoint_{:d}_demo.pth.tar'.format(cp_dir, epoch + 1))


def setup_logging(opt):
    mode = 'a' if opt.train['checkpoint'] else 'w'

    # create logger for training information
    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.DEBUG)
    # create console handler and file handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler('{:s}/train.log'.format(opt.train['save_dir']), mode=mode)
    file_handler.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s\t%(message)s', datefmt='%Y-%m-%d %I:%M')
    # add formatter to handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    # add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # create logger for epoch results
    logger_results = logging.getLogger('results')
    logger_results.setLevel(logging.DEBUG)
    file_handler2 = logging.FileHandler('{:s}/epoch_results.txt'.format(opt.train['save_dir']), mode=mode)
    file_handler2.setFormatter(logging.Formatter('%(message)s'))
    logger_results.addHandler(file_handler2)

    logger.info('***** Training starts *****')
    logger.info('save directory: {:s}'.format(opt.train['save_dir']))
    if mode == 'w':
        logger_results.info('epoch\ttrain_loss\ttrain_loss_CE\ttrain_loss_var\ttrain_acc\ttrain_iou\t'
                            'val_loss\tval_acc\tval_iou')

    return logger, logger_results


if __name__ == '__main__':
    main()
