import torch.optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.utils.data
import os
import shutil
from skimage import measure
import logging
from torch.utils.tensorboard import SummaryWriter
from CE_Net import CE_Net_,Our_CE_Net_,URPC_CE_Net_
import utils
from data_folder import DataFolder
from hausdorff_loss import HausdorffERLoss

from options_glas_semi import Options
from my_transforms import get_transforms
from loss import LossVariance
from UnetPlus import *

from copy import deepcopy


def main():
    global opt, best_iou, num_iter, tb_writer, logger, logger_results
    best_iou = 0
    opt = Options(isTrain=True)
    opt.parse()
    opt.save_options()
    torch.backends.cudnn.enabled = False

    tb_writer = SummaryWriter('{:s}/tb_logs'.format(opt.train['save_dir']))
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.train['gpu'])
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # set up logger
    logger, logger_results = setup_logging(opt)
    opt.print_options(logger)

    model = URPC_CE_Net_(3, 3)
    model = nn.DataParallel(model)
    model = model.cuda()
    torch.backends.cudnn.benchmark = True

    # ----- define optimizer ----- #
    optimizer = torch.optim.Adam(model.parameters(), opt.train['lr'], betas=(0.9, 0.99),
                                 weight_decay=opt.train['weight_decay'])

    # ----- define criterion ----- #
    global mseloss, criterion_hau, kl_distance
    kl_distance = nn.KLDivLoss(reduction='none')
    mseloss = torch.nn.MSELoss(reduction='mean').cuda()
    criterion = torch.nn.NLLLoss(reduction='none').cuda()
    criterion_hau = HausdorffERLoss()
    if opt.train['alpha'] > 0:
        logger.info('=> Using variance term in loss...')
        global criterion_var
        criterion_var = LossVariance()

    data_transforms = {'train': get_transforms(opt.transform['train']),
                       'train1': get_transforms(opt.transform['train1']),
                       'valA': get_transforms(opt.transform['val']),
                       'valB': get_transforms(opt.transform['val'])}

    # ----- load data ----- #
    dsets = {}
    for x in ['train', 'valA', 'valB']:
        img_dir = '{:s}/{:s}'.format(opt.train['img_dir'], x)
        target_dir = '{:s}/{:s}'.format(opt.train['label_dir'], x)
        weight_map_dir = '{:s}/{:s}'.format(opt.train['weight_map_dir'], x)
        dir_list = [img_dir, weight_map_dir, target_dir]
        if opt.dataset == 'MultiOrgan':
            post_fix = ['weight.png', 'label.png']
        else:
            post_fix = ['anno_weight.png', 'anno_label.png']
        num_channels = [3, 1, 3]
        dsets[x] = DataFolder(dir_list, post_fix, num_channels, data_transforms[x])

    dsets2 = {}
    for x1 in ['train1']:
        img_dir = '{:s}/{:s}'.format(opt.train['img_dir'], 'unlabel' + x1)
        dir_list = [img_dir]
        if opt.dataset == 'MultiOrgan':
            post_fix = ['weight.png', 'label.png']
        else:
            post_fix = []
        num_channels = [3]
        dsets2[x1] = DataFolder(dir_list, post_fix, num_channels, data_transforms[x1])
    # 加载DeepLabv3_plus时，由于batchnorm层需要大于一个样本去计算其中的参数，
    # 解决方法是将dataloader的一个丢弃参数设置为true
    train_loader = DataLoader(dsets['train'], batch_size=opt.train['batch_size'], shuffle=True,
                              num_workers=opt.train['workers'], drop_last=True)
    train_loader1 = DataLoader(dsets2['train1'], batch_size=opt.train['batch_size'], shuffle=True,
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
        train_results = train(train_loader, train_loader1, model, optimizer, criterion, epoch)
        train_loss, train_loss_ce, train_loss_var, train_pixel_acc, train_iou = train_results

        # evaluate on validation set
        with torch.no_grad():
            val_loss, val_pixel_acc, val_iou = validate(val_loader, model, criterion)
            val_loss1, val_pixel_acc1, val_iou1 = validate(val_loader1, model, criterion)

        # check if it is the best accuracy
        is_best = val_iou > best_iou
        best_iou = max(val_iou, best_iou)

        # is_second = val_iou >= 0.84
        if (val_iou >= 0.82):
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


def train(train_loader, train_loader1, model, optimizer, criterion, epoch):
    # list to store the average loss and iou for this epoch
    results = utils.AverageMeter(5)

    # switch to train mode
    model.train()
    label_iter = iter(train_loader1)

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
        try:
            input1 = next(label_iter)
        except StopIteration:
            label_iter = iter(train_loader1)
            input1 = next(label_iter)

        target = F.one_hot(target, num_classes=3)
        # print(target1.shape)
        target_one_hot0 = target[:, :, :, :, 0]
        target_one_hot1 = target[:, :, :, :, 1]
        target_one_hot2 = target[:, :, :, :, 2]
        # print(target_one_hot0.shape)
        input_var = input.cuda()
        target_var = target1.cuda()

        # compute output

        # output, out1, out2 = model(input_var)
        output, dsv1,dsv2,dsv3 = model(input_var)


        # output_u = torch.softmax(output_u,dim=1)
        # unsup_loss = entropy_loss(output_u, C=3)
        # print(unsup_loss)


        log_prob_maps = F.log_softmax(output, dim=1)
        loss_map = criterion(log_prob_maps, target_var)
        loss_map *= weight_map_var
        loss_CE = loss_map.mean()

        log_prob_maps = F.log_softmax(dsv1, dim=1)
        loss_map = criterion(log_prob_maps, target_var)
        loss_map *= weight_map_var
        loss_CE1 = loss_map.mean()

        log_prob_maps = F.log_softmax(dsv2, dim=1)
        loss_map = criterion(log_prob_maps, target_var)
        loss_map *= weight_map_var
        loss_CE2 = loss_map.mean()

        log_prob_maps = F.log_softmax(dsv3, dim=1)
        loss_map = criterion(log_prob_maps, target_var)
        loss_map *= weight_map_var
        loss_CE3 = loss_map.mean()

        loss_sum_CE = (loss_CE + loss_CE1 + loss_CE2 + loss_CE3)/4

        output_u, dsv1_u, dsv2_u, dsv3_u = model(input1[0].cuda())

        outputs_u_soft = torch.softmax(output_u, dim=1)
        dsv1_u_soft = torch.softmax(dsv1_u, dim=1)
        dsv2_u_soft = torch.softmax(dsv2_u, dim=1)
        dsv3_u_soft = torch.softmax(dsv3_u, dim=1)

        preds = (outputs_u_soft + dsv1_u_soft +
                 dsv2_u_soft + dsv3_u_soft) / 4

        variance_main = torch.sum(kl_distance(torch.log(outputs_u_soft), preds), dim=1, keepdim=True)
        exp_variance_main = torch.exp(-variance_main)

        variance_aux1 = torch.sum(kl_distance(
            torch.log(dsv1_u_soft), preds), dim=1, keepdim=True)
        exp_variance_aux1 = torch.exp(-variance_aux1)

        variance_aux2 = torch.sum(kl_distance(
            torch.log(dsv2_u_soft), preds), dim=1, keepdim=True)
        exp_variance_aux2 = torch.exp(-variance_aux2)

        variance_aux3 = torch.sum(kl_distance(
            torch.log(dsv3_u_soft), preds), dim=1, keepdim=True)
        exp_variance_aux3 = torch.exp(-variance_aux3)

        # consistency_weight = get_current_consistency_weight(iter_num // 150)
        consistency_dist_main = (
                                        preds - outputs_u_soft) ** 2

        consistency_loss_main = torch.mean(
            consistency_dist_main * exp_variance_main) / (torch.mean(exp_variance_main) + 1e-8) + torch.mean(
            variance_main)

        consistency_dist_aux1 = (
                                        preds - dsv1_u_soft) ** 2
        consistency_loss_aux1 = torch.mean(
            consistency_dist_aux1 * exp_variance_aux1) / (torch.mean(exp_variance_aux1) + 1e-8) + torch.mean(
            variance_aux1)

        consistency_dist_aux2 = (
                                        preds - dsv2_u_soft) ** 2
        consistency_loss_aux2 = torch.mean(
            consistency_dist_aux2 * exp_variance_aux2) / (torch.mean(exp_variance_aux2) + 1e-8) + torch.mean(
            variance_aux2)

        consistency_dist_aux3 = (
                                        preds - dsv3_u_soft) ** 2
        consistency_loss_aux3 = torch.mean(
            consistency_dist_aux3 * exp_variance_aux3) / (torch.mean(exp_variance_aux3) + 1e-8) + torch.mean(
            variance_aux3)

        consistency_loss = (consistency_loss_main + consistency_loss_aux1 +
                            consistency_loss_aux2 + consistency_loss_aux3) / 4




        if opt.train['alpha'] != 0:
            prob_maps = F.softmax(output, dim=1)
            target_labeled = torch.zeros(target1.size()).long()

            # label instances in target
            for k in range(target1.size(0)):
                target_labeled[k] = torch.from_numpy(measure.label(target1[k].numpy() == 1))
                # utils.show_figures((target[k].numpy(), target[k].numpy()==1, target_labeled[k].numpy()))
            loss_var = criterion_var(prob_maps, target_labeled.cuda())
            # print("consistency_loss:",consistency_loss)
            loss = loss_sum_CE + opt.train['alpha'] * loss_var +  consistency_loss

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

        # for b in range(input.size(0)):
        #     utils.show_figures((input[b, 0, :, :].numpy(), target[b,0,:,:].numpy(), weight_map[b, :, :]))

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

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def entropy_loss(p, C=3):
    # p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1) / \
        torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent




if __name__ == '__main__':
    main()
