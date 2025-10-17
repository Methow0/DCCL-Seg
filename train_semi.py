import torch.optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.utils.data
import os
import shutil
from skimage import measure
import skimage
import logging
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
import matplotlib.pylab as plt
from torchvision.utils import make_grid
from skimage import measure, io
from augmention import generate_unsup_data
from CE_Net import CE_Net_,Our_Semic_Seg
from torch.nn import functional as F
import numpy as np
import utils
from data_folder import DataFolder
from hausdorff_loss import HausdorffERLoss
from options_semi import Options
from my_transforms import get_transforms
from loss import LossVariance
from torch import nn
from copy import deepcopy
from sklearn.cluster import KMeans
import cv2
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg

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

    model = Our_Semic_Seg(3, 3)
    ema_model = deepcopy(model)
    gta_model = deepcopy(model)
    model = nn.DataParallel(model)
    model = model.cuda()
    ema_model = ema_model.cuda()
    gta_model = gta_model.cuda()
    torch.backends.cudnn.benchmark = True

    # ----- define optimizer ----- #
    optimizer = torch.optim.Adam(model.parameters(), opt.train['lr'], betas=(0.9, 0.99),
                                 weight_decay=opt.train['weight_decay'])

    # ----- define criterion ----- #
    global mseloss, criterion_hau
    mseloss = torch.nn.MSELoss(reduction='mean').cuda()
    criterion = torch.nn.NLLLoss(reduction='none').cuda()
    criterion_hau = HausdorffERLoss()
    if opt.train['alpha'] > 0:
        logger.info('=> Using variance term in loss...')
        global criterion_var
        criterion_var = LossVariance()

    data_transforms = {'train': get_transforms(opt.transform['train']),
                       'train_sdm': get_transforms(opt.transform['train_sdm']),
                       'train1': get_transforms(opt.transform['train1']),
                       'valA': get_transforms(opt.transform['val']),
                       'valB': get_transforms(opt.transform['val'])}


    # ----- load data ----- #
    dsets = {}
    for x in ['train', 'valA', 'valB']:
        img_dir = '{:s}/{:s}'.format(opt.train['img_dir'], x)
        target_dir = '{:s}/{:s}'.format(opt.train['label_dir'], x)
        target_dir1 = '{:s}/{:s}'.format(opt.train['label_dir'], x + '_sdm')
        weight_map_dir = '{:s}/{:s}'.format(opt.train['weight_map_dir'], x)
        dir_list = [img_dir, weight_map_dir, target_dir1, target_dir]
        if opt.dataset == 'MultiOrgan':
            post_fix = ['weight.png', 'label.png']
        else:
            post_fix = ['anno_weight.png', 'anno.bmp', 'anno_label.png']
        num_channels = [3, 1, 1, 3]
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
        train_results = train(train_loader, train_loader1, model, ema_model, gta_model, optimizer, criterion, epoch)
        train_loss, train_loss_ce, train_loss_var, train_pixel_acc, train_iou = train_results

        # evaluate on validation set
        with torch.no_grad():
            val_loss, val_pixel_acc, val_iou = validate(val_loader, model, criterion)
            val_loss1, val_pixel_acc1, val_iou1 = validate(val_loader1, model, criterion)

        # check if it is the best accuracy
        is_best = val_iou > best_iou
        best_iou = max(val_iou, best_iou)

        # is_second = val_iou >= 0.80
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


def train(train_loader, train_loader1, model, ema_model, gta_model,optimizer, criterion, epoch):
    ite = 0
    # Loss_list = list()

    # list to store the average loss and iou for this epoch
    results = utils.AverageMeter(5)
    # switch to train mode
    model.train()
    ema_model.train()
    label_iter = iter(train_loader1)

    for i, sample in enumerate(train_loader):
        ite += 1
        input, weight_map,targetdis, target = sample
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


        input_var = input.cuda()
        target_var = target1.cuda()
        input_var1 = input1[0].cuda()


        # compute output

        # output, out1, out2 = model(input_var)
        output_ema,out_sdm,output_rep = ema_model(input_var1)
        output_ema3 = F.softmax(output_ema, dim=1)
        # index2 = torch.argmax(output_ema3, dim=1)
        # sd2 = torch.where(index2 == 1, 255, 0)
        # featuremap_visual(sd2.unsqueeze(1), i=i, out_dir='/root/Code/FullNet/result/pseudo', num_ch=-1)
        target_sdm = compute_sdf(targetdis, out_sdm.shape)
        target_sdm = torch.tensor(target_sdm)

        pred_sdm = torch.tensor(compute_sdf(output_ema[:, 1:2, :, :].cpu(), out_sdm.shape))
        # mse_loss1 = consistency_loss(pred_sdm.to(torch.float32).cuda(), target_sdm.to(torch.float32).cuda())
        mse_loss = consistency_loss(out_sdm.to(torch.float32).cuda(), pred_sdm.to(torch.float32).cuda())


        logits_u_aug, label_u_aug = torch.max(output_ema3,dim=1)



        input_var1_aug, label_u_aug, logits_u_aug = generate_unsup_data(input_var1,label_u_aug.clone(),logits_u_aug.clone(),mode="classmix")
        output_l, out_sdm1, out_rep, output_u, out_sdm2 = model(input_var,input_var1_aug)

        pred_sdm1 = torch.tensor(compute_sdf(output_l[:, 1:2, :, :].cpu(), out_sdm1.shape))
        mse_loss1 = consistency_loss(pred_sdm1.to(torch.float32).cuda(), out_sdm1.to(torch.float32).cuda())
        mse_loss2 = consistency_loss(out_sdm1.to(torch.float32).cuda(), target_sdm.to(torch.float32).cuda())

        pred_sdm2 = torch.tensor(compute_sdf(output_u[:, 1:2, :, :].cpu(), out_sdm2.shape))
        mse_loss3 = consistency_loss(out_sdm2.to(torch.float32).cuda(), out_sdm.to(torch.float32).cuda())

        mse_loss4 = consistency_loss(pred_sdm2.to(torch.float32).cuda(), out_sdm2.to(torch.float32).cuda())

        loss_total_mse = mse_loss + mse_loss1 + mse_loss2 + mse_loss3 + mse_loss4



        loss_CE1 = compute_unsupervised_loss_conf_weight(label_u_aug,100,output_u)

        log_prob_maps = F.log_softmax(output_l, dim=1)
        loss_map = criterion(log_prob_maps, target_var)
        loss_map *= weight_map_var
        loss_CE = loss_map.mean()
        # print("loss_CE:",loss_CE)

        # InNce_loss = criterion_loss(output_rep,out_rep,0.1,8,1.0,'hard')
        # print("InNce_loss:",InNce_loss/10)




        if opt.train['alpha'] != 0:
            prob_maps = F.softmax(output_l, dim=1)

            # label instances in target
            target_labeled = torch.zeros(target1.size()).long()
            for k in range(target1.size(0)):
                target_labeled[k] = torch.from_numpy(measure.label(target1[k].numpy() == 1))
                # utils.show_figures((target[k].numpy(), target[k].numpy()==1, target_labeled[k].numpy()))
            loss_var = criterion_var(prob_maps, target_labeled.cuda())
            # print(2*loss_total_mse)
            loss = loss_CE + opt.train['alpha'] * loss_var + loss_CE1 + 1.2*loss_total_mse


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
        # for param in model.named_parameters():
        #     print(param[0])

        model, ema_model = update_ema_variables(model, ema_model, 0.999)

        del input_var, output_l, target_var, log_prob_maps, loss

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




def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask

def criterion_loss(out_1, out_2, tau_plus, batch_size, beta, estimator):
    # neg score

    out_1 = out_1.view(8,256)
    out_2 = out_2.view(8,256)
    out = torch.cat([out_1, out_2], dim=0)
    neg = torch.exp(torch.mm(out, out.t().contiguous()) / 0.5)
    old_neg = neg.clone()
    mask = get_negative_mask(batch_size).cuda()
    neg = neg.masked_select(mask).view(2 * batch_size, -1)

    # pos score
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / 0.5)
    pos = torch.cat([pos, pos], dim=0)

    # negative samples similarity scoring
    if estimator == 'hard':
        N = batch_size * 2 - 2
        imp = (beta * neg.log()).exp()
        reweight_neg = (imp * neg).sum(dim=-1) / imp.mean(dim=-1)
        Ng = (-tau_plus * N * pos + reweight_neg) / (1 - tau_plus)
        # constrain (optional)
        Ng = torch.clamp(Ng, min=N * np.e ** (-1 / 0.5))
    elif estimator == 'easy':
        Ng = neg.sum(dim=-1)
    else:
        raise Exception('Invalid estimator selected. Please use any of [hard, easy]')

        # contrastive loss
    loss = (- torch.log((pos+1e-12) / (pos + Ng+1e-12))).mean()

    return loss

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
    for b in range(out_shape[0]): # batch size
        posmask = img_gt[b].astype(np.bool)
        if posmask.any():
            negmask = ~posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            # 归一化，分割区域内部为负，外部为正
            sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)+1e-12) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis)+1e-12)
            # 边界置零
            sdf[boundary==1] = 0
            normalized_sdf[b] = sdf
            # assert np.min(sdf) == -1.0, print(np.min(posdis), np.max(posdis), np.min(negdis), np.max(negdis))
            # assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))
    return normalized_sdf

def compute_unsupervised_loss_conf_weight(target, percent, pred_teacher):
    batch_size, num_class, h, w = pred_teacher.shape

    with torch.no_grad():
        # drop pixels with high entropy
        prob = torch.softmax(pred_teacher, dim=1)
        conf, ps_label = torch.max(prob, dim=1)
        conf = conf.detach()
        conf_thresh = np.percentile(
            conf[target < 1 ].cpu().numpy().flatten(), 100 - percent
        )

        # print(conf_thresh)
        thresh_mask = conf.le(conf_thresh).bool() * (target <1).bool()
        # print(thresh_mask)
        conf[thresh_mask] = 0
        target[thresh_mask] = 1
        # for k in range(0,8):
        #     cv2.imwrite('/root/Code/FullNet/result/repseudo/prob_inside_{:s}.png'.format(str(k)),
        #                 target[k].detach().cpu().numpy().astype(np.uint8))

        # print("target.min:",target.min())
        # print("target.max:",target.max())
        weight = batch_size * h * w / (torch.sum(target < 1 ) + 1e-6)

    loss_ = weight * F.cross_entropy(pred_teacher, target, ignore_index=1, reduction='none')  # [10, 321, 321]
    conf = (conf + 1.0) / (conf + 1.0).sum() * (torch.sum(target <1 ) + 1e-6)
    loss = torch.mean(conf * loss_)
    return loss

def validate(val_loader, model, criterion):
    # list to store the losses and accuracies: [loss, pixel_acc, iou ]
    results = utils.AverageMeter(3)

    # switch to evaluate mode
    model.eval()

    for i, sample in enumerate(val_loader):
        input, weight_map, targetdis, target = sample
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

def featuremap_visual(feature, i,out_dir, save_feature=True, show_feature=True, feature_title=None, num_ch=-1, nrow=8, padding=10, pad_value=1):
    # feature = feature.detach().cpu()
    b, c, h, w = feature.shape
    for w in range(0,8):
        feature_t = feature[w]
        # feature = feature.unsqueeze(1)

        if c > num_ch > 0:
            feature_t = feature_t[:num_ch]

        # img = torchvision.utils.make_grid(feature, nrow=nrow, padding=padding, pad_value=pad_value)
        feature_t = feature_t.squeeze(0)
        img = feature_t.detach().cpu().numpy()
        images = img
        cv2.imwrite(out_dir +'/{:s}_prob_inside_{:s}.png'.format(str(i),str(w)), images.astype(np.uint8))


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


def update_ema_variables(model, model_teacher, alpha):
    # Use the true average until the exponential average is more correct
    # alpha = min(1.0 - 1.0 / float(global_step + 1), alpha)
    for param_t, param in zip(model_teacher.parameters(), model.named_parameters()):
        if 'decoder1'and 'decoder1_1' and 'decoder1_2' not in param[0]:
            param_t.data.mul_(alpha).add_(param[1].data, alpha=1 - alpha)

    return model, model_teacher


def consistency_loss(logits_w1, logits_w2):
       # logits_w2 = logits_w2.detach()
       # assert logits_w1.size() == logits_w2.size()
    return F.mse_loss(logits_w1, logits_w2,reduction='mean')


if __name__ == '__main__':
    main()