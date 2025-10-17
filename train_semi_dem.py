import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import torch.optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.utils.data
import random
import shutil
from skimage import measure
import skimage
import logging
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
from torchvision.utils import make_grid
from skimage import measure, io
from augmention import generate_unsup_data
from CE_Net import New_Semic_Seg
from torch.nn import functional as F
import numpy as np
import utils
from data_folder import DataFolder
from hausdorff_loss import HausdorffERLoss
from options_semi import Options
from my_transforms import get_transforms
from loss import LossVariance, dice_loss, FlowLoss
from torch import nn
from copy import deepcopy
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import cv2
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
from torch.cuda.amp import GradScaler, autocast
from NVP import RNVP
from distributions import FastGMM
import yaml
import argparse
from utils2 import clip_grad_norm, bits_per_dim
from functools import partial
from attack import attack
from scipy.signal import find_peaks
print(torch.cuda.get_device_name(0)) 
# device = torch.device("cuda:1")
# print(torch.cuda.get_device_name(0))


parser = argparse.ArgumentParser(description="Semi-Supervised Semantic Segmentation")
parser.add_argument("--config", type=str, default="config.yaml")
# parser.add_argument('--batch-size', type=int, default=16, help='input batch size for training')
parser.add_argument('--alpha', type=float, default=1, help='The weight for the variance term in loss')
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--save-dir', type=str, default='./experiments/GlaS_0.125/newcrop/train_2025-5-25', help='directory to save training results')
parser.add_argument('--gpu', type=list, default=[], help='GPUs for training')
def to_status(m, status):
    if hasattr(m, 'batch_type'):
        m.batch_type = status

to_clean_status = partial(to_status, status='clean')
to_adv_status = partial(to_status, status='adv')
to_mix_status = partial(to_status, status='mix')
to_warm_status = partial(to_status, status='warm_up')
device = torch.device("cuda:1")


def set_seed(seed=42):
    print(f"[INFO] Setting seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 开启这个会影响可复现性

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(42)


def main():
    global opt, best_iou, num_iter, tb_writer, logger, logger_results, args, cfg, scaler_nf, scaler
    set_seed(42)
    args = parser.parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    scaler = GradScaler()
    scaler_nf = GradScaler()
    opt = Options(isTrain=True)
    opt.parse()
    opt.save_options()
    torch.backends.cudnn.enabled = False

    tb_writer = SummaryWriter('{:s}/tb_logs'.format(opt.train['save_dir']))


    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # set up logger
    logger, logger_results = setup_logging(opt)
    opt.print_options(logger)

    model = New_Semic_Seg(3,3)
    teacher_model = deepcopy(model)
    for p in teacher_model.parameters():
        p.requires_grad = False

    means = torch.randn(cfg['flow']['n_components'], cfg['flow']['input_dims'])   
    nf_model = RNVP(cfg, means, learnable_mean=True)
    nf_model= nf_model.cuda()

    # model = nn.DataParallel(model,device_ids=[0])
    model = model.cuda()
    teacher_model = teacher_model.cuda()


    torch.backends.cudnn.benchmark = True

    # ----- define optimizer ----- #
    optimizer = torch.optim.Adam(model.parameters(), opt.train['lr'], betas=(0.9, 0.99),
                                 weight_decay=opt.train['weight_decay'])
    optimizer_nf = torch.optim.Adam([{'params':nf_model.parameters()}], lr=cfg['flow']['lr'], betas=(0.5, 0.9))

    # ----- define criterion ----- #
    global mseloss, criterion_hau, loss_flow 
    prior = FastGMM(nf_model.means)

    loss_flow = FlowLoss(prior)
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
    # train_loader = DataLoader(dsets['train'], batch_size=opt.train['batch_size'], shuffle=True,
    #                           num_workers=opt.train['workers'], drop_last=False,
    #                           worker_init_fn=seed_worker, generator=g)
    # train_loader1 = DataLoader(dsets2['train1'], batch_size=opt.train['batch_size'], shuffle=True,
    #                            num_workers=opt.train['workers'], drop_last=False,
    #                            worker_init_fn=seed_worker, generator=g)

    train_loader = DataLoader(dsets['train'], batch_size=4, shuffle=True,
                              num_workers=opt.train['workers'], drop_last=False,
                              worker_init_fn=seed_worker, generator=g)
    train_loader1 = DataLoader(dsets2['train1'], batch_size=12, shuffle=True,
                               num_workers=opt.train['workers'], drop_last=False,
                               worker_init_fn=seed_worker, generator=g)

    val_loader = DataLoader(dsets['valB'], batch_size=1, shuffle=False,
                            num_workers=opt.train['workers'], drop_last=False,
                            worker_init_fn=seed_worker, generator=g)
    val_loader1 = DataLoader(dsets['valA'], batch_size=1, shuffle=False,
                             num_workers=opt.train['workers'], drop_last=False,
                             worker_init_fn=seed_worker, generator=g)
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

    # === 初始化部分（放到脚本开头） ===
    START_RECORD = 50  # 预热轮数
    PEAK_HEIGHT = 0.75  # 峰值最小高度阈值
    PEAK_DISTANCE = 5  # 峰值最小间隔
    iou_history_student = []  # 记录从 START_RECORD 之后的 weighted_iou
    iou_history_teacher = []
    candidate_models = []  # 保存所有候选模型的信息
    save_dir = opt.train['save_dir']
    cp_dir = os.path.join(save_dir, 'checkpoints')
    os.makedirs(cp_dir, exist_ok=True)

    # ----- training and validation ----- #
    for epoch in range(opt.train['start_epoch'], opt.train['num_epochs']):
        # train for one epoch or len(train_loader) iterations
        logger.info('Epoch: [{:d}/{:d}]'.format(epoch + 1, opt.train['num_epochs']))
        train_results = train(train_loader, train_loader1, model, teacher_model, nf_model, optimizer, optimizer_nf, criterion, epoch)
        train_loss, train_loss_ce, train_loss_var, train_unsup_loss, train_unsup_loss_pt, train_consistency_loss,train_nf_loss, train_pixel_acc, train_iou = train_results

        # evaluate on validation set
        with torch.no_grad():
            # 学生模型进行验证
            val_loss, val_pixel_acc, val_iou = validate(val_loader, model, criterion,True)
            val_loss1, val_pixel_acc1, val_iou1 = validate(val_loader1, model, criterion,True)

            # 教师模型进行验证
            val_loss_t, val_pixel_acc_t, val_iou_t = validate(val_loader, teacher_model, criterion,False)
            val_loss1_t, val_pixel_acc1_t, val_iou1_t = validate(val_loader1, teacher_model, criterion,False)

            weighted_iou_student = 0.75 * val_iou + 0.25 * val_iou1
            weighted_iou_teacher = 0.75 * val_iou_t + 0.25 * val_iou1_t

            # 记录较好的模型（教师 or 学生）
            if weighted_iou_student >= weighted_iou_teacher:
                # 学生更好或相等，保留学生模型
                candidate_models.append({
                    'epoch': epoch + 1,
                    'weighted_iou': weighted_iou_student,
                    'model_state_dict': model.state_dict()
                })
                logger.info(
                    f"[Epoch {epoch + 1}] Student model selected (weighted_iou: {weighted_iou_student:.4f} >= teacher: {weighted_iou_teacher:.4f})")
            else:
                # 教师更好，保留教师模型
                candidate_models.append({
                    'epoch': epoch + 1,
                    'weighted_iou': weighted_iou_teacher,
                    'model_state_dict': teacher_model.state_dict()
                })
                logger.info(
                    f"[Epoch {epoch + 1}] Teacher model selected (weighted_iou: {weighted_iou_teacher:.4f} > student: {weighted_iou_student:.4f})")
            iou_history_student.append(weighted_iou_student)
            iou_history_teacher.append(weighted_iou_teacher)

        # save the training results to txt files
        logger_results.info('{:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
                            .format(epoch + 1, train_loss, train_loss_ce, train_loss_var, train_pixel_acc,
                                    train_iou, val_loss, val_pixel_acc, val_iou))
        # tensorboard logs
        tb_writer.add_scalars('epoch_losses',
                              {'train_loss': train_loss, 'train_loss_ce': train_loss_ce,
                               'train_unsup_loss': train_unsup_loss, 'train_unsup_loss_pt': train_unsup_loss_pt,
                               'train_consistency_loss': train_consistency_loss, 'train_nf_loss': train_nf_loss,
                               'train_loss_var': train_loss_var, 'val_loss': val_loss}, epoch)

        tb_writer.add_scalars('student_epoch_accuracies',
                              {'train_pixel_acc': train_pixel_acc, 'train_iou': train_iou,
                               'val_pixel_accB': val_pixel_acc, 'val_iouB': val_iou,
                               'val_pixel_accA': val_pixel_acc1, 'val_iouA': val_iou1}, epoch)

        tb_writer.add_scalars('teacher_epoch_accuracies',
                              {'train_pixel_acc': train_pixel_acc, 'train_iou': train_iou,
                               'val_pixel_accB': val_pixel_acc_t, 'val_iouB': val_iou_t,
                               'val_pixel_accA': val_pixel_acc1_t, 'val_iouA': val_iou1_t}, epoch)

        tb_writer.add_scalars('iou_history_weighted_iou',
                              {'train_pixel_acc': train_pixel_acc, 'train_iou': train_iou,
                               'weighted_iou_student': weighted_iou_student,
                               'weighted_iou_teacher': weighted_iou_teacher}, epoch)


    # === 训练结束后，自动检测曲线峰值数量作为 K ===
    y = np.array(iou_history_student)
    peaks, props = find_peaks(y, height=PEAK_HEIGHT, distance=PEAK_DISTANCE)
    K = len(peaks)
    print(f"\nDetected {K} significant peaks (height>={PEAK_HEIGHT}, dist>={PEAK_DISTANCE}) -> Top-K = {K}")

    # === 保留 Top-K 模型 ===
    topk_models = sorted(candidate_models, key=lambda x: x['weighted_iou'], reverse=True)[:K]
    for i, m in enumerate(topk_models):
        model_name = f"topk_{i + 1:02d}_epoch{m['epoch']:03d}_iou{m['weighted_iou']:.4f}.pth"
        model_path = os.path.join(cp_dir, model_name)
        torch.save({
            'epoch': m['epoch'],
            'state_dict': m['model_state_dict'],
            'weighted_iou': m['weighted_iou'],
            'optimizer': optimizer.state_dict()
        }, model_path)

def ramp_up_weight(epoch, max_epoch=1000, max_weight=1.0):
    if epoch < 100:
        return max_weight * (epoch / 100)  # 前100轮线性增长
    else:
        return max_weight


def train(train_loader, train_loader1, model, teacher_model, nf_model, optimizer, optimizer_nf,criterion, epoch):
    ite = 0
    # Loss_list = list()

    # list to store the average loss and iou for this epoch
    results = utils.AverageMeter(9)
    # switch to train mode

    label_iter = iter(train_loader1)

    for i, sample in enumerate(train_loader):
        ite += 1
        input, weight_map, target = sample
        weight_map = weight_map.float().div(20)
        if weight_map.dim() == 4:
            weight_map = weight_map.squeeze(1)
        weight_map_var = weight_map.cuda()
        if torch.max(target) == 255:
            target = target / 255
        if target.dim() == 4:
            target = target.squeeze(1)
        try:
            input1 = next(label_iter)
        except StopIteration:
            label_iter = iter(train_loader1)
            input1 = next(label_iter)

        input_var = input.cuda()
        target_var = target.cuda()
        input_var1 = input1[0].cuda()
 

        # compute teacher_model,teacher_model predicts on all data
        teacher_model.train()
        with torch.no_grad():
            out_labeled,out_unlabeled,output_rep = teacher_model(input_var,input_var1,label=None, nf_model=None, loss_flow=None, cfg=None, eps=0, adv=False)
            # Get predictions of original unlabeled data
            # output_ema_unlabeled = output_ema_all[b_labeled:]

            output_ema_u = F.softmax(out_unlabeled, dim=1)
            logits_u_aug, label_u = torch.max(output_ema_u,dim=1)

        if random.uniform(0, 1) < 0.5:
            input_var1_aug, label_u_aug, logits_u_aug = generate_unsup_data(input_var1,label_u.clone(),logits_u_aug.clone(),mode="cutmix")
        else:
            input_var1_aug, label_u_aug, logits_u_aug = input_var1.clone(),label_u.clone(),logits_u_aug.clone()
        ignore_mask = ((logits_u_aug < 0.6).long()) * 255




        # compute student_model
        model.train()
        # label_u[label_u == 0] = 255
        unsup_loss_pt=0.0
        consistency_loss=0.0
        label_st = label_u_aug.clone()
        label_st[label_st == 0] = 255
        with autocast():
            if epoch >= cfg['trainer']['nf_start_epoch'] + 1:
                model.apply(to_mix_status)

                out_labeled,out_unlabeled, out_all_unlabeled_pt = model(input_var, input_var1_aug,label_st, nf_model, loss_flow, cfg, eps=cfg['adv']['eps'],
                                         adv=True)
                # out1 = torch.softmax(out_unlabeled, dim=1)
                # out2 = torch.softmax(out_all_unlabeled_pt, dim=1)
                # diff = (out1 - out2).abs().mean()
                # print("diff changes:", diff.item())


                if epoch % 100 == 0:
                    visualize_tsne_preds(out_unlabeled, out_all_unlabeled_pt,
                                                   save_path=f'/root/Code/FullNet/data/t_SNE/tsne_epoch_{epoch}')

                if epoch < 200:
                    percent=95
                elif epoch < 500:
                    percent=90
                elif epoch < 800:
                    percent=85
                else:
                    percent=80

                valid_mask = (ignore_mask != 255).float()  # [B, H, W]

                # pred_l_s_large = output_all[:b_labeled]  # 有标签部分
                # pred_u_aug_s_large, pred_u_aug_s_large_pt = output_all[b_labeled:].chunk(2)
                # pred_u_aug_s_large = output_all[B:]  # 无标签（增强）
                # pred_u_aug_s_large_pt = output_all[2 * B:]  # 无标签（对抗增强）

                # print("pred_l_s_large.shape", pred_l_s_large.shape)
                # print("pred_u_aug_s_large.shape", pred_u_aug_s_large.shape)
                # print("pred_u_aug_s_large_pt.shape", pred_u_aug_s_large_pt.shape)

                log_prob_maps = F.log_softmax(out_labeled, dim=1)
                loss_map = criterion(log_prob_maps, target_var)
                loss_map = loss_map * weight_map_var
                loss_CE = loss_map.mean()
                unsup_loss = compute_unsupervised_loss_conf_weight(
                    label_u_aug.clone(),percent,out_unlabeled)


                unsup_loss_pt = compute_unsupervised_loss_conf_weight(
                    label_u_aug.clone(),percent,out_all_unlabeled_pt)

                # consistency_loss = F.kl_div(
                #     F.log_softmax(out_all_unlabeled_pt, dim=1),
                #     F.softmax(out_unlabeled.detach(), dim=1),
                #     reduction='batchmean'
                # )
                consistency_loss = compute_consistency_loss(out_unlabeled, out_all_unlabeled_pt, mask=valid_mask)

                lambda_u = ramp_up_weight(epoch, max_epoch=1000, max_weight=1.0)  # 无标签损失最大权重为1.0
                lambda_pt = 0.5 * lambda_u  # 扰动无标签损失的权重通常略小一点
                lambda_cons = 0.3 * lambda_u  # 或者先用更小值如 0.1 开始


                loss = loss_CE + lambda_u * unsup_loss + lambda_pt * unsup_loss_pt + lambda_cons * consistency_loss

                # loss = loss_CE + unsup_loss + 0.5*unsup_loss_pt
                # loss = loss_CE
            else:
                output_labeled, output_unlabeled, _= model(input_var, input_var1_aug,label=None, nf_model=None, loss_flow=None, cfg=None, eps=0, adv=False)

                unsup_loss = compute_unsupervised_loss_conf_weight(label_u_aug,100,output_unlabeled)

                log_prob_maps = F.log_softmax(output_labeled,dim=1)
                loss_map = criterion(log_prob_maps, target_var)
                loss_map *= weight_map_var
                loss_CE = loss_map.mean()
                lambda_u = ramp_up_weight(epoch, max_epoch=1000, max_weight=1.0)  # 无标签损失最大权重为1.0

                loss= loss_CE + lambda_u * unsup_loss


        if opt.train['alpha'] != 0:
            if epoch >= cfg['trainer']['nf_start_epoch'] + 1:
                prob_maps = F.softmax(out_labeled, dim=1)
            else:
                prob_maps = F.softmax(output_labeled,dim=1)
    
                # label instances in target
            target_labeled = torch.zeros(target.size()).long()
            for k in range(target.size(0)):
                target_labeled[k] = torch.from_numpy(measure.label(target[k].numpy() == 1))
                loss_var = criterion_var(prob_maps, target_labeled.cuda())
            loss_total = loss + opt.train['alpha'] * loss_var


        else:
            loss_var = torch.ones(1) * -1
            loss_total = loss



        ###############################Training of Normalizing FLow ################################
        nf_loss=0.0
        if epoch >= cfg['trainer']['nf_start_epoch']:

            nf_model.train()
            with autocast():
                label_l_small = F.interpolate(target_var.unsqueeze(1).float(), size=output_rep.shape[2:],
                                              mode="nearest").squeeze(1)

                label_u_small = F.interpolate(label_u_aug.unsqueeze(1).float(), size=output_rep.shape[2:],
                                              mode="nearest").squeeze(1)
                ignore_mask_small = F.interpolate(ignore_mask.unsqueeze(1).float(), size=output_rep.shape[2:],
                                                  mode="nearest").squeeze(1).long()

                label_u_small[ignore_mask_small == 255] = 255
                # label_u_small[ignore_mask_small != 255] = 300

                b, c, h, w = output_rep.size()  # pred_all, rep_all, fts_all
                total_n = int(b * h * w)

                fts_all = output_rep.detach().permute(0, 2, 3, 1).reshape(total_n, c)
                label_nf = torch.cat([label_l_small.long(), label_u_small.long()], dim=0)
                label_nf = label_nf.detach().clone().view(-1)  # n

                valid_map = (label_nf != 255)  # filter out ignored pixels
                valid_fts_num = int(valid_map.sum())
                valid_fts = fts_all[valid_map]
                valid_label = label_nf[valid_map]


                sample_num = min(20 * 1024, valid_fts.size(0))
                sample_idx = torch.randperm(valid_fts.size(0))[:sample_num]
                input_nf_sample = valid_fts[sample_idx]  # [sample_num, c]
                label_nf_sample = valid_label[sample_idx]

                # add noise
                input_nf_sample += cfg['flow']['noise'] * torch.randn_like(input_nf_sample)
                # print("===================")
                # print(input_nf_sample.shape)
                z, log_jac_det = nf_model(input_nf_sample)
                nf_loss, ll, sldj = loss_flow(z, sldj=log_jac_det, y=label_nf_sample)

            optimizer_nf.zero_grad()
            scaler_nf.scale(nf_loss).backward()
            scaler_nf.unscale_(optimizer_nf)
            # Clip the gradient
            clip_grad_norm(optimizer_nf, cfg['flow']['grad_clip'])
            scaler_nf.step(optimizer_nf)
            scaler_nf.update()
            nf_model.eval()
            maybe_freeze_nf_model_by_loss(epoch, nf_model, nf_loss.item())

        pred = np.argmax(log_prob_maps.data.cpu().numpy(), axis=1)
        metrics = utils.accuracy_pixel_level(pred, target.numpy())
        pixel_accu, iou = metrics[0], metrics[1]

        result = [loss, loss_CE, loss_var, unsup_loss, unsup_loss_pt, consistency_loss, nf_loss, pixel_accu, iou]
        results.update(result, input.size(0))


        # compute gradient and do SGD step

        optimizer.zero_grad()
        scaler.scale(loss_total).backward()
        scaler.step(optimizer)
        scaler.update()

        model, teacher_model = update_ema_variables(model, teacher_model, epoch=epoch,base_alpha=0.99)

        del input_var, target_var, log_prob_maps, loss_total

        if i % opt.train['log_interval'] == 0:
            logger.info('\tIteration:[{:d}/{:d}]'
                        '\tLoss {r[0]:.4f}'
                        '\tLoss_CE {r[1]:.4f}'
                        '\tLoss_var {r[2]:.4f}'
                        '\tunsup_loss {r[3]:.4f}'
                        '\tunsup_loss_pt {r[4]:.4f}'
                        '\tmse {r[5]:.4f}'
                        '\tnf_loss {r[6]:.4f}'
                        '\tpixel_accu {r[7]:.4f}'
                        '\tIoU {r[8]:.4f}'.format(i, len(train_loader), r=results.avg))

    logger.info('\t => Train_Avg:Loss_total {r[0]:.4f}'
                '\tLoss_CE {r[1]:.4f}'
                '\tLoss_var {r[2]:.4f}'
                '\tunsup_loss {r[3]:.4f}'
                '\tunsup_loss_pt {r[4]:.4f}'
                '\tmse {r[5]:.4f}'
                '\tnf_loss {r[6]:.4f}'
                '\tpixel_accu {r[7]:.4f}'
                '\tIoU {r[8]:.4f}'.format(epoch, opt.train['num_epochs'], r=results.avg))

    return results.avg





# 全局变量建议放在脚本开头定义
nf_frozen = False
nf_loss_buffer = []

def maybe_freeze_nf_model_by_loss(epoch, nf_model, current_nf_loss, freeze_patience=15, freeze_threshold=35, std_threshold=5.0):
    """
    动态判断 nf_model 是否冻结：
    - 连续 freeze_patience 轮内 nf_loss 均 < freeze_threshold 且 std < std_threshold，则冻结。
    """
    global nf_frozen, nf_loss_buffer

    if nf_frozen:
        return  # 已冻结则直接跳过

    nf_loss_buffer.append(current_nf_loss)

    # 只在足够轮数后再判断是否冻结
    if len(nf_loss_buffer) >= freeze_patience:
        recent_losses = nf_loss_buffer[-freeze_patience:]
        avg_loss = sum(recent_losses) / freeze_patience
        std_loss = torch.std(torch.tensor(recent_losses)).item()

        if avg_loss < freeze_threshold and std_loss < std_threshold:
            print(f"[INFO] Freezing nf_model at epoch {epoch}: avg_nf_loss={avg_loss:.2f}, std={std_loss:.2f}")
            nf_model.eval()
            for p in nf_model.parameters():
                p.requires_grad = False
            nf_frozen = True



def compute_consistency_loss(out_unlabeled, out_all_unlabeled_pt, mask=None):
    if mask is not None:
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mse = F.mse_loss(out_unlabeled * mask, out_all_unlabeled_pt * mask, reduction='sum')
        norm = mask.sum() * out_unlabeled.shape[1]
        loss = mse / (norm + 1e-6)
    else:
        loss = F.mse_loss(out_unlabeled, out_all_unlabeled_pt)
    return loss




def visualize_tsne_preds(out_unlabeled, out_all_unlabeled_pt, save_path='tsne.png',
                         max_points=5000, pca_before_tsne=True):
    """
    保存无标签预测、对抗增强无标签预测的 t-SNE 分布图像

    参数：
        out_unlabeled: Tensor [B, C, H, W] （无标签预测）
        out_all_unlabeled_pt: Tensor [B, C, H, W] （对抗增强无标签预测）
        save_path: 图像保存路径（.png）
        max_points: 降维点数上限（避免过多点）
        pca_before_tsne: 是否先用 PCA 降到 50 维再做 t-SNE

    返回：
        None（只保存图像）
    """
    # 1. flatten 特征: [B, C, H, W] -> [B*H*W, C]
    def flatten_features(tensor):
        B, C, H, W = tensor.shape
        return tensor.permute(0, 2, 3, 1).reshape(-1, C).detach().cpu().numpy()

    features1 = flatten_features(out_unlabeled)
    features2 = flatten_features(out_all_unlabeled_pt)

    # 2. 降采样（避免点太多）
    if len(features1) > max_points:
        idx1 = np.random.choice(len(features1), max_points, replace=False)
        features1 = features1[idx1]
    if len(features2) > max_points:
        idx2 = np.random.choice(len(features2), max_points, replace=False)
        features2 = features2[idx2]

    # 3. 构造标签 & 拼接特征
    labels = np.array([0] * len(features1) + [1] * len(features2))
    features_all = np.concatenate([features1, features2], axis=0)

    # 4. PCA 降维（可选）
    if pca_before_tsne and features_all.shape[1] > 50:
        features_all = PCA(n_components=50).fit_transform(features_all)

    # 5. t-SNE
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
    features_2d = tsne.fit_transform(features_all)

    # 6. 绘制 t-SNE 可视化图并保存

    # 画无标签预测
    plt.figure(figsize=(8, 6))
    plt.scatter(features_2d[:len(features1), 0], features_2d[:len(features1), 1],
                alpha=0.6, s=5, color='blue')
    plt.title("t-SNE of Unlabeled Predictions")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{save_path}_unlabeled.png', dpi=300)
    plt.close()

    # 画对抗增强无标签预测
    plt.figure(figsize=(8, 6))
    plt.scatter(features_2d[len(features1):, 0], features_2d[len(features1):, 1],
                alpha=0.6, s=5, color='orange')
    plt.title("t-SNE of Adv-Aug Unlabeled Predictions")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{save_path}_adv_aug_unlabeled.png', dpi=300)
    plt.close()

    # 画合并结果
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'orange']
    labels_text = ['Unlabeled', 'Adv-Aug Unlabeled']

    for class_id in [0, 1]:
        idx = labels == class_id
        plt.scatter(features_2d[idx, 0], features_2d[idx, 1],
                    label=labels_text[class_id],
                    alpha=0.6, s=5, color=colors[class_id])

    plt.legend()
    plt.title("t-SNE of Unlabeled vs Adv-Augmented Unlabeled Predictions")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{save_path}_combined.png', dpi=300)
    plt.close()




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

def validate(val_loader, model, criterion,Flag=False):
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

    if Flag==True:
        if (len(val_loader)==20):
            logger.info('\t=> S_Model_ValB Avg:   Loss {r[0]:.4f}\tPixel_Acc {r[1]:.4f}'
                        '\tIoU {r[2]:.4f}'.format(r=results.avg))
        else:
            logger.info('\t=> S_Model_ValA Avg:   Loss {r[0]:.4f}\tPixel_Acc {r[1]:.4f}'
                        '\tIoU {r[2]:.4f}'.format(r=results.avg))
    else:
        if (len(val_loader) == 20):
            logger.info('\t=> T_Model_ValB Avg:   Loss {r[0]:.4f}\tPixel_Acc {r[1]:.4f}'
                        '\tIoU {r[2]:.4f}'.format(r=results.avg))
        else:
            logger.info('\t=> T_Model_ValA Avg:   Loss {r[0]:.4f}\tPixel_Acc {r[1]:.4f}'
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


def update_ema_variables(model, model_teacher, epoch, base_alpha=0.99):
    # 动态调整 alpha
    alpha = min(1.0 - 1.0 / float(epoch + 1), base_alpha)
    for param_t, param in zip(model_teacher.parameters(), model.parameters()):
        param_t.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

    return model, model_teacher




# def consistency_loss(logits_w1, logits_w2):
#        # logits_w2 = logits_w2.detach()
#        # assert logits_w1.size() == logits_w2.size()
#     return F.mse_loss(logits_w1, logits_w2,reduction='mean')

def compute_unsupervised_loss(predict, target, ignore_mask):

    target[ignore_mask==255] = 255
    loss = F.cross_entropy(predict, target, ignore_index=255)

    return loss

if __name__ == '__main__':
    main()