
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torchvision.models.segmentation.deeplabv3 import ASPP
from torchvision.transforms import Resize, Compose, RandomCrop, ToTensor, ToPILImage, Normalize
import cv2
from DefEDNetmain.DefEDNet import SeparableConv2d
from FullNet import DoubleConv
from MyUnet import DoubleConvn
from RFBmodel import BasicRFB_a
from backbones.resnet.resnet_factory import get_resnet_backbone

from functools import partial

from backbones.scale_attention_layer import scale_atten_convblock, conv3x3, conv1x1
from models import DeepLab
from network import deeplabv3plus_resnet101
from smatunetmodels.layers import DepthwiseSeparableConv, CBAM, ChannelAttention, SpatialAttention
from smatunetmodels.unet_parts_depthwise_separable import DoubleConvDS
from einops import rearrange
nonlinearity = partial(F.elu, inplace=True)

from attack import attack








class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out



class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.upsample(self.conv(self.pool1(x)), size=(h, w), mode='bilinear')
        self.layer2 = F.upsample(self.conv(self.pool2(x)), size=(h, w), mode='bilinear')
        self.layer3 = F.upsample(self.conv(self.pool3(x)), size=(h, w), mode='bilinear')
        self.layer4 = F.upsample(self.conv(self.pool4(x)), size=(h, w), mode='bilinear')

        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        # self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.conv1 = DepthwiseSeparableConv(in_channels,in_channels//4,1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        # self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.conv3 = DepthwiseSeparableConv(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x
class my_up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True, kernels_per_layer=2):
        super(my_up,self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConvDS(in_channels, out_channels, in_channels // 2, kernels_per_layer=kernels_per_layer)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvDS(in_channels, out_channels, kernels_per_layer=kernels_per_layer)
        self.conv1 = nn.Conv2d(in_channels,in_channels//2,1,padding=0)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x1_ = self.conv1(x1)
        x3 = x2*x1_
        x = torch.cat([x3, x1_], dim=1)
        return self.conv(x)+self.conv(x1)
class my_up1(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True, kernels_per_layer=2):
        super(my_up1,self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConvDS(in_channels, out_channels, in_channels // 2, kernels_per_layer=kernels_per_layer)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvDS(in_channels, out_channels, kernels_per_layer=kernels_per_layer)
        self.conv1 = nn.Conv2d(in_channels,out_channels,1,padding=0)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x3 = x2*x1
        x = torch.cat([x3, x1], dim=1)
        return self.conv(x)+x1


class UnetDsv3(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(UnetDsv3, self).__init__()
        self.dsv = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0),
                                 nn.Upsample(size=scale_factor, mode='bilinear'), )

    def forward(self, input):
        return self.dsv(input)


class New_Semic_Seg(nn.Module):
    def __init__(self, num_classes=3, num_channels=3):
        super(New_Semic_Seg, self).__init__()
        resnet = get_resnet_backbone('resnet34')(pretrain=True)
        print("__init__, New_Semic_Seg")
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            DACblock(512),
            SPPblock(512)
        )
        # mix_norm = MixSyncBatchNorm

        # self.representation = nn.Sequential(
        #     nn.Conv2d(516, 256, kernel_size=3, stride=1, padding=1, bias=True),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout2d(0.1),
        #     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout2d(0.1)
        # )

        self.representation = nn.Sequential(DecoderBlock(516, 256))
   
        self.decoder0 = nn.Sequential(
            DecoderBlock(256, 128),
            DecoderBlock(128, 64),
            DecoderBlock(64, 64),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU()

        )
        self.decoder0_1 = nn.Sequential(
            nn.Conv2d(32, 3, 3, padding=1)
        )
            
    def forward(self, x_l, x_u=None,label=None, nf_model=None, loss_flow=None, cfg=None, eps=0, adv=False):
        if x_u==None:
            x_l1 = self.encoder(x_l)
            res_head_l1 = self.representation(x_l1)
            out_fin_labeled = self.decoder0(res_head_l1)
            out_labeled = self.decoder0_1(out_fin_labeled)

            return out_labeled, res_head_l1
        else:
            if adv:
                # 有标签预测
                x_l1 = self.encoder(x_l)
                res_head_l1 = self.representation(x_l1)
                out_fin_labeled = self.decoder0(res_head_l1)
                out_labeled = self.decoder0_1(out_fin_labeled)

                # 无标签预测
                x_u1 = self.encoder(x_u)
                res_head_u1 = self.representation(x_u1)
                out_fin_unlabeled = self.decoder0(res_head_u1)
                out_unlabeled = self.decoder0_1(out_fin_unlabeled)

                # 无标签扰动增强预测

                x_u1_pt = self.encoder(x_u.clone()).float()
                pt = attack(x_u1_pt, label, self.representation, self.decoder0_1, nf_model, loss_flow, cfg, eps)
                mean = x_u1_pt.mean(dim=[2, 3], keepdim=True)
                std = x_u1_pt.std(dim=[2, 3], keepdim=True) + 1e-6
                x_u1_pt_norm = (x_u1_pt - mean) / std

                # print("pt.max:",pt.max())
                # print("pt.min:", pt.min())
                # print("x_u1_pt_norm.max:", x_u1_pt_norm.max())
                # print("x_u1_pt_norm.min:", x_u1_pt_norm.min())
                fts_half_pt = x_u1_pt_norm + pt*3
                out_fts_half_pt = self.representation(fts_half_pt)
                out_fin_unlabeled_pt = self.decoder0(out_fts_half_pt)
                out_all_unlabeled_pt = self.decoder0_1(out_fin_unlabeled_pt)

                return out_labeled,out_unlabeled, out_all_unlabeled_pt

            else:
                # 有标签预测
                x_l1 = self.encoder(x_l)
                res_head_l1 = self.representation(x_l1)
                out_fin_labeled = self.decoder0(res_head_l1)
                out_labeled = self.decoder0_1(out_fin_labeled)

                # 无标签预测
                x_u1 = self.encoder(x_u)
                res_head_u1 = self.representation(x_u1)
                out_fin_unlabeled = self.decoder0(res_head_u1)
                out_unlabeled = self.decoder0_1(out_fin_unlabeled)
                res_head = torch.cat([res_head_l1,res_head_u1],dim=0)
                
                return out_labeled, out_unlabeled,res_head



