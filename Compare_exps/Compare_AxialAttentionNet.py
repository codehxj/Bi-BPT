import math
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils import model_zoo
from torchvision import models
import argparse

from metrics import f1_score, eval_metrics, Metric, accuracy
from utils_com import JointTransform2D, ImageToImage2D, Image2D
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='MedT')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N', help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run(default: 400)')
parser.add_argument('-b', '--batch_size', default=4, type=int, metavar='N', help='batch size (default: 1)')
parser.add_argument('--learning_rate', default=1e-3, type=float, metavar='LR', help='initial learning rate (default: 0.001)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float, metavar='W', help='weight decay (default: 1e-5)')
parser.add_argument('--train_dataset', default='D:/Data/Code_Dataset/MoNuSeg/Train_Folder', type=str)
parser.add_argument('--val_dataset', default='D:/Data/Code_Dataset/MoNuSeg/Val_Folder', type=str)
parser.add_argument('--save_freq', type=int, default=10)
parser.add_argument('--cuda', default="on", type=str, help='switch on/off cuda option (default: off)')
parser.add_argument('--load', default='default', type=str, help='load a pretrained model')
parser.add_argument('--save', default='default', type=str, help='save the model')
parser.add_argument('--direc', default="D:/Data/MedT_SAM/Compare/AxialAtten/MoNuSeg", type=str,help='directory to save')
parser.add_argument('--crop', type=int, default=None)
parser.add_argument('--device', default='cuda', type=str)

args = parser.parse_args()
direc = args.direc
imgsize = 128
imgchant = 3

tf_train = JointTransform2D(crop=None, p_flip=0.5, color_jitter_params=None, long_mask=True)
tf_val = JointTransform2D(crop=None, p_flip=0, color_jitter_params=None, long_mask=True)
train_dataset = ImageToImage2D(args.train_dataset, tf_train)
val_dataset = ImageToImage2D(args.val_dataset, tf_val)
predict_dataset = Image2D(args.val_dataset)
dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
valloader = DataLoader(val_dataset, 1, shuffle=True)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class qkv_transform(nn.Conv1d):
    """Conv1d for qkv_transform"""

class AxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)
        #self.bn_qk = nn.BatchNorm2d(groups)
        #self.bn_qr = nn.BatchNorm2d(groups)
        #self.bn_kr = nn.BatchNorm2d(groups)
        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        # Position embedding
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # N, W, C, H
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H), [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        # Calculate position embedding
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2, self.kernel_size, self.kernel_size)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings, [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=0)
        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
        qk = torch.einsum('bgci, bgcj->bgij', q, k)
        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 3, self.groups, H, H).sum(dim=1)
        #stacked_similarity = self.bn_qr(qr) + self.bn_kr(kr) + self.bn_qk(qk)
        # (N, groups, H, H, W)
        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)
        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W, self.out_planes * 2, H)
        output = self.bn_output(stacked_output).view(N, W, self.out_planes, 2, H).sum(dim=-2)

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        #nn.init.uniform_(self.relative, -0.1, 0.1)
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))


class AxialBlock(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=56):
        super(AxialBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_down = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.hight_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size)
        self.width_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size, stride=stride, width=True)
        self.conv_up = conv1x1(width, planes * self.expansion)
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.hight_block(out)
        out = self.width_block(out)
        out = self.relu(out)

        out = self.conv_up(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class AxialAttentionNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=True,
                 groups=8, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, s=0.5):
        super(AxialAttentionNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = int(64 * s)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, int(128 * s), layers[0], kernel_size=32)
        self.layer2 = self._make_layer(block, int(256 * s), layers[1], stride=2, kernel_size=32,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, int(512 * s), layers[2], stride=2, kernel_size=16,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, int(1024 * s), layers[3], stride=2, kernel_size=8,
                                       dilate=replace_stride_with_dilation[2])

        self.decoder1 = nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1)
        self.decoder2 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1)
        self.decoder3 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.decoder4 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(128, 2, kernel_size=3, stride=1, padding=1)
        self.bn = norm_layer(2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(1024 * block.expansion * s), num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                if isinstance(m, qkv_transform):
                    pass
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, AxialBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, kernel_size=56, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation,
                            norm_layer=norm_layer, kernel_size=kernel_size))
        self.inplanes = planes * block.expansion
        if stride != 1:
            kernel_size = kernel_size // 2

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, kernel_size=kernel_size))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)   #[4, 128, 32, 32]
        x2 = self.layer2(x1)  #[4, 256, 16, 16]
        x3 = self.layer3(x2)  #[4, 512, 8, 8]
        x4 = self.layer4(x3)  #[4, 1024, 4, 4]  #[4, 1024, 4, 4]

        x = F.interpolate(self.decoder1(x4), scale_factor=2, mode='bilinear')
        x = torch.add(x, x4)
        x = F.interpolate(self.decoder2(x), scale_factor=2, mode='bilinear')
        x = torch.add(x, x3)
        x = F.interpolate(self.decoder3(x), scale_factor=2, mode='bilinear')
        x = torch.add(x, x2)
        x = F.interpolate(self.decoder4(x), scale_factor=2, mode='bilinear')
        x = torch.add(x, x1)
        x = F.interpolate(self.decoder5(x), scale_factor=4, mode='bilinear')
        x = self.bn(x)
        x = self.relu(x)

        # x = self.avgpool(x)    #[4, 1024, 1, 1]
        # x = torch.flatten(x, 1)  #[4, 1024]
        # x = self.fc(x)  #[4,  1000]

        return x

    def forward(self, x):
        return self._forward_impl(x)


if __name__ == '__main__':
    device = torch.device("cuda")
    model = AxialAttentionNet(AxialBlock, [3, 4, 6, 3], s=0.5).to(device)
    #model.load_state_dict(torch.load("./'result2_original_pyramid'final_model.pth"))

    optimizer = torch.optim.Adam(list(model.parameters()), lr=args.learning_rate,weight_decay=1e-5)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))

    for epoch in range(100):

        train_loss = Metric('train_loss')
        train_accuracy = Metric('train_accuracy')
        epoch_running_loss = 0
        # f1 = 0
        # mIoU1 = 0
        total_inter, total_union = 0, 0
        total_correct, total_label = 0, 0
        for batch_idx, (X_batch, y_batch, *rest) in enumerate(dataloader):
            X_batch = Variable(X_batch.to(device='cuda'))  # shape=[4, 3, 128, 128]
            y_batch = Variable(y_batch.to(device='cuda'))  # shape=[4, 128, 128]
            #y_batch1 = torch.flatten(y_batch, 1)
            # ===================forward=====================

            output = model(X_batch)  # MedT

            loss = F.cross_entropy(output, y_batch)
            F1 = f1_score(output, y_batch).mean()
            seg_metrics = eval_metrics(output, y_batch, 2)
            total_correct += seg_metrics[0]
            total_label += seg_metrics[1]
            total_inter += seg_metrics[2]
            total_union += seg_metrics[3]
            # pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
            IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
            mIoU = IoU.mean()

            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #epoch_running_loss += loss.item()
            train_loss.update(loss)
            train_accuracy.update(accuracy(output, y_batch))
            # f1 += F1
            # mIoU1 += mIoU
            # ===================log========================
            #if (batch_idx + 1) % 20 == 0:
        print('epoch [{}/{}], loss:{:.4f}, F1:{}, Acc {:.4f}, mIoU {:.4f}'
            .format(epoch, args.epochs, train_loss.avg.item(), F1,train_accuracy.avg.item(), mIoU))

        if epoch == 10:
            for param in model.parameters():
                param.requires_grad = True
        if (epoch % args.save_freq) == 0:

            for batch_idx, (X_batch, y_batch, *rest) in enumerate(valloader):
                # print(batch_idx)
                if isinstance(rest[0][0], str):
                    image_filename = rest[0][0]
                else:
                    image_filename = '%s.png' % str(batch_idx + 1).zfill(3)

                X_batch = Variable(X_batch.to(device='cuda'))
                y_batch = Variable(y_batch.to(device='cuda'))
                # start = timeit.default_timer()
                y_out = model(X_batch)
                # stop = timeit.default_timer()
                # print('Time: ', stop - start)
                tmp2 = y_batch.detach().cpu().numpy()
                tmp = y_out.detach().cpu().numpy()
                tmp[tmp >= 0.5] = 1554
                tmp[tmp < 0.5] = 0
                tmp2[tmp2 > 0] = 1
                tmp2[tmp2 <= 0] = 0
                tmp2 = tmp2.astype(int)
                tmp = tmp.astype(int)

                # print(np.unique(tmp2))
                yHaT = tmp
                yval = tmp2

                epsilon = 1e-20

                del X_batch, y_batch, tmp, tmp2, y_out

                yHaT[yHaT == 1] = 255
                yval[yval == 1] = 255
                fulldir = direc + "/{}/".format(epoch)
                # print(fulldir+image_filename)
                if not os.path.isdir(fulldir):
                    os.makedirs(fulldir)

                cv2.imwrite(fulldir + image_filename, yHaT[0, 1, :, :])
                # cv2.imwrite(fulldir+'/gt_{}.png'.format(count), yval[0,:,:])
            fulldir = direc + "/{}/".format(epoch)
            torch.save(model.state_dict(), fulldir +  "AxialNet.pth")
            torch.save(model.state_dict(), "AxialNet.pth")


