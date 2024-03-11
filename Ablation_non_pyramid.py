# D:/Data/Code_Dataset/MoNuSeg/Train_Folder
'''
未化简的源代码加上新的块:Row+Col和CTB
'''
# Code for MedT
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import math
import argparse
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
from lib.models import ConvTransBlock
from utils_comp_DAEF import JointTransform2D, ImageToImage2D, Image2D
from lib.models.RowCol import ColAttention, RowAttention
from metrics import jaccard_index, f1_score
import cv2

parser = argparse.ArgumentParser(description='MedT')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N', help='number of data loading workers (default: 8)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
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
parser.add_argument('--direc', default='D:\Data\MedT_SAM\Anlation\no_pyramid', type=str,help='directory to save')
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
dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valloader = DataLoader(val_dataset, 1, shuffle=True)

class qkv_transform(nn.Conv1d):
    """Conv1d for qkv_transform"""

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class AxialAttention_wopos(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention_wopos, self).__init__()
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
        self.bn_similarity = nn.BatchNorm2d(groups )

        self.bn_output = nn.BatchNorm1d(out_planes * 1)

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

        qk = torch.einsum('bgci, bgcj->bgij', q, k)

        stacked_similarity = self.bn_similarity(qk).reshape(N * W, 1, self.groups, H, H).sum(dim=1).contiguous()

        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)

        sv = sv.reshape(N*W,self.out_planes * 1, H).contiguous()
        output = self.bn_output(sv).reshape(N, W, self.out_planes, 1, H).sum(dim=-2).contiguous()


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
        # nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))

class AxialBlock_dynamic(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(AxialBlock_dynamic, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_down = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = norm_layer(planes)
        self.hight_block = ColAttention(planes, planes)
        self.width_block = RowAttention(planes, planes, stride=stride)
        self.conv_up = nn.Conv2d(planes, planes * 2, kernel_size=1, stride=stride, bias=False)
        self.bn2 = norm_layer(planes * 2)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)     #[4, 16, 64, 64]

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

class AxialBlock_wopos(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None, kernel_size=56):
        super(AxialBlock_wopos, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # print(kernel_size)
        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_down = nn.Conv2d(inplanes, width, kernel_size=1, stride=stride, bias=False)
        self.conv1 = nn.Conv2d(width, width, kernel_size = 1)
        self.bn1 = norm_layer(width)
        self.hight_block = AxialAttention_wopos(width, width, groups=groups, kernel_size=kernel_size)
        self.width_block = AxialAttention_wopos(width, width, groups=groups, kernel_size=kernel_size, stride=stride, width=True)
        self.conv_up = nn.Conv2d(width, planes * 2, kernel_size=1, stride=stride, bias=False)
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        # pdb.set_trace()

        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)
        # print(out.shape)
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


class medt_net(nn.Module):

    def __init__(self, block, block_2, layers, num_classes=2, groups=8, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(medt_net, self).__init__()

        self.inplanes = 8
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, 8, kernel_size=7, stride=2, padding=3,bias=False)
        self.conv2 = nn.Conv2d(8, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0], kernel_size=64)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2, kernel_size=64,dilate=False)

        self.decoder4 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.adjust = nn.Conv2d(16, num_classes, kernel_size=1, stride=1, padding=0)
        self.soft = nn.Softmax(dim=1)

        # revised at 2022-4-1-22:11 for testing SCUNet
        self.conv_test1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_test2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv1_p = nn.Conv2d(3, 8, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv2_p = nn.Conv2d(8, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3_p = nn.Conv2d(128, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_p = nn.BatchNorm2d(8)
        self.bn2_p = nn.BatchNorm2d(128)
        self.bn3_p = nn.BatchNorm2d(8)
        self.relu_p = nn.ReLU(inplace=True)

        self.layer1_p = self._make_layer(block_2, 16, layers[0], kernel_size=16)
        self.layer2_p = self._make_layer(block_2, 32, layers[1], stride=2, kernel_size=16,dilate=False)
        self.layer3_p = self._make_layer(block_2, 64, layers[2], stride=2, kernel_size=8,dilate=False)
        self.layer4_p = self._make_layer(block_2, 128, layers[3], stride=2, kernel_size=4,dilate=False)

        # Decoder
        self.decoder1_p = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.decoder2_p = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.decoder3_p = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.decoder4_p = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.decoder5_p = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        # self.SC = SCUNet()
        self.CTB1 = ConvTransBlock(16, 48, 16, 8, 0, 'W', 256)
        self.CTB2 = ConvTransBlock(8, 24, 8, 8, 0, 'W', 256)

        self.decoderf = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.adjust_p = nn.Conv2d(16, num_classes, kernel_size=1, stride=1, padding=0)
        self.soft_p = nn.Softmax(dim=1)

    def _make_layer(self, block, planes, blocks, kernel_size=56, stride=1, dilate=False):
        norm_layer = nn.BatchNorm2d
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
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        if stride != 1:
            kernel_size = kernel_size // 2

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=nn.BatchNorm2d))

        return nn.Sequential(*layers)

    def forward(self, x):

        xin = x.clone()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)  # [4,8,64,64]

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x = F.relu(F.interpolate(self.decoder4(x2), scale_factor=(2, 2), mode='bilinear'))
        x = torch.add(x, x1)
        x = F.relu(F.interpolate(self.decoder5(x), scale_factor=(2, 2), mode='bilinear'))
        # end of full image training

        x_loc = x.clone()

        # start
        for i in range(0, 4):
            for j in range(0, 4):
                x_p = xin[:, :, 32 * i:32 * (i + 1), 32 * j:32 * (j + 1)]
                # begin patch wise
                x_p = self.conv1_p(x_p)
                x_p = self.bn1_p(x_p)
                x_p = self.relu(x_p)

                x_p = self.conv2_p(x_p)
                x_p = self.bn2_p(x_p)
                x_p = self.relu(x_p)
                x_p = self.conv3_p(x_p)
                x_p = self.bn3_p(x_p)
                x_p = self.relu(x_p)  # [4,64,16,16]

                x_p = self.CTB1(x_p)
                x1_p = self.conv_test1(x_p)  # [4,32,16,16]
                # x_p = self.CTB2(x1_p)
                # x2_p = self.conv_test2(x_p)
                x2_p = self.layer2_p(x1_p)
                x3_p = self.layer3_p(x2_p)
                x4_p = self.layer4_p(x3_p)

                x_p = F.relu(F.interpolate(self.decoder1_p(x4_p), scale_factor=(2, 2), mode='bilinear'))
                x_p = torch.add(x_p, x4_p)
                x_p = F.relu(F.interpolate(self.decoder2_p(x_p), scale_factor=(2, 2), mode='bilinear'))
                x_p = torch.add(x_p, x3_p)
                x_p = F.relu(F.interpolate(self.decoder3_p(x_p), scale_factor=(2, 2), mode='bilinear'))
                x_p = torch.add(x_p, x2_p)
                x_p = F.relu(F.interpolate(self.decoder4_p(x_p), scale_factor=(2, 2), mode='bilinear'))
                x_p = torch.add(x_p, x1_p)
                x_p = F.relu(F.interpolate(self.decoder5_p(x_p), scale_factor=(2, 2), mode='bilinear'))

                x_loc[:, :, 32 * i:32 * (i + 1), 32 * j:32 * (j + 1)] = x_p

        x = torch.add(x, x_loc)
        x = F.relu(self.decoderf(x))

        x = self.adjust(F.relu(x))

        return x

device = torch.device("cuda")
model = medt_net(AxialBlock_dynamic,AxialBlock_wopos, [1, 2, 4, 1])
model.to(device)

optimizer = torch.optim.Adam(list(model.parameters()), lr=args.learning_rate,weight_decay=1e-5)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total_params: {}".format(pytorch_total_params))


for epoch in range(200):

    epoch_running_loss = 0
    F1 = 0
    IoU = 0

    for batch_idx, (X_batch, y_batch, *rest) in enumerate(dataloader):
        X_batch = Variable(X_batch.to(device='cuda'))
        y_batch = Variable(y_batch.to(device='cuda'))

        # ===================forward=====================

        output = model(X_batch)

        loss = F.cross_entropy(output, y_batch, weight=None, ignore_index=-100)
        f1 = f1_score(output, y_batch)
        iou = jaccard_index(output, y_batch)

        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_running_loss += loss.item()
        F1 += f1
        IoU += iou

    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}, F1:{}, IoU:{}'
          .format(epoch, args.epochs, epoch_running_loss / (batch_idx + 1), F1 / (batch_idx + 1),
                  IoU / (batch_idx + 1)))

    if epoch == 10:
        for param in model.parameters():
            param.requires_grad = True   #打印参数个数

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
            tmp[tmp >= 0.5] = 1
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
            fulldir = "/{}/".format(epoch)
            # print(fulldir+image_filename)
            if not os.path.isdir(fulldir):
                os.makedirs(fulldir)

            cv2.imwrite(fulldir + image_filename, yHaT[0, 1, :, :])
            # cv2.imwrite(fulldir+'/gt_{}.png'.format(count), yval[0,:,:])
        fulldir = direc + "/{}/".format(epoch)
        torch.save(model.state_dict(), fulldir  + "medt.pth")
        torch.save(model.state_dict(), direc + "final_model.pth")



