'''
化简的源代码加上新的块:Row+Col和CTB
4.9 加金字塔
'''
# Code for MedT
import os

import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import math
import argparse

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
from lib.models import ConvTransBlock
from utils_LOGO import JointTransform2D, ImageToImage2D, ImageToImage2D_, Image2D
from lib.models.RowCol import ColAttention, RowAttention
from metrics import jaccard_index, f1_score, dice_coeff, iouFun, Evaluation_Metrics, eval_metrics
import imgResize
import cv2
#导入SAM相关的包文件：
from segment_anything import SamAutomaticMaskGenerator,sam_model_registry,SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide

device = torch.device("cuda")
sam_model = sam_model_registry['vit_b'](checkpoint="sam_vit_b_01ec64.pth")
sam_model.to(device)
# predictor = SamPredictor(sam_model)

parser = argparse.ArgumentParser(description='MedT')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N', help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=260, type=int, metavar='N', help='number of total epochs to run(default: 400)')
parser.add_argument('-b', '--batch_size', default=2, type=int, metavar='N', help='batch size (default: 1)')
parser.add_argument('--learning_rate', default=1e-3, type=float, metavar='LR', help='initial learning rate (default: 0.001)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float, metavar='W', help='weight decay (default: 1e-5)')
# parser.add_argument('--train_dataset', default='D:/Code_Dataset/_Dataset/1/Task04_Hippocampus/train', type=str)
# parser.add_argument('--val_dataset', default='D:/Code_Dataset/_Dataset/1/Task04_Hippocampus/val', type=str)
parser.add_argument('--train_dataset', default='D:/Data/Code_Dataset/MoNuSeg/Train_Folder', type=str)
parser.add_argument('--val_dataset', default='D:/Data/Code_Dataset/MoNuSeg/Val_Folder', type=str)
parser.add_argument('--save_freq', type=int, default=20)
parser.add_argument('--cuda', default="on", type=str, help='switch on/off cuda option (default: off)')
parser.add_argument('--load', default='default', type=str, help='load a pretrained model')
parser.add_argument('--save', default='default', type=str, help='save the model')
parser.add_argument('--direc', default="D:/Data/MedT_SAM/original_LOGO_SAM/MSD/again", type=str,help='directory to save')
parser.add_argument('--crop', type=int, default=None)
parser.add_argument('--device', default='cuda', type=str)


args = parser.parse_args()
direc = args.direc
imgsize = 128

imgchant = 3

tf_train = JointTransform2D(crop=None, p_flip=0.5, color_jitter_params=None, long_mask=True)
tf_val = JointTransform2D(crop=None, p_flip=0, color_jitter_params=None, long_mask=True)
train_dataset = ImageToImage2D(args.train_dataset, tf_train)
val_dataset = ImageToImage2D_(args.val_dataset, tf_val)
predict_dataset = Image2D(args.val_dataset)
dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
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

class AxialBlock_dynamic(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(AxialBlock_dynamic, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_down = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.hight_block = ColAttention(planes, planes)
        self.width_block = RowAttention(planes, planes, stride=stride)
        self.conv_up = nn.Conv2d(planes, planes * 2, kernel_size=1, stride=1, bias=False)
        self.bn2 = norm_layer(planes * 2)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)     #[4, 16, 64, 64]

        out = self.hight_block(out)  #layer1取值[0, 5.8757]    layer2①取值[0, 10.0754]    ②取值[0, 9.2974]
        out = self.width_block(out)  #layer1取值[0, 5.8757]    layer2①取值[0, 4.1366]     ②取值[0, 9.2974]
        out = self.relu(out)

        out = self.conv_up(out)         # 16-->32
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)       # [4,32,64,64]

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
        self.conv_down = nn.Conv2d(inplanes, width, kernel_size=1, stride=1, bias=False)
        self.conv1 = nn.Conv2d(width, width, kernel_size = 1)
        self.bn1 = norm_layer(width)
        self.hight_block = AxialAttention_wopos(width, width, groups=groups, kernel_size=kernel_size)
        self.width_block = AxialAttention_wopos(width, width, groups=groups, kernel_size=kernel_size, stride=stride, width=True)
        self.conv_up = nn.Conv2d(width, planes * 2, kernel_size=1, stride=1, bias=False)
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

    def __init__(self):
        super(medt_net, self).__init__()

        self.inplanes = 8
        self.dilation = 1
        self.groups = 8
        self.base_width = 64
        self.conv1 = nn.Conv2d(3, 8, kernel_size=7, stride=2, padding=3,bias=False)
        self.conv2 = nn.Conv2d(8, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.downsample1 = nn.Sequential(
            conv1x1(8, 32, stride=1),
            nn.BatchNorm2d(32),
        )
        self.downsample2 = nn.Sequential(
            conv1x1(32, 64, stride=2),
            nn.BatchNorm2d(64)
        )
        self.layer1 = AxialBlock_dynamic(8, 16, stride=1, downsample=self.downsample1, norm_layer= nn.BatchNorm2d)
        self.layer2 = nn.Sequential(
            AxialBlock_dynamic(32, 32, stride=2, downsample=self.downsample2, norm_layer=nn.BatchNorm2d),
            AxialBlock_dynamic(64, 32, norm_layer=nn.BatchNorm2d)
        )

        self.decoder4 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.adjust = nn.Conv2d(16, 2, kernel_size=1, stride=1, padding=0)
        self.soft = nn.Softmax(dim=1)

        self.conv1_p = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv2_p = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3_p = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_p = nn.BatchNorm2d(64)
        self.bn2_p = nn.BatchNorm2d(128)
        self.bn3_p = nn.BatchNorm2d(64)
        self.relu_p = nn.ReLU(inplace=True)
        self.downsample2_p = nn.Sequential(
            conv1x1(32, 64, stride=2),
            nn.BatchNorm2d(64),
        )
        self.downsample3_p = nn.Sequential(
            conv1x1(64, 128, stride=2),
            nn.BatchNorm2d(128),
        )
        self.downsample4_p = nn.Sequential(
            conv1x1(128, 256, stride=2),
            nn.BatchNorm2d(256),
        )

        self.CTB1 = ConvTransBlock(16, 48, 16, 8, 0, 'W', 256)
        #self.CTB2 = ConvTransBlock(8, 24, 8, 8, 0, 'W', 256)
        self.conv_test1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        #self.conv_test2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer2_p = nn.Sequential(
            AxialBlock_wopos(32, 32, stride=2, downsample=self.downsample2_p, groups=self.groups, base_width=self.base_width, norm_layer=nn.BatchNorm2d, kernel_size=16),
            AxialBlock_wopos(64, 32, groups=self.groups, base_width=self.base_width, norm_layer=nn.BatchNorm2d, kernel_size=8),
        )
        self.layer3_p = nn.Sequential(
            AxialBlock_wopos(64, 64, stride=2, downsample=self.downsample3_p, groups=self.groups,base_width=self.base_width, norm_layer=nn.BatchNorm2d, kernel_size=8),
            AxialBlock_wopos(128, 64, groups=self.groups, base_width=self.base_width, norm_layer=nn.BatchNorm2d, kernel_size=4),
            AxialBlock_wopos(128, 64, groups=self.groups, base_width=self.base_width, norm_layer=nn.BatchNorm2d, kernel_size=4),
            AxialBlock_wopos(128, 64, groups=self.groups, base_width=self.base_width, norm_layer=nn.BatchNorm2d, kernel_size=4),
        )
        self.layer4_p = AxialBlock_wopos(128, 128, stride=2, downsample=self.downsample4_p, groups=self.groups, base_width=self.base_width, norm_layer=nn.BatchNorm2d, kernel_size=4)

        # Decoder
        self.decoder1_p = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.decoder2_p = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.decoder3_p = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.decoder4_p = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.decoder5_p = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.convLocal = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
        self.decoderf = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.adjust_p = nn.Conv2d(16, 2, kernel_size=1, stride=1, padding=0)
        # self.decoderf = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        # self.adjust_p = nn.Conv2d(3, 2, kernel_size=1, stride=1, padding=0)
        self.soft_p = nn.Softmax(dim=1)

        # 将SAM的编码变换到[b,8,64,64]的维度
        self.sam_tran = nn.Sequential(
            nn.Conv2d(256, 8, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.in_channels = 8
        self.inter_channels = 16
        self.w = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.miu = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

        # # 添加SAM_encoder模块：输入是xin[b,3,128,128],输出是[b,256,64,64]
        # self.sam_model = sam_model_registry['vit_b'](checkpoint="sam_vit_b_01ec64.pth")
        # # sam_model.to(device)
        # self.predictor = SamPredictor(self.sam_model)

    # 读取到的数据集进行预处理 处理成可以输入到SAM的encoder的格式
    def pre_batch(self,batch):
        processed_images = []
        for image in batch:
            image = (image * 255).clamp(0, 255).byte()  # 将格式从float改成unit8
            image = image.cpu().numpy().transpose(1, 2, 0)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换颜色空间
            image = ResizeLongestSide(1024).apply_image(image)  # 调整图像大小
            tensor = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1).contiguous()  # 转换为PyTorch张量
            processed_images.append(tensor)
        # 将数据堆叠为一个张量
        processed_images = torch.stack(processed_images).to(device=device)
        return processed_images

    # 获取SAM的encoder编码操作
    def sam_encoder(self,img):
        # 对batch进行预处理
        in_sam = self.pre_batch(img)
        input_image = sam_model.preprocess(in_sam)
        with torch.no_grad():
            image_embedding = sam_model.image_encoder(input_image)
        return image_embedding

    # 将卷积后的两个tensor进行拼接操作，输入x是表示SAM的encoder结果，a表示的是CRIS的neck结果
    def fuse(self,x, a, in_channels, inter_channels):
        # w = nn.Conv2d(in_channels=in_channels, out_channels=inter_channels, kernel_size=1).to(device)
        # theta = nn.Conv2d(in_channels=in_channels, out_channels=inter_channels, kernel_size=1).to(device)
        # phi = nn.Conv2d(in_channels=in_channels, out_channels=inter_channels, kernel_size=1).to(device)
        # miu = nn.Conv2d(in_channels=inter_channels, out_channels=in_channels, kernel_size=1).to(device)

        batch_size, C = x.size(0), x.size(1)  # batch_size:2 C：512
        w_x = self.w(x).view(batch_size, inter_channels, -1)  # [2,256,26*26]
        w_x = w_x.permute(0, 2, 1)  # [bs, HW, C]  [2,676,256]

        theta_x = self.theta(x).view(batch_size, inter_channels, -1)  # [bs, C', HW]
        phi_x = self.phi(a).view(batch_size, inter_channels, -1)  # [bs, C', HW] [2,256,676]
        theta_x = theta_x.permute(0, 2, 1)  # [bs, HW, C'] [2,676,256]
        f = torch.matmul(theta_x, phi_x)  # [bs, HW, HW] [2,676,676]
        N = f.size(-1)  # number of position in x   N: 676
        f_div_C = f / N  # [bs, HW, HW]

        y = torch.matmul(f_div_C, w_x)  # [bs, HW, C]  [2,676,256]
        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()  # [bs, C, HW]  [2,256,676]
        y = y.view(batch_size, inter_channels, *x.size()[2:])  # [bs, C', H, W] [2,256,26,26]
        W_y = self.miu(y)  # [bs, C, H, W]  [2,512,26,26]
        z = W_y + x
        return z

    def forward(self, x):
        xin = x.clone()    #[4, 3, 128, 128]
        x = x.to(torch.float32)
        x = self.conv1(x)       # 7x7conv   3->8
        x = self.bn1(x)
        x = self.relu(x)        # [4,8,64,64]
        x = self.conv2(x)       # 3x3conv   8->128
        x = self.bn2(x)
        x = self.relu(x)        # [4,128,64,64]
        x = self.conv3(x)       # 3x3conv   128->8
        x = self.bn3(x)

        x = self.relu(x)        # [4,8,64,64]

        # concate上SAM的编码
        sam_emb = self.sam_encoder(xin)
        # print(sam_emb.shape)

        sam_t = self.sam_tran(sam_emb)      # SAM编码结果卷积到8的维度 [b,8,64,64]

        # 拼接操作:
        x = self.fuse(sam_t,x,8,16)

        # 临时保存特征的可视化图像
        arr = torch.sum(x[0],0)
        up = cv2.resize(arr.detach().cpu().numpy(), (128, 128), interpolation=cv2.INTER_LINEAR)
        plt.imsave(r'D:\Data\MedT_SAM\show_img\feature_visual\covid_128.png',up)



        x1 = self.layer1(x)  #取值范围[0, 12.1742]      x1:[4,32,64,64]
        x2 = self.layer2(x1) #取值范围[0, 18.6337]      x2:[4,64,32,32]
        x = F.relu(F.interpolate(self.decoder4(x2), scale_factor=(2, 2), mode='bilinear'))      # decoder4(x2):[4,32,32,32]-----> 插值之后：x:[4,32,64,64]
        x = torch.add(x, x1)            # x:[4,32,64,64]
        x = F.relu(F.interpolate(self.decoder5(x), scale_factor=(2, 2), mode='bilinear'))       # decoder5(x):[4,16,64,64]------> 插值之后：x:[4,16,128,128]
        # x = self.adjust(F.relu(x))
        # end of full image training

        x_loc = x.clone()       # [4,16,128,128]
        # start
        for i in range(0, 4):
            for j in range(0, 4):
                B, C, H, W = x_loc.shape
                p = H // 4
                x_p = xin[:, :, p * i:p * (i + 1), p * j:p * (j + 1)]       # x_p:[4,3,32,32]
                #x_p = xin[:, :, 32 * i:32 * (i + 1), 32 * j:32 * (j + 1)]
                # begin patch wise
                x_p = x_p.to(torch.float32)
                x_p = self.conv1_p(x_p)     # [4,64,16,16]
                x_p = self.bn1_p(x_p)
                x_p = self.relu(x_p)

                x_p = self.conv2_p(x_p)     # [4,128,16,16]
                x_p = self.bn2_p(x_p)
                x_p = self.relu(x_p)

                x_p = self.conv3_p(x_p)     # [4,64,16,16]
                x_p = self.bn3_p(x_p)
                x_p = self.relu(x_p)  # [4,64,16,16]     取值范围[0, 4.6500]

                x_p = self.CTB1(x_p)  #取值范围[-1.2908, 5.2889]
                x1_p = self.conv_test1(x_p)  #取值范围[-2.0171, 1.9073]  加了BN和ReLU之后 [0, 5.7009] # [4,32,16,16]
                # x_p = self.CTB2(x1_p)
                # x2_p = self.conv_test2(x_p)
                x2_p = self.layer2_p(x1_p)   #取值范围[0, 7.5741]    #金字塔[4,64,8,8]
                x3_p = self.layer3_p(x2_p)   #取值范围[0, 7.2563]    #金字塔[4,128,4,4]
                x4_p = self.layer4_p(x3_p)   #取值范围[0, 4.2588]    #金字塔[4,256,2,2]

                # if x4_p.shape == torch.Size([4, 256, 1, 1]):
                #     x_p = F.relu(F.interpolate(self.decoder1_p(x4_p), scale_factor=(1, 1), mode='bilinear'))
                # else:
                #
                x_p = F.relu(F.interpolate(self.decoder1_p(x4_p), scale_factor=(2, 2), mode='bilinear'))        # self.decoder1_p(x4_p) [4,256,1,1]  relu之后：[4,256,2,2]
                x_p = torch.add(x_p, x4_p)          # x_p[4,256,2,2] x4_p[4,256,2,2]------> [4,128,4,4]
                x_p = F.relu(F.interpolate(self.decoder2_p(x_p), scale_factor=(2, 2), mode='bilinear'))         # self.decoder2_p(x_p) [4,256,1,1]   插值后 [4,128,4,4]
                x_p = torch.add(x_p, x3_p)          # [4,128,4,4]------> [4,64,8,8]
                x_p = F.relu(F.interpolate(self.decoder3_p(x_p), scale_factor=(2, 2), mode='bilinear'))         # [4,64,8,8]
                x_p = torch.add(x_p, x2_p)          # [4,64,8,8]-------> [4,32,16,16]
                x_p = F.relu(F.interpolate(self.decoder4_p(x_p), scale_factor=(2, 2), mode='bilinear'))         # [4,32,16,16]
                x_p = torch.add(x_p, x1_p)          # [4,16,32,32]------> []
                x_p = F.relu(F.interpolate(self.decoder5_p(x_p), scale_factor=(2, 2), mode='bilinear'))         #[4,16,32,32]
                #x_p = self.convLocal(x_p)
                x_loc[:, :, p * i:p * (i + 1), p * j:p * (j + 1)] = x_p

        x = torch.add(x, x_loc)         # x[4,16,128,128] x_loc[4,16,128,128]-------->x[4,16,128,128]
        x = F.relu(self.decoderf(x))    # [4,16,128,128]
        x = self.adjust_p(F.relu(x))    # [4,2,128,128]

        return x


if __name__ == '__main__':
    device = torch.device("cuda")
    model = medt_net().to(device=device)
    # model.load_state_dict(torch.load("D:/Data/MedT_SAM/original_LOGO_SAM/MSD/_test_output/_weight/best_epoch119_best_iou0.9332_best_local.pth"))
    optimizer = torch.optim.Adam(list(model.parameters()), lr=args.learning_rate,weight_decay=1e-5)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))
    best_iou = 0
    best_val_iou = 0
    patience = 12
    counter = 0
    for epoch in range(args.epochs):

        epoch_running_loss = 0
        # F1 = 0
        # IoU = 0
        total_inter, total_union = 0, 0
        total_correct, total_label = 0, 0
        model.train()
        for batch_idx, (X_batch, y_batch, *rest) in enumerate(dataloader):
            # X_batch = Variable(X_batch.to(device='cuda'))   #取值[0,1]
            # y_batch = Variable(y_batch.to(device='cuda'))   #取值0和1
            #print("========batch:{}".format(batch_idx))
            #print(X_batch[i].shape)
            X_batch = Variable(X_batch.to(device=device))       # [4,3,128,128]
            y_batch = Variable(y_batch.to(device=device))       # [4,128,128]

            output = model(X_batch)                  # [4,2,128,128]

            loss = F.cross_entropy(output, y_batch, weight=None, ignore_index=-100)
            #y_batch_i = y_batch_i.float()
            F1 = f1_score(output, y_batch)
            #metrics1 = Evaluation_Metrics(output, y_batch_i)
            seg_metrics = eval_metrics(output, y_batch, 2)
            total_correct += seg_metrics[0]
            total_label += seg_metrics[1]
            total_inter += seg_metrics[2]
            total_union += seg_metrics[3]
            pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
            IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
            mIoU = IoU.mean()
            #dice = dice_coeff(output, y_batch_i)
            #iou = iouFun(output, y_batch_i)
            #IOU = jaccard_index(output, y_batch_i)

            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_running_loss += loss.item()
            # F1 += f1
            # IoU += iou
        # ===================log========================
        # print('epoch [{}/{}], loss:{:.4f}, F1:{}, IoU:{}'
        #       .format(epoch, args.epochs, epoch_running_loss / (batch_idx + 1), F1 / (batch_idx + 1), IoU / (batch_idx + 1)))
        print('epoch [{}/{}], loss:{:.4f}, F1:{}, Acc {:.4f} mIoU {:.4f}'
              .format(epoch, args.epochs, epoch_running_loss / (batch_idx + 1), F1, pixAcc, mIoU))

        # # 保存训练时候最优的模型参数
        # if (mIoU > best_iou):
        #     best_iou = mIoU
        #     fulldir = direc + "/_train_output"
        #     if not os.path.isdir(fulldir):
        #         os.makedirs(fulldir)
        #     torch.save(model.state_dict(),
        #                fulldir + '/best_epoch{}_best_iou{:.4f}'.format(epoch, best_iou) + "_best_local.pth")
        #     torch.save(model, direc + "AATC.pth")

        if epoch == 10:
            for param in model.parameters():
                param.requires_grad = True  # 打印参数个数

        # if (epoch % args.save_freq) == 0:  # 10的倍数 进行测试

        model.eval()
        with torch.no_grad():
            # val集进行验证
            for batch_idx, (X_batch_, y_batch_, *rest) in enumerate(valloader):
                if isinstance(rest[0][0], str):
                    image_filename = rest[0][0]
                else:
                    image_filename = '%s.png' % str(batch_idx + 1).zfill(3)

                X_batch_ = Variable(X_batch_.to(device=device))
                y_batch_ = Variable(y_batch_.to(device=device))
                y_out_ = model(X_batch_)

                # 计算val上的指标iou

                seg_metrics = eval_metrics(y_out_, y_batch_, 2)
                total_correct += seg_metrics[0]
                total_label += seg_metrics[1]
                total_inter += seg_metrics[2]
                total_union += seg_metrics[3]
                pixAcc_ = 1.0 * total_correct / (np.spacing(1) + total_label)
                val_IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
                val_mIoU = val_IoU.mean()

                tmp_ = y_out_.detach().cpu().numpy()
                tmp_[tmp_ >= 0.5] = 1
                tmp_[tmp_ < 0.5] = 0
                tmp_ = tmp_.astype(int)

                yHaT_ = tmp_

                del X_batch_, y_batch_, tmp_, y_out_

                yHaT_[yHaT_ == 1] = 255
                fulldir = direc + "/_test_output" + "/{}/".format(epoch)
                if not os.path.isdir(fulldir):
                    os.makedirs(fulldir)

                cv2.imwrite(fulldir + image_filename, yHaT_[0, 1, :, :])
        # 保存训练时候最优的模型参数
        if (val_mIoU > best_val_iou)and(epoch>10):
            best_val_iou = val_mIoU
            fulldir = direc + "/_test_output" + "/_weight"
            if not os.path.isdir(fulldir):
                os.makedirs(fulldir)
            torch.save(model.state_dict(),
                        fulldir + '/best_epoch{}_best_iou{:.4f}'.format(epoch, best_val_iou) + "_best_local.pth")
            torch.save(model, direc + "AATC.pth")
            counter = 0
        else:
            counter+=1
        # # 判断是否满足 Early Stopping 条件
        # if counter > patience:
        #     print("Validation metric did not improve for {} consecutive epochs. Training stopped.".format(patience))
        #     break