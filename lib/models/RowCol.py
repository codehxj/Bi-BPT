import torch
from torch import nn

device = torch.device("cpu")
class RowAttention(nn.Module):

    def __init__(self, in_dim, q_k_dim, stride):
        '''
        Parameters
        ----------
        in_dim : int
            channel of input img tensor
        q_k_dim: int
            channel of Q, K vector
        '''
        super(RowAttention, self).__init__()
        self.in_dim = in_dim
        self.q_k_dim = q_k_dim
        self.stride = stride

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        self.gamma = nn.Parameter(torch.zeros(1)).to(device=device)

        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

    def forward(self, x):
        '''
        Parameters
        ----------
        x : Tensor
            4-D , (batch, in_dims, height, width) -- (b,c1,h,w)
        '''

        ## c1 = in_dims; c2 = q_k_dim
        b, _, h, w = x.size()

        Q = self.query_conv(x)  # size = (b,c2, h,w)
        K = self.key_conv(x)  # size = (b, c2, h, w)
        V = self.value_conv(x)  # size = (b, c1,h,w)

        Q = Q.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w).permute(0, 2, 1)  # size = (b*h,w,c2) = [256, 256, 32]
        K = K.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)  # size = (b*h,c2,w) = [256, 32, 256]
        V = V.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)  # size = (b*h, c1,w) = [256, 16, 256]

        # size = (b*h,w,w) [:,i,j] 表示Q的所有h的第 Wi行位置上所有通道值与 K的所有h的第 Wj列位置上的所有通道值的乘积，
        # 即(1,c2) * (c2,1) = (1,1)
        row_attn = torch.bmm(Q, K)  #size = [256,256,256]
        ########
        # 此时的 row_atten的[:,i,0:w] 表示Q的所有h的第 Wi行位置上所有通道值与 K的所有行的 所有列(0:w)的逐个位置上的所有通道值的乘积
        # 此操作即为 Q的某个（i,j）与 K的（i,0:w）逐个位置的值的乘积，得到行attn
        ########

        # 对row_attn进行softmax
        row_attn = self.softmax(row_attn)  # 对列进行softmax，即[k,i,0:w] ，某一行的所有列加起来等于1，

        # size = (b*h,c1,w) 这里先需要对row_atten进行 行列置换，使得某一列的所有行加起来等于1
        # [:,i,j]即为V的所有行的某个通道上，所有列的值 与 row_attn的行的乘积，即求权重和
        out = torch.bmm(V, row_attn.permute(0, 2, 1))  #size = [256, 16, 256]

        # size = (b,c1,h,2)
        out = out.view(b, h, -1, w).permute(0, 2, 1, 3)  #size = [1, 16, 256, 256]

        out = self.gamma * out + x

        if self.stride > 1:
            out = self.pooling(out)

        return out


class ColAttention(nn.Module):

    def __init__(self, in_dim, q_k_dim):
        '''
        Parameters
        ----------
        in_dim : int
            channel of input img tensor
        q_k_dim: int
            channel of Q, K vector
        device : torch.device
        '''
        super(ColAttention, self).__init__()
        self.in_dim = in_dim
        self.q_k_dim = q_k_dim
        # self.device = torch.device("cuda")

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        self.gamma = nn.Parameter(torch.zeros(1)).to(device=device)

    def forward(self, x):
        '''
        Parameters
        ----------
        x : Tensor
            4-D , (batch, in_dims, height, width) -- (b,c1,h,w)
        '''

        ## c1 = in_dims; c2 = q_k_dim
        b, _, h, w = x.size()

        Q = self.query_conv(x)  # size = (b,c2, h,w) = (1,32,256,256)
        K = self.key_conv(x)  # size = (b, c2, h, w) = (1,32,256,256)
        V = self.value_conv(x)  # size = (b, c1,h,w) = (1,16,256,256)

        Q = Q.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h).permute(0, 2, 1)  # size = (b*w,h,c2) = (256, 256, 32)
        K = K.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)  # size = (b*w,c2,h) = (256, 32 256)
        V = V.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)  # size = (b*w,c1,h) = (256, 16 256)

        # size = (b*w,h,h) [:,i,j] 表示Q的所有W的第 Hi行位置上所有通道值与 K的所有W的第 Hj列位置上的所有通道值的乘积，
        # 即(1,c2) * (c2,1) = (1,1)
        col_attn = torch.bmm(Q, K)  #size = (256,256,256)
        ########
        # 此时的 col_atten的[:,i,0:w] 表示Q的所有W的第 Hi行位置上所有通道值与 K的所有W的 所有列(0:h)的逐个位置上的所有通道值的乘积
        # 此操作即为 Q的某个（i,j）与 K的（i,0:h）逐个位置的值的乘积，得到列attn
        ########

        # 对row_attn进行softmax
        col_attn = self.softmax(col_attn)  # 对列进行softmax，即[k,i,0:w] ，某一行的所有列加起来等于1，

        # size = (b*w,c1,h) 这里先需要对col_atten进行 行列置换，使得某一列的所有行加起来等于1
        # [:,i,j]即为V的所有行的某个通道上，所有列的值 与 col_attn的行的乘积，即求权重和
        out = torch.bmm(V, col_attn.permute(0, 2, 1))

        # size = (b,c1,h,w)
        out = out.view(b, w, -1, h).permute(0, 2, 3, 1)  #[4, 32, 64, 64]

        out = self.gamma * out + x   #size = (1,16,256,256)     [1, 64, 256, 256]

        return out