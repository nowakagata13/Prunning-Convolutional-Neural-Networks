"""Parts of this code were devloped by Lee Jun Hyun, available: https://github.com/LeeJunHyun/Image_Segmentation"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from network import conv_block, up_conv

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_mid, ch_mid2, ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_mid, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_mid2, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in, ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class U_Net_SP(nn.Module):
    def __init__(self,channels_file=None, img_ch=3,output_ch=1):
        super(U_Net_SP,self).__init__()

        with open(channels_file, 'r') as file:
            ch = file.readlines()
        
        ch = [int(c.strip()) for c in ch]
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_mid = ch[1], ch_mid2=ch[2], ch_out=ch[3]) #62
        self.Conv2 = conv_block(ch_in=ch[4], ch_mid = ch[5], ch_mid2=ch[6], ch_out=ch[7]) # 64, 128
        self.Conv3 = conv_block(ch_in=ch[8], ch_mid = ch[9], ch_mid2=ch[10], ch_out=ch[11]) # 128, 256
        self.Conv4 = conv_block(ch_in=ch[12], ch_mid = ch[13], ch_mid2=ch[14], ch_out=ch[15]) # 256, 512
        self.Conv5 = conv_block(ch_in=ch[16], ch_mid = ch[17], ch_mid2=ch[18], ch_out=ch[19]) # 512, 1024

        self.Up5 = up_conv(ch_in=ch[20],ch_out=ch[21]) #1024
        self.Up_conv5 = conv_block(ch_in=ch[22], ch_mid=ch[23], ch_mid2=ch[24], ch_out=ch[25])

        self.Up4 = up_conv(ch_in=ch[26],ch_out=ch[27])
        self.Up_conv4 = conv_block(ch_in=ch[28], ch_mid = ch[29], ch_mid2=ch[30], ch_out=ch[31])
        
        self.Up3 = up_conv(ch_in=ch[32],ch_out=ch[33])
        self.Up_conv3 = conv_block(ch_in=ch[34], ch_mid = ch[35], ch_mid2 = ch[36], ch_out=ch[37])
        
        self.Up2 = up_conv(ch_in=ch[38],ch_out=ch[39])
        self.Up_conv2 = conv_block(ch_in=ch[40], ch_mid = ch[41], ch_mid2 =ch[42], ch_out=ch[43])

        self.Conv_1x1 = nn.Conv2d(ch[44],output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
