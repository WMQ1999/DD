import torch
from torch import nn
import numpy as np
from torch.nn import functional
from torchsummary import summary


class SiamFC(nn.Module):
    def __init__(self):
        super(SiamFC, self).__init__()
        self.netx = AlexNet()
        self.netz = AlexNet()
        #self.norm = nn.BatchNorm2d(1)
        self.corr = Corr()

    def forward(self, x, z):
        outx = self.netx(x)
        outz = self.netx(z)
        out = self.corr(outx, outz)
        return out

class SiamFu(nn.Module):
    def __init__(self):
        super(SiamFu, self).__init__()
        self.netx = AlexNetReAll()
        self.netz = AlexNetReAll()
        #self.norm = nn.BatchNorm2d(1)
        self.corr = Corr()
        #self.upsample = nn.UpsamplingBilinear2d(scale_factor=(2,2))
        self.upsample = nn.Upsample(scale_factor=(2,2), mode= 'bicubic')
    def forward(self, x, z):
        x1, x2 = self.netx(x)
        z1, z2 = self.netx(z)
        x2 = self.upsample(x2)
        z2 = self.upsample(z2)
        x1_sz = x1.shape[-1]
        x2_sz = x2.shape[-1]
        x_pad = (x1_sz - x2_sz) // 2
        x2 = nn.functional.pad(x2, (x_pad, x1_sz - x_pad - x2_sz, x_pad, x1_sz - x_pad - x2_sz))
        x_fu = torch.cat([x1, x2], dim = 1)
        z1_sz = z1.shape[-1]
        z2_sz = z2.shape[-1]
        z_pad = (z1_sz - z2_sz) // 2
        z2 = nn.functional.pad(z2, (z_pad, z1_sz - z_pad - z2_sz, z_pad, z1_sz - z_pad - z2_sz))
        z_fu = torch.cat([z1, z2], dim = 1)
        out = self.corr(x_fu, z_fu)
        # print(x2.shape, z2.shape, x_fu.shape, z_fu.shape)
        return out

class SiamFuV2(nn.Module):
    def __init__(self):
        super(SiamFuV2, self).__init__()
        self.netx = AlexNetFusion()
        self.netz = AlexNetFusion()
        #self.norm = nn.BatchNorm2d(1)
        self.corr = Corr()
        #self.upsample = nn.UpsamplingBilinear2d(scale_factor=(2,2))
        # self.upsample1 = nn.Upsample(size=(13, 13), mode= 'bicubic')
        # self.upsample2 = nn.Upsample(size=(29, 29), mode= 'bicubic')
        self.pool = nn.MaxPool2d(kernel_size=(7, 7), stride = 1)
        # self.softmax = nn.Softmax2d()
        # self.bn1 = nn.BatchNorm2d(1)
        # self.bn2 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(512, 256, 1)
        self.conv2 = nn.Conv2d(512, 256, 1)
    def forward(self, x, z):
        x1, x2 = self.netx(x)
        z1, z2 = self.netx(z)
        # x1 = self.conv1(x1)
        # z1 = self.conv2(z1)
        # x2 = self.upsample2(x2)
        # z2 = self.upsample1(z2)
        x1 = self.pool(x1)
        z1 = self.pool(z1)

        x_fu = torch.cat([x1, x2], dim = 1)
        x_fu = self.conv2(x_fu)
        z_fu = torch.cat([z1, z2], dim = 1)
        z_fu  =self.conv1(z_fu)
        out = self.corr(x_fu, z_fu)
        # print(x2.shape, z2.shape, x_fu.shape, z_fu.shape)
        return out

class AlexNetFusion(nn.Module):
    def __init__(self):
        super(AlexNetFusion, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 3, 2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            nn.BatchNorm2d(384),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 192, 3, 1),
            nn.BatchNorm2d(192),
            nn.ReLU()
        )
        self.conv5 =nn.Sequential(
            nn.Conv2d(192, 256, 3, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out5 = self.conv5(out4)
        return out2, out5
        

class AlexNetReAll(nn.Module):
    def __init__(self):
        super(AlexNetReAll, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 3, 2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            nn.BatchNorm2d(384),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 192, 3, 1),
            nn.BatchNorm2d(192),
            nn.ReLU()
        )
        self.conv5 =nn.Sequential(
            nn.Conv2d(192, 256, 3, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out5 = self.conv5(out4)
        return out1, out5
        

class Corr(nn.Module):
    def __init__(self):
        super(Corr, self).__init__()
        self.adjust = nn.Conv2d(1, 1, 1)
        #self.adjust = nn.ReLU()
        #self.adjust = nn.BatchNorm2d(1)
    def forward(self, x, z):
        Bx, Cx, Hx, Wx = x.shape
        Bz, Cz, Hz, Wz = z.shape

        outx = torch.reshape(x, (1, Bx*Cx, Hx, Wx))
        outz = torch.reshape(z, (Bz*Cz, 1, Hz, Wz))
        out = functional.conv2d(outx, outz, groups=Bx*Cx)
        out = torch.cat(torch.split(out, Cz, dim = -3))
        out = torch.sum(out, dim = -3, keepdim = True)
        out = self.adjust(out)
        return out

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 3, 2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            nn.BatchNorm2d(384),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 192, 3, 1),
            nn.BatchNorm2d(192),
            nn.ReLU()
        )
        self.conv5 = nn.Conv2d(192, 256, 3, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        return out

# conv with groups = 2, just like the paper :ImageNet Classification with Deep Convolutional Neural Networks
class AlexNetV2(nn.Module):
    def __init__(self):
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 3, 2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            nn.BatchNorm2d(384),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 192, 3, 1, groups=2),
            nn.BatchNorm2d(192),
            nn.ReLU()
        )
        self.conv5 = nn.Conv2d(192, 256, 3, 1, groups=2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        return out
        
class ConvLayers(nn.Module):
    def __init__(self):
        super(ConvLayers, self).__init__()
        self.conv_stride = np.array([2, 1, 1, 1, 1])
        self.filtergroup_yn = np.array([0, 1, 0, 1, 1], dtype= bool)
        self.bnorm_yn = np.array([1, 1, 1, 1, 0], dtype= bool)
        self.relu_yn = np.array([1, 1, 1, 1, 0], dtype = bool)
        self.pool_stride = np.array([2, 2, 0, 0, 0]) # the newest version siamFc is [2,1,0,0,0]
        self.pool_sz = 3
        self.bnorm_adjust = True
        # self.channels = np.array([3, 96, 256, 192, 198, 128])
        self.channels = np.array([3, 96, ])
        self.kernel_sz = np.array([11, 5, 3, 3, 3])
        self.deepth = 5
        self.convs = nn.ModuleList(self.set_convs())
        self.bns = nn.ModuleList(self.set_bns())
        self.relus = nn.ModuleList(self.set_relus())
        self.maxpools = nn.ModuleList(self.set_maxpools())
    def set_convs(self):
        convs = []
        for i in range(self.deepth):
            if self.filtergroup_yn[i]:
                convs.append(nn.Conv2d(self.channels[i]//2, self.channels[i+1]//2,
                    kernel_size = self.kernel_sz[i], stride=self.conv_stride[i],
                    padding = 0 
                ))
            else:
                convs.append(nn.Conv2d(self.channels[i], self.channels[i+1],
                    kernel_size = self.kernel_sz[i], stride=self.conv_stride[i],
                    padding = 0 
                ))
        return convs

    def set_bns(self):
        bns = []
        for i in range(len(self.bnorm_yn)):
            if self.bnorm_yn[i]:
                bns.append(nn.BatchNorm2d(self.channels[i+1]))
            else:
                bns.append(None)
        return bns

    def set_relus(self):
        relus = []
        for flag in self.relu_yn:
            if flag:
                relus.append(nn.ReLU())
            else:
                relus.append(None)
        return relus

    def set_maxpools(self):
        maxpools = []
        for stride in self.pool_stride:
            if stride > 0:
                maxpools.append(nn.MaxPool2d(3, stride=stride))
            else:
                maxpools.append(None)
        return maxpools
            
    def forward(self, x):
        out = x
        for i in range(self.deepth):
            # print(i)
            # print(out.shape)
            if self.filtergroup_yn[i]:
                
                X0, X1 = torch.chunk(out, 2, dim = -3)
                out0 = self.convs[i](X0)
                out1 = self.convs[i](X1)
                out = torch.cat([out0, out1], dim = -3)
            else:
                out = self.convs[i](out)

            if self.bnorm_yn[i]:
                out = self.bns[i](out)
            
            if self.relu_yn[i]:
                out = self.relus[i](out)

            if self.pool_stride[i] > 0:
                out = self.maxpools[i](out)
        
        return out

# model = SiamFC()
# print(model.state_dict().keys())
# netx = AlexNet()
# print(netx.state_dict().keys())
# img1 = torch.zeros(1, 3, 255, 255)
# img2 = torch.zeros(1,3, 127, 127)
# model.forward(img1, img2)

# model = ConvLayers()
# print(model.state_dict())
# model = siamese_train()
# netx = ConvLayers()


# mkeys, ukeys = netx.load_state_dict(model.state_dict(), False)
# print(mkeys, ukeys)
# model = SiamFu()
# x = torch.randn(1,3, 255, 255)
# z = torch.randn(1,3, 127, 127)
# print(model.forward(x, z).shape)