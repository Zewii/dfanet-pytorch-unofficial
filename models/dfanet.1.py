import torch
from torch.nn.modules import Module
from torch.nn.modules import Conv2d
from torch.nn.modules import Linear
from torch.nn.modules import AdaptiveAvgPool2d
from torch.nn.modules import BatchNorm2d
from torch.nn.modules import ReLU
from torch.nn.modules import Sequential
from torch.nn.functional import interpolate
from torchsummary import summary

class bna(Module):
    def __init__(self, features):
        super(bna, self).__init__()
        self.batchnorm = BatchNorm2d(features)
        self.activate = ReLU(inplace=True)

    def forward(self, x):
        out = self.batchnorm(x)
        out = self.activate(out)
        return out

class dsc(Module):
    def __init__(self, in_channels, out_channels, ksize=3, stride=2, padding=1, bias=True):
        super(dsc,self).__init__()
        self.conv1 = Conv2d(in_channels=in_channels, out_channels=in_channels,
                            kernel_size=ksize,stride=stride,padding=padding,bias=bias,groups=in_channels)
        self.bna1 = bna(in_channels)
        self.conv2 = Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,groups=1,bias=bias)
        self.bna2 = bna(out_channels)
    def forward(self, x):
        x=self.conv1(x)
        x=self.bna1(x)
        x=self.conv2(x)
        y=self.bna2(x)
        return y

class block(Module):
    def __init__(self,in_channels,out_channels,stride,bias):
        super(block,self).__init__()
        self.conv1 = dsc(in_channels,out_channels[0],3,stride=stride,bias=bias)
        self.conv2 = dsc(in_channels,out_channels[1],3,stride=stride,bias=bias)
        self.conv3 = dsc(in_channels,out_channels[2],3,stride=stride,bias=bias)
    def forward(self, x):
        y1= self.conv1(x)
        y2= self.conv2(x)
        y3= self.conv3(x)
        y= torch.cat((y1,y2,y3),dim=1)
        return y

class enc(Module):
    def __init__(self,in_channels,stage=2,mode='A',bias=True):
        super(enc,self).__init__()
        assert stage in [2,3,4]
        repeats = [4, 6, 4]
        channels = [[12,12,48],[24,24,96],[48,48,192]] if mode=='A' else [[8,8,32],[16,16,64],[32,32,128]]
        channel = channels[stage-2]
        inter_channels=in_channels
        ops = []
        for i in range(repeats[stage-2]):
            ops.append(block(inter_channels,channel,stride=2 if i==1 else 1,bias=bias))
            inter_channels = sum(channel)
        self.ops = Sequential(*ops)

    def forward(self,x):
        y = self.ops(x)
        return y
    def get_parameters(self):
        pass

class fca(Module):
    def __init__(self, in_channels, out_channels,bias):
        super(fca,self).__init__()
        self.pool = AdaptiveAvgPool2d(1)
        self.fc= Linear(in_channels,1000,bias=bias)
        self.conv = Conv2d(1000, out_channels, 1, bias=bias)
        self.bna = bna(out_channels)
    def forward(self, x):
        shape=x.shape
        y= self.pool(x).view(-1,1,1,shape[1])
        y= self.fc(y).view(-1,1000,1,1)
        y= self.conv(y)
        y= self.bna(y)
        y = y.expand_as(x)*x
        return  y
    def get_parameters(self):
        pass

class DFANet(Module):
    def __init__(self,in_channels,n_classes=13,bias=True,mode='B'):
        super(DFANet,self).__init__()
        channels={'A':72,'B':48}
        ch=channels[mode]
        self.conv1= Sequential(Conv2d(in_channels,8,3,2,1,bias=bias),bna(8))
        self.enc2_1 = enc(in_channels=8,stage=2,mode=mode,bias=bias)
        self.enc3_1 = enc(in_channels=ch, stage=3, mode=mode,bias=bias)
        self.enc4_1 = enc(in_channels=ch*2, stage=4, mode=mode, bias=bias)
        self.fca1 = fca(ch*4, ch*4, bias=bias)
        self.enc2_2 = enc(in_channels=ch*5, stage=2, mode=mode, bias=bias)
        self.enc3_2 = enc(in_channels=ch*3, stage=3, mode=mode, bias=bias)
        self.enc4_2 = enc(in_channels=ch*6, stage=4, mode=mode, bias=bias)
        self.fca2 = fca(ch*4, ch*4, bias=bias)
        self.enc2_3 = enc(in_channels=ch*5, stage=2, mode=mode, bias=bias)
        self.enc3_3 = enc(in_channels=ch*3, stage=3, mode=mode, bias=bias)
        self.enc4_3 = enc(in_channels=ch*6, stage=4, mode=mode, bias=bias)
        self.fca3 = fca(ch*4, ch*4, bias=bias)
        self.de2_1 = Sequential(Conv2d(ch,ch//2,1,bias=bias),bna(ch//2))
        self.de2_2 = Sequential(Conv2d(ch,ch//2,1,bias=bias),bna(ch//2))
        self.de2_3 = Sequential(Conv2d(ch,ch//2,1,bias=bias),bna(ch//2))
        self.final = Sequential(Conv2d(ch//2,n_classes,1,bias=bias),bna(n_classes))
        self.de4_1 = Sequential(Conv2d(ch*4,n_classes,1,bias=bias),bna(n_classes))
        self.de4_2 = Sequential(Conv2d(ch*4,n_classes,1,bias=bias),bna(n_classes))
        self.de4_3 = Sequential(Conv2d(ch*4,n_classes,1,bias=bias),bna(n_classes))
    def forward(self,x):
        oenc2_1 = self.conv1(x)
        oenc2_1 = self.enc2_1(oenc2_1)
        oenc3_1 = self.enc3_1(oenc2_1)
        oenc4_1 = self.enc4_1(oenc3_1)
        ofc1 = self.fca1(oenc4_1)
        up1 = interpolate(ofc1, scale_factor=(4,4),mode='bilinear', align_corners=False)
        ienc2_2 = torch.cat((oenc2_1,up1),dim=1)
        oenc2_2 = self.enc2_2(ienc2_2)
        ienc3_2 = torch.cat((oenc2_2,oenc3_1), dim=1)
        oenc3_2 = self.enc3_2(ienc3_2)
        ienc4_2 = torch.cat((oenc3_2,oenc4_1),dim=1)
        oenc4_2 = self.enc4_2(ienc4_2)
        ofc2 = self.fca2(oenc4_2)
        up2 = interpolate(ofc2, scale_factor=(4,4),mode='bilinear', align_corners=False)
        ienc2_3 = torch.cat((oenc2_2,up2),dim=1)
        oenc2_3 = self.enc2_3(ienc2_3)
        ienc3_3 = torch.cat((oenc2_3,oenc3_2),dim=1)
        oenc3_3 = self.enc3_3(ienc3_3)
        ienc4_3 = torch.cat((oenc3_3,oenc4_2),dim=1)
        oenc4_3 = self.enc4_3(ienc4_3)
        ofc3 = self.fca3(oenc4_3)
        ode2_1 = self.de2_1(oenc2_1)
        ode2_2 = self.de2_2(oenc2_2)
        ode2_2 = interpolate(ode2_2,scale_factor=(2,2),mode='bilinear',align_corners=False)
        ode2_3 = self.de2_3(oenc2_3)
        ode2_3 = interpolate(ode2_3,scale_factor=(4,4),mode='bilinear',align_corners=False)
        ode4_1 = self.de4_1(ofc1)
        ode4_1 = interpolate(ode4_1,scale_factor=(4,4),mode='bilinear',align_corners=False)
        ode4_2 = self.de4_2(ofc2)
        ode4_2 = interpolate(ode4_2,scale_factor=(8,8),mode='bilinear',align_corners=False)
        ode4_3 = self.de4_3(ofc3)
        ode4_3 = interpolate(ode4_3,scale_factor=(16,16),mode='bilinear',align_corners=False)
        ifinal = ode2_1+ode2_2+ode2_3
        ofinal = self.final(ifinal)
        ofinal = ofinal+ode4_1+ode4_2+ode4_3
        ofinal = interpolate(ofinal,scale_factor=(4,4),mode='bilinear',align_corners=False)
        return ofinal
    def get_parameters(self):
        pass

if __name__=='__main__':
    modelB = DFANet(3,13,False,mode='B')
    modelB.cuda()
    example = torch.rand((2,3,1024,1024)).cuda()
    print(modelB(example).shape)
    summary(modelB, (3,1024,1024))
