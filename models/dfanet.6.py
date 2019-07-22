import torch
from torch import nn
import time
class dsc_with_bn(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, padding=0, dilation=1, bias=True):
        super(dsc_with_bn, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, ksize, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        output = self.conv(x)
        output = self.pointwise(output)
        output = self.bn(output)
        return output

class fca(nn.Module):
    def __init__(self, num_classes=1000, in_channels=192, out_channels=192, dropout_prob=0.01, bias=True):
        super(fca,self).__init__()
        self.classes = num_classes
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc= nn.Linear(in_channels,num_classes,bias=bias)
        self.conv = nn.Conv2d(num_classes, out_channels, 1, bias=bias)
        self.bna = nn.Sequential(nn.BatchNorm2d(out_channels),nn.PReLU())
        self.regular = nn.Dropout2d(p=dropout_prob)
        self.out_activition = nn.ReLU()
    def forward(self, x):
        shape=x.shape
        output = self.pool(x).view(-1,1,1,shape[1])
        output = self.fc(output).view(-1,self.classes,1,1)
        output = self.conv(output)
        output = self.bna(output)
        output = self.regular(output)
        output = output.expand_as(x)*x
        output = self.out_activition(output)
        return  output

class block(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, downsampling=False, relu_first=False):
        super(block, self).__init__()
        conv1 = dsc_with_bn(in_channels,out_channels[0],ksize=3,stride=1,padding=1,dilation=1,bias=bias)
        conv2 = dsc_with_bn(out_channels[0],out_channels[1],ksize=3,stride=1,padding=1,dilation=1,bias=bias)
        conv3 = dsc_with_bn(out_channels[1],out_channels[2],ksize=3,stride=1,padding=1,dilation=1,bias=bias)
        maxpool = nn.MaxPool2d(kernel_size=3,stride=2 if downsampling else 1,padding=1,dilation=1,return_indices=False)
        self.main = nn.Sequential(nn.Conv2d(in_channels,out_channels[2],kernel_size=3,stride=2 if downsampling else 1,padding=1,dilation=1,groups=1,bias=bias),
                    nn.BatchNorm2d(out_channels[2]))
        if relu_first:
            self.ext = nn.Sequential(nn.ReLU(),conv1,nn.ReLU(),conv2,nn.ReLU(),conv3, maxpool)
        else:
            self.ext = nn.Sequential(conv1,nn.ReLU(),conv2,nn.ReLU(),conv3, maxpool)
    def forward(self,x):
        m = self.main(x[0])
        e = self.ext(x[1]) + m
        return (m, e)

class enc(nn.Module):
    def __init__(self, in_channels, bias=False, relu_first=True, stage=2):
        super(enc, self).__init__()
        out_channels = [[12, 12, 48],[24,24,96],[48,48,192]]
        out_channels = out_channels[stage-2]
        repeats = [4, 6, 4]
        repeats = repeats[stage-2]
        blocks = []
        for i in range(repeats):
            if i == 0:
                blocks.append(block(in_channels,out_channels,bias=bias,downsampling=True,relu_first=relu_first))
            elif i == repeats-1:
                blocks.append(block(out_channels[2],out_channels,bias=bias,downsampling=False,relu_first=True))
            else:
                blocks.append(block(out_channels[2],out_channels,bias=bias,downsampling=False,relu_first=True))
        self.blocks = nn.Sequential(*blocks)
    def forward(self, x):
        o = self.blocks((x,x))
        return o[1]

class XceptionAx3(nn.Module):
    def __init__(self, bias=False):
        super(XceptionAx3, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3,8,3,2,padding=1, dilation=1, groups=1, bias=bias),nn.BatchNorm2d(8),nn.ReLU())
        self.e21 = enc(8,bias,relu_first=False,stage=2)
        self.e31 = enc(48,bias,relu_first=True,stage=3)
        self.e41 = enc(96,bias,relu_first=True,stage=4)
        self.fc1 = fca(bias=False)
        self.e22 = enc(240,bias,relu_first=True,stage=2)
        self.e32 = enc(144,bias,relu_first=True,stage=3)
        self.e42 = enc(288,bias,relu_first=True,stage=4)
        self.fc2 = fca(bias=False)
        self.e23 = enc(240,bias,relu_first=True,stage=2)
        self.e33 = enc(144,bias,relu_first=True,stage=3)
        self.e43 = enc(288,bias,relu_first=True,stage=4)
        self.fc3 = fca(bias=False)
    def forward(self,x):
        out_conv1 = self.conv1(x)
        out_enc21 = self.e21(out_conv1)
        out_enc31 = self.e31(out_enc21)
        out_enc41 = self.e41(out_enc31)
        out_fca1 = self.fc1(out_enc41)
        up1 = nn.functional.interpolate(out_fca1,scale_factor=(4,4), mode='bilinear',align_corners=False)
        # --------------------------------------------------------------------------------------
        x2 = torch.cat((out_enc21,up1), dim=1)
        out_enc22 = self.e22(x2)
        x2 = torch.cat((out_enc31,out_enc22), dim=1)
        out_enc32 = self.e32(x2)
        x2 = torch.cat((out_enc41,out_enc32), dim=1)
        out_enc42 = self.e42(x2)
        out_fca2 = self.fc2(out_enc42)
        up2 = nn.functional.interpolate(out_fca2, scale_factor=(4,4), mode='bilinear',align_corners=False)
        # --------------------------------------------------------------------------------------
        x3 = torch.cat((out_enc22,up2),dim=1)
        out_enc23 = self.e23(x3)
        x3 = torch.cat((out_enc32,out_enc23),dim=1)
        out_enc33 = self.e33(x3)
        x3 = torch.cat((out_enc42,out_enc33),dim=1)
        out_enc43 = self.e43(x3)
        out_fca3 = self.fc3(out_enc43)
        return out_enc21, out_enc22, out_enc23, out_fca1, out_fca2, out_fca3

class decoder(nn.Module):
    def __init__(self, num_classes=19, bias=True, dropout_prob=0):
        super(decoder, self).__init__()
        self.convt1 = nn.Sequential(nn.ConvTranspose2d(48, num_classes, kernel_size=3, stride=1, padding=1, output_padding=0, bias=bias, dilation=1),nn.BatchNorm2d(num_classes))
        self.convt2 = nn.Sequential(nn.ConvTranspose2d(48, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1, bias=bias, dilation=1),nn.BatchNorm2d(num_classes))
        self.convt3 = nn.Sequential(nn.ConvTranspose2d(48, 48, kernel_size=3, stride=2, padding=1, output_padding=1, bias=bias, dilation=1),
                     nn.ConvTranspose2d(48, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1, bias=bias, dilation=1),nn.BatchNorm2d(num_classes))
        self.convt4 = nn.Sequential(nn.ConvTranspose2d(192, 48, kernel_size=3, stride=2, padding=1, output_padding=1, bias=bias, dilation=1),
                     nn.ConvTranspose2d(48, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1, bias=bias, dilation=1),nn.BatchNorm2d(num_classes))
        self.convt5 = nn.Sequential(nn.ConvTranspose2d(192, 48, kernel_size=3, stride=2, padding=1, output_padding=1, bias=bias, dilation=1),
                     nn.ConvTranspose2d(48, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1, bias=bias, dilation=1),nn.BatchNorm2d(num_classes))
        self.convt6 = nn.Sequential(nn.ConvTranspose2d(192, 48, kernel_size=3, stride=2, padding=1, output_padding=1, bias=bias, dilation=1),
                     nn.ConvTranspose2d(48, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1, bias=bias, dilation=1),nn.BatchNorm2d(num_classes))
        self.act = nn.ReLU()
        self.conv = nn.Sequential(nn.Conv2d(num_classes,num_classes,kernel_size=1,stride=1,padding=0,bias=bias),nn.BatchNorm2d(num_classes),nn.ReLU())
    def forward(self,x):
        x1 = self.convt1(x[0])
        x2 = self.convt2(x[1])
        x3 = self.convt3(x[2])
        x4 = self.convt4(x[3])
        x5 = self.convt5(x[4])
        x5 = nn.functional.interpolate(x5, scale_factor=(2,2), mode='bilinear', align_corners=False)
        x6 = self.convt6(x[5])
        x6 = nn.functional.interpolate(x6, scale_factor=(4,4), mode='bilinear', align_corners=False)
        out = self.conv(self.act(x1+x2+x3))
        out = self.act(out+x4+x5+x6)
        out = nn.functional.interpolate(out, scale_factor=(4,4),mode='bilinear',align_corners=False)
        return out

class DFANet(nn.Module):
    def __init__(self, num_classes=19, bias = True):
        super(DFANet,self).__init__()
        self.backbonex3 = XceptionAx3(bias)
        self.decoder = decoder(num_classes,bias)
    def forward(self, x):
        out = self.backbonex3(x)
        out = self.decoder(out)
        return out

if __name__=='__main__':
    '''
    b1 = block2(8,bias=False)
    b2 = block2(48, bias=False)
    e1 = enc2(144, False)
    for name, child in e1.named_children():
        print("==",name)
        for name, cchild in child.named_children():
            print("====", name)
            for name, ccchild in cchild.named_children():
                print('======', name)
                for name, cccchild in ccchild.named_children():
                    print('========', name)
                    for name, ccx in cccchild.named_parameters():
                        print(name)
                        print(ccx.shape)
    '''

        