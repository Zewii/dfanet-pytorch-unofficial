import torch
from torch import nn
import time
import cv2
class dsc(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, padding=0, dilation=1, bias=True):
        super(dsc, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, ksize, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=bias)
        self.bna = nn.Sequential(nn.BatchNorm2d(out_channels),nn.PReLU())
    def forward(self, x):
        output = self.conv(x)
        output = self.pointwise(output)
        output = self.bna(output)
        return output

class block2(nn.Module):
    def __init__(self, in_channels, bias=True, dropout_prob=0.01):
        super(block2, self).__init__()
        self.conv1 = dsc(in_channels, 12, ksize=3, stride=1, padding=1, dilation=1, bias=bias)
        self.conv2 = dsc(12, 12, ksize=3, stride=1, padding=1, dilation=1, bias=bias)
        self.conv3 = dsc(12, 48, ksize=3, stride=1, padding=1, dilation=1, bias=bias)
        self.regular = nn.Dropout2d(p=dropout_prob)
    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.regular(output)
        return output

class block3(nn.Module):
    def __init__(self, in_channels, bias=True, dropout_prob=0.01):
        super(block3, self).__init__()
        self.conv1 = dsc(in_channels, 24, ksize=3, stride=1, padding=1, dilation=1, bias=bias)
        self.conv2 = dsc(24, 24, ksize=3, stride=1, padding=1, dilation=1, bias=bias)
        self.conv3 = dsc(24, 96, ksize=3, stride=1, padding=1, dilation=1, bias=bias)
        self.regular = nn.Dropout2d(p=dropout_prob)
    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.regular(output)
        return output

class block4(nn.Module):
    def __init__(self, in_channels, bias=True, dropout_prob=0.01):
        super(block4, self).__init__()
        self.conv1 = dsc(in_channels, 48, ksize=3, stride=1, padding=1, dilation=1, bias=bias)
        self.conv2 = dsc(48, 48, ksize=3, stride=1, padding=1, dilation=1, bias=bias)
        self.conv3 = dsc(48, 192, ksize=3, stride=1, padding=1, dilation=1, bias=bias)
        self.regular = nn.Dropout2d(p=dropout_prob)
    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.regular(output)
        return output

class enc2(nn.Module):
    def __init__(self, bias=True):
        super(enc2, self).__init__()
        self.blocks = nn.Sequential(
            block2(8, bias=bias),
            block2(48, bias=bias),
            block2(48, bias=bias),
            block2(48, bias=bias)
        )
    def forward(self,x):
        output = self.blocks(x)
        return output

class enc3(nn.Module):
    def __init__(self, bias=True):
        super(enc3, self).__init__()
        self.blocks = nn.Sequential(
            block3(48, bias=bias),
            block3(96, bias=bias),
            block3(96, bias=bias),
            block3(96, bias=bias),
            block3(96, bias=bias),
            block3(96, bias=bias)
        )
    def forward(self,x):
        output = self.blocks(x)
        return output

class enc4(nn.Module):
    def __init__(self, bias=True):
        super(enc4, self).__init__()
        self.blocks = nn.Sequential(
            block4(96, bias=bias),
            block4(192, bias=bias),
            block4(192, bias=bias),
            block4(192, bias=bias)
        )
    def forward(self,x):
        output = self.blocks(x)
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
        self.out_activition = nn.PReLU()
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

class XceptionA(nn.Module):
    def __init__(self, bias=False):
        super(XceptionA, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3,8,3,2,padding=1, dilation=1, groups=1, bias=bias),nn.BatchNorm2d(8),nn.PReLU())
        self.e2 = enc2(bias)
        self.e3 = enc3(bias)
        self.e4 = enc4(bias)
        self.fc = fca(num_classes=1000, bias=False)
    def forward(self, x):
        output = self.conv1(x)
        output = self.e2(output)
        output = self.e3(output)
        output = self.e4(output)
        output = self.fc(output)
        return output

class XceptionAx3(nn.Module):
    def __init__(self, bias=False):
        super(XceptionAx3, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3,8,3,2,padding=1, dilation=1, groups=1, bias=bias),nn.BatchNorm2d(8),nn.PReLU())
        self.maxpool = nn.MaxPool2d(2,2,0,1,False)
        self.e21 = enc2(bias)
        self.e31 = enc3(bias)
        self.e41 = enc4(bias)
        self.fc1 = fca(bias=False)
        self.conv2 = nn.Sequential(nn.Conv2d(240,8,3,2,padding=1, dilation=1, groups=1, bias=bias),nn.BatchNorm2d(8),nn.PReLU())
        self.e22 = enc2(bias)
        self.e32 = enc3(bias)
        self.e42 = enc4(bias)
        self.fc2 = fca(bias=False)
        self.conv3 = nn.Sequential(nn.Conv2d(240,8,3,2,padding=1, dilation=1, groups=1, bias=bias),nn.BatchNorm2d(8),nn.PReLU())
        self.e23 = enc2(bias)
        self.e33 = enc3(bias)
        self.e43 = enc4(bias)
        self.fc3 = fca(bias=False)
        self.conv4 = nn.Sequential(nn.Conv2d(48,48,3,2,padding=1, dilation=1, groups=1, bias=bias),nn.BatchNorm2d(48),nn.PReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(144,48,3,2,padding=1, dilation=1, groups=1, bias=bias),nn.BatchNorm2d(48),nn.PReLU())
        self.conv6 = nn.Sequential(nn.Conv2d(144,48,3,2,padding=1, dilation=1, groups=1, bias=bias),nn.BatchNorm2d(48),nn.PReLU())
        self.conv7 = nn.Sequential(nn.Conv2d(96,96,3,2,padding=1, dilation=1, groups=1, bias=bias),nn.BatchNorm2d(96),nn.PReLU())
        self.conv8 = nn.Sequential(nn.Conv2d(288,96,3,2,padding=1, dilation=1, groups=1, bias=bias),nn.BatchNorm2d(96),nn.PReLU())
        self.conv9 = nn.Sequential(nn.Conv2d(288,96,3,2,padding=1, dilation=1, groups=1, bias=bias),nn.BatchNorm2d(96),nn.PReLU())
    def forward(self,x):
        out_conv1 = self.conv1(x)
        out_conv1 = self.maxpool(out_conv1)
        out_enc21 = self.e21(out_conv1)
        out_conv4 = self.conv4(out_enc21)
        out_enc31 = self.e31(out_conv4)
        out_conv7 = self.conv7(out_enc31)
        out_enc41 = self.e41(out_conv7)
        out_fca1 = self.fc1(out_enc41)
        up1 = nn.functional.interpolate(out_fca1,scale_factor=(4,4), mode='bilinear',align_corners=False)
        # --------------------------------------------------------------------------------------
        x2 = torch.cat((out_enc21,up1), dim=1)
        out_conv2 = self.conv2(x2)
        out_enc22 = self.e22(out_conv2)
        x2 = torch.cat((out_enc22,out_enc31), dim=1)
        out_conv5 = self.conv5(x2)
        out_enc32 = self.e32(out_conv5)
        x2 = torch.cat((out_enc32,out_enc41), dim=1)
        out_conv8 = self.conv8(x2)
        out_enc42 = self.e42(out_conv8)
        out_fca2 = self.fc2(out_enc42)
        up2 = nn.functional.interpolate(out_fca2, scale_factor=(4,4), mode='bilinear',align_corners=False)
        # --------------------------------------------------------------------------------------
        x3 = torch.cat((out_enc22,up2),dim=1)
        out_conv3 = self.conv3(x3)
        out_enc23 = self.e23(out_conv3)
        x3 = torch.cat((out_enc23,out_enc32),dim=1)
        out_conv6 = self.conv6(x3)
        out_enc33 = self.e33(out_conv6)
        x3 = torch.cat((out_enc33,out_enc42),dim=1)
        out_conv9 = self.conv9(x3)
        out_enc43 = self.e43(out_conv9)
        out_fca3 = self.fc3(out_enc43)
        return out_enc21, out_enc22, out_enc23, out_fca1, out_fca2, out_fca3

class decoder(nn.Module):
    def __init__(self, num_classes=19, bias=True, dropout_prob=0):
        super(decoder, self).__init__()
        self.convt1 = nn.Sequential(nn.ConvTranspose2d(48, 48, kernel_size=3, stride=1, padding=1, output_padding=0, bias=bias, dilation=1),nn.BatchNorm2d(48),nn.Dropout2d(p=dropout_prob))
        self.convt2 = nn.Sequential(nn.ConvTranspose2d(48, 48, kernel_size=3, stride=2, padding=1, output_padding=1, bias=bias, dilation=1),nn.BatchNorm2d(48),nn.Dropout2d(p=dropout_prob))
        self.convt3 = nn.Sequential(nn.ConvTranspose2d(48, 48, kernel_size=3, stride=4, padding=0, output_padding=1, bias=bias, dilation=1),nn.BatchNorm2d(48),nn.Dropout2d(p=dropout_prob))
        self.convt4 = nn.Sequential(nn.ConvTranspose2d(192, 48, kernel_size=3, stride=4, padding=0, output_padding=1, bias=bias, dilation=1),nn.BatchNorm2d(48),nn.Dropout(p=dropout_prob))
        self.convt5 = nn.Sequential(nn.ConvTranspose2d(192, 48, kernel_size=3, stride=8, padding=0, output_padding=1, bias=bias, dilation=3),nn.BatchNorm2d(48),nn.Dropout(p=dropout_prob))
        self.convt6 = nn.Sequential(nn.ConvTranspose2d(192, 48, kernel_size=3, stride=16, padding=0, output_padding=1, bias=bias, dilation=7),nn.BatchNorm2d(48),nn.Dropout(p=dropout_prob))
        self.act1 = nn.PReLU()
        self.conv = nn.Sequential(nn.Conv2d(48,48,kernel_size=1,stride=1,padding=0,bias=bias),nn.BatchNorm2d(48),nn.PReLU())
        self.act2 = nn.PReLU()
        self.final = nn.Sequential(nn.ConvTranspose2d(48, 48, kernel_size=3, stride=2, padding=1, output_padding=1, bias=bias, dilation=1),
                     nn.ConvTranspose2d(48, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1, bias=bias, dilation=1))
    def forward(self,x):
        x1 = self.convt1(x[0])
        x2 = self.convt2(x[1])
        x3 = self.convt3(x[2])
        x4 = self.convt4(x[3])
        x5 = self.convt5(x[4])
        x6 = self.convt6(x[5])
        out = self.conv(self.act1(x1+x2+x3))
        out = self.final(self.act2(out+x4+x5+x6))
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
    import numpy as np
    img = cv2.imread("3.jpg")
    img = np.array(img).astype('float32')/255.0
    img = torch.Tensor(img).cuda()
    img = torch.unsqueeze(img, 0).view(1,3,1024,1024)
    net = DFANet(19,bias=False)
    net.load_state_dict(torch.load("res/Cityscapes/model_final_15000.pth"))
    net.cuda()
    net.eval()
    s = time.time()
    res = net(img)
    res = res.argmax(dim = 1).cpu().view(1024,1024,1)
    e = time.time()
    res = res.numpy().astype('uint8')*50
    cv2.imshow("r", res)
    cv2.imwrite("strange.jpg", res)
    cv2.waitKey(0)
    print(e-s)
    for i in res:
        print(i.shape)
        