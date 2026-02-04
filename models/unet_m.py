import torch, torch.nn as nn

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch

from thop import profile, clever_format










class DSConv(nn.Module):                       # Depthwise-Separable
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_in, k, s, p, groups=c_in, bias=False),
            nn.BatchNorm2d(c_in), nn.ReLU6(True),
            nn.Conv2d(c_in, c_out, 1, bias=False),
            nn.BatchNorm2d(c_out), nn.ReLU6(True)
        )
    def forward(self, x): return self.conv(x)

class Block(nn.Module):                        # 同 U-Net 的 conv_block
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Sequential(
            DSConv(c_in, c_out),
            DSConv(c_out, c_out)
        )
    def forward(self, x): return self.conv(x)

class UNet_M(nn.Module):
    def __init__(self, in_ch=3, out_ch=4):
        super().__init__()

        # base = [64,128,256,512,1024]
#medium 32-64-128-256-512
#  中间层 c = [24,48,96,192,384]
        #small c = [16, 32, 64, 128, 256]            
        #tiny c = [8, 16, 32, 64, 128]        

        c = [32,64,128,256,512]
        # = [24, 48, 96, 192, 384]
        self.pool = nn.MaxPool2d(2)
        # encoder
        self.enc0 = Block(in_ch, c[0])
        self.enc1 = Block(c[0], c[1])
        self.enc2 = Block(c[1], c[2])
        self.enc3 = Block(c[2], c[3])
        self.enc4 = Block(c[3], c[4])
        # decoder
        self.up4 = nn.ConvTranspose2d(c[4], c[3], 2, 2)
        self.dec4 = Block(c[4], c[3])

        self.up3 = nn.ConvTranspose2d(c[3], c[2], 2, 2)
        self.dec3 = Block(c[3], c[2])

        self.up2 = nn.ConvTranspose2d(c[2], c[1], 2, 2)
        self.dec2 = Block(c[2], c[1])

        self.up1 = nn.ConvTranspose2d(c[1], c[0], 2, 2)
        self.dec1 = Block(c[1], c[0])

        self.head = nn.Conv2d(c[0], out_ch, 1)

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        d4 = self.dec4(torch.cat([self.up4(e4), e3], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e2], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e0], 1))
        return self.head(d1)



if __name__ == "__main__":

    model = UNet_M(in_ch=3, out_ch=4)


    input_tensor = torch.randn(1, 3, 256, 256)


    flops, params = profile(model, inputs=(input_tensor, ), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")

    print(f"FLOPs: {flops}")
    print(f"Params: {params}")
