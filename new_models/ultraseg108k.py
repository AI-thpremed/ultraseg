import torch
from torch import nn
import torch.nn.functional as F
import math
from thop import profile, clever_format
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


class EnhancedDilatedBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        key=3
        self.dilated_convs = nn.ModuleList([
            nn.Conv2d(channels, channels//3, 3, padding=1, dilation=1, groups=channels//key),
            nn.Conv2d(channels, channels//3, 3, padding=2, dilation=2, groups=channels//key),
            nn.Conv2d(channels, channels//3, 3, padding=3, dilation=3, groups=channels//key),
        ])
        self.fusion = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        features = []
        for conv in self.dilated_convs:
            features.append(conv(x))
        
        fused = torch.cat(features, dim=1)
        return x + self.fusion(fused) * 0.1

class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
    

class Down(nn.Sequential):
    def __init__(self, in_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2)
    
    def forward(self, x):
        return self.conv(self.bn(x))

class ConvLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=7, padding=3, stride=1, groups=dim, padding_mode='reflect') # depthwise conv
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, 4 * dim, kernel_size=1, padding=0, stride=1)
        self.act1 = nn.GELU()
        self.norm2 = nn.BatchNorm2d(dim)
        self.conv3 = nn.Conv2d(4 * dim, dim, kernel_size=1, padding=0, stride=1)
        self.act2 = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.act1(x)
        x = self.conv3(x)
        x = self.norm2(x)
        x = self.act2(x)
        return x
    
    
class Boundary_Prediction_Generator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)

    def forward(self, x):
        boundary = torch.sigmoid(self.conv(x))
        x = x + x * boundary
        return x, boundary

class Image_Prediction_Generator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)

    def forward(self, x):
        gt_pre = self.conv(x)
        x = x + x * torch.sigmoid(gt_pre)
        return x, gt_pre

class Prediction_Generator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)

    def forward(self, x):
        boundary = torch.sigmoid(self.conv1(x))
        gt_pre = self.conv2(x)
        return (x + x * boundary + x * torch.sigmoid(gt_pre)), gt_pre, boundary


class Group_shuffle_block(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        if dim_in % 4 != 0:
            dim_in = (dim_in // 4) * 4
        c_dim = dim_in // 4

        self.share_space1 = nn.Parameter(torch.Tensor(1, c_dim, 8, 8), requires_grad=True)
        nn.init.ones_(self.share_space1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(c_dim, c_dim, kernel_size=3, padding=1, groups=c_dim),
            nn.GELU(),
            nn.Conv2d(c_dim, c_dim, 1)
        )
        self.share_space2 = nn.Parameter(torch.Tensor(1, c_dim, 8, 8), requires_grad=True)
        nn.init.ones_(self.share_space2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c_dim, c_dim, kernel_size=3, padding=1, groups=c_dim),
            nn.GELU(),
            nn.Conv2d(c_dim, c_dim, 1)
        )
        self.share_space3 = nn.Parameter(torch.Tensor(1, c_dim, 8, 8), requires_grad=True)
        nn.init.ones_(self.share_space3)
        self.conv3 = nn.Sequential(
            nn.Conv2d(c_dim, c_dim, kernel_size=3, padding=1, groups=c_dim),
            nn.GELU(),
            nn.Conv2d(c_dim, c_dim, 1)
        )
        self.share_space4 = nn.Parameter(torch.Tensor(1, c_dim, 8, 8), requires_grad=True)
        nn.init.ones_(self.share_space4)
        self.conv4 = nn.Sequential(
            nn.Conv2d(c_dim, c_dim, kernel_size=3, padding=1, groups=c_dim),
            nn.GELU(),
            nn.Conv2d(c_dim, c_dim, 1)
        )

        self.norm1 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')
        self.norm2 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')
        
        self.ldw = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1, groups=dim_in),
            nn.GELU(),
            nn.Conv2d(dim_in, dim_out, 1),
        )

    def forward(self, x):
        x = self.norm1(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        B, C, H, W = x1.size()
        x1 = x1 * self.conv1(F.interpolate(self.share_space1, size=x1.shape[2:4],mode='bilinear', align_corners=True))
        x2 = x2 * self.conv2(F.interpolate(self.share_space2, size=x1.shape[2:4],mode='bilinear', align_corners=True))
        x3 = x3 * self.conv3(F.interpolate(self.share_space3, size=x1.shape[2:4],mode='bilinear', align_corners=True))
        x4 = x4 * self.conv4(F.interpolate(self.share_space4, size=x1.shape[2:4],mode='bilinear', align_corners=True))
        x = torch.cat([x2,x4,x1,x3], dim=1)
        x = self.norm2(x)
        x = self.ldw(x)
        return x
    


class AdaptiveMerge(nn.Module):
    def __init__(self, dim_in, use_boundary=True):
        super().__init__()
        self.use_boundary = use_boundary
        self.region_weight = nn.Parameter(torch.tensor(0.3))
        if self.use_boundary:
            self.boundary_weight = nn.Parameter(torch.tensor(0.1))
        self.feature_adapter = nn.Sequential(
            nn.Conv2d(dim_in, max(dim_in//8, 4), 1),
            nn.ReLU(),
            nn.Conv2d(max(dim_in//8, 4), dim_in, 1),
        ) if dim_in > 16 else nn.Identity()
        
    def forward(self, x1, x2, gt_pre, boundary_pre=None):
        x1_adapted = self.feature_adapter(x1)
        if gt_pre.size(1) == 1:
            gt_pre = gt_pre.expand(-1, x2.size(1), -1, -1)
        
        if self.use_boundary and boundary_pre is not None:
            if boundary_pre.size(1) == 1:
                boundary_pre = boundary_pre.expand(-1, x2.size(1), -1, -1)
        
        base_fusion = x1_adapted + x2
        region_contrib = torch.sigmoid(gt_pre) * x2 * self.region_weight
        
        boundary_contrib = 0
        if self.use_boundary and boundary_pre is not None:
            boundary_contrib = boundary_pre * x2 * self.boundary_weight
        
        return base_fusion + region_contrib + boundary_contrib



class UltraSeg108(nn.Module):
    def __init__(self, out_ch=2, in_ch=3):
        super().__init__()
        c_list = [8, 16, 48, 64, 96] 
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_ch, c_list[0], 3, stride=1, padding=1),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
        ) 

        self.encoder3 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[2], 3, padding=1),  # 16->48
            EnhancedDilatedBlock(c_list[2]),  
            EnhancedDilatedBlock(c_list[2]),  
        )
        
        self.encoder4 = nn.Sequential(
            Group_shuffle_block(c_list[2], c_list[3]), 
        )        

        self.bottleneck = nn.Conv2d(c_list[3], c_list[4], 3, padding=1)  

        self.Down1 = Down(c_list[0])
        self.Down2 = Down(c_list[1])
        self.Down3 = Down(c_list[2])

        self.merge1 = AdaptiveMerge(c_list[0], use_boundary=True)
        self.merge2 = AdaptiveMerge(c_list[1], use_boundary=True)
        self.merge3 = AdaptiveMerge(c_list[2], use_boundary=True)
        self.merge4 = AdaptiveMerge(c_list[3], use_boundary=False)

        self.decoder1 = nn.Sequential(
            Group_shuffle_block(c_list[4], c_list[3]),  # 128->64
        ) 
        self.decoder2 = nn.Sequential(
            Group_shuffle_block(c_list[3], c_list[2]),  # 64->32
        ) 
        self.decoder3 = nn.Sequential(
            Group_shuffle_block(c_list[2], c_list[1]),  # 32->16
        )  
        self.decoder4 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1),  # 16->8
        )  

        self.pred1 = Image_Prediction_Generator(c_list[3])
        self.gate1 = Prediction_Generator(c_list[2])
        self.gate2 = Prediction_Generator(c_list[1])
        self.gate3 = Prediction_Generator(c_list[0])

        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        
        self.dbn1 = nn.GroupNorm(4, c_list[3])
        self.dbn2 = nn.GroupNorm(4, c_list[2])
        self.dbn3 = nn.GroupNorm(4, c_list[1])
        self.dbn4 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Sequential(
            nn.Conv2d(c_list[0], out_ch, kernel_size=1),
        )

        self.apply(self._init_weights)



    def forward(self, x):
        out = self.encoder1(x)
        out = F.gelu(self.Down1(self.ebn1(out)))
        t1 = out  # b, 8, 128, 128

        out = self.encoder2(out)
        out = F.gelu(self.Down2(self.ebn2(out)))
        t2 = out  # b, 16, 64, 64

        out = self.encoder3(out)
        out = F.gelu(self.Down3(self.ebn3(out)))
        t3 = out  # b, 32, 32, 32

        out = self.encoder4(out)  
        out = F.gelu(F.max_pool2d(self.ebn4(out), 2))
        t4 = out  # b, 64, 16, 16
        
        bottleneck = self.bottleneck(out)  

        out = self.decoder1(bottleneck) 
        out = F.gelu(self.dbn1(out))  

        out, gt_pre4 = self.pred1(out)
        out = self.merge4(out, t4, gt_pre4)  
        gt_pre4 = F.interpolate(gt_pre4, scale_factor=16, mode='bilinear', align_corners=True)
        
        out = self.decoder2(out)  # 64->32
        out = F.gelu(F.interpolate(self.dbn2(out), scale_factor=(2,2), mode='bilinear', align_corners=True))  # b, 32, 32, 32
        out, gt_pre3, weight1 = self.gate1(out)
        out = self.merge3(out, t3, gt_pre3, weight1)  
        weight1 = F.interpolate(weight1, scale_factor=8, mode='bilinear', align_corners=True)
        gt_pre3 = F.interpolate(gt_pre3, scale_factor=8, mode='bilinear', align_corners=True)
        
        out = self.decoder3(out)  # 32->16
        out = F.gelu(F.interpolate(self.dbn3(out), scale_factor=(2,2), mode='bilinear', align_corners=True))  # b, 16, 64, 64
        out, gt_pre2, weight2 = self.gate2(out)
        out = self.merge2(out, t2, gt_pre2, weight2)  
        weight2 = F.interpolate(weight2, scale_factor=4, mode='bilinear', align_corners=True)
        gt_pre2 = F.interpolate(gt_pre2, scale_factor=4, mode='bilinear', align_corners=True)
        
        out = self.decoder4(out)  # 16->8
        out = F.gelu(F.interpolate(self.dbn4(out), scale_factor=(2,2), mode='bilinear', align_corners=True))  # b, 8, 128, 128
        out, gt_pre1, weight3 = self.gate3(out)
        out = self.merge1(out, t1, gt_pre1, weight3)  
        weight3 = F.interpolate(weight3, scale_factor=2, mode='bilinear', align_corners=True)
        gt_pre1 = F.interpolate(gt_pre1, scale_factor=2, mode='bilinear', align_corners=True)
        
        out = self.final(out)
        out = F.interpolate(out, scale_factor=(2,2), mode='bilinear', align_corners=True)  # b, num_class, 256, 256

        gt_pre1 = torch.sigmoid(gt_pre1)
        gt_pre2 = torch.sigmoid(gt_pre2)
        gt_pre3 = torch.sigmoid(gt_pre3)
        gt_pre4 = torch.sigmoid(gt_pre4)

        return (gt_pre4, gt_pre3, gt_pre2, gt_pre1), (weight1, weight2, weight3), out

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

if __name__ == "__main__":
    model = UltraSeg108(in_ch=3, out_ch=4)
    input_tensor = torch.randn(1, 3, 256, 256)

    # 测试前向传播
    try:
        with torch.no_grad():
            outputs = model(input_tensor)
        print("Forward pass successful!")
        
        flops, params = profile(model, inputs=(input_tensor, ), verbose=False)
        flops, params = clever_format([flops, params], "%.8f")
        print(f"FLOPs: {flops}")
        print(f"Params: {params}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

