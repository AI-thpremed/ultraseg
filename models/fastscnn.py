# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Fast-SCNN : Fast Semantic Segmentation Network (single-file, PyTorch)
Author  : your_name
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile, clever_format


class _ConvBNReLU(nn.Sequential):
    def __init__(self, in_c, out_c, k=3, s=1, p=None, groups=1):
        if p is None:
            p = k // 2
        super().__init__(
            nn.Conv2d(in_c, out_c, k, s, p, groups=groups, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True)
        )


class _DSConv(nn.Sequential):
    """Depthwise separable convolution"""
    def __init__(self, in_c, out_c, k=3, s=1, p=None):
        if p is None:
            p = k // 2
        super().__init__(
            nn.Conv2d(in_c, in_c, k, s, p, groups=in_c, bias=False),
            nn.BatchNorm2d(in_c),
            nn.ReLU(True),
            nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True)
        )


class _LearningDownsample(nn.Module):
    """Two-branch downsampling (detail & semantic)"""
    def __init__(self):
        super().__init__()
        # low-level detail branch (32× downsample)
        self.detail = nn.Sequential(
            _ConvBNReLU(3, 32, 3, 2),      # /2
            _ConvBNReLU(32, 48, 3, 2)      # /4
        )
        # high-level semantic branch (32× downsample)
        self.semantic = nn.Sequential(
            _ConvBNReLU(3, 32, 3, 2, 1),
            _ConvBNReLU(32, 32, 3, 2, 1),
            _ConvBNReLU(32, 64, 3, 2, 1),
            nn.AvgPool2d(3, 2, 1),         # /8
            nn.AvgPool2d(3, 2, 1),         # /16
            nn.AvgPool2d(3, 2, 1),         # /32
            _ConvBNReLU(64, 96, 1)
        )

    def forward(self, x):
        return self.detail(x), self.semantic(x)


class _FeatureFusion(nn.Module):
    def __init__(self, d_ch=48, s_ch=96, out_ch=128):
        super().__init__()
        self.low_conv = _ConvBNReLU(d_ch, out_ch, 1)
        self.high_conv = _ConvBNReLU(s_ch, out_ch, 1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.Sigmoid()
        )
        self.out_conv = _ConvBNReLU(out_ch, out_ch, 3)

    def forward(self, low, high):
        low = self.low_conv(low)
        high = F.interpolate(self.high_conv(high), size=low.shape[-2:], mode='bilinear', align_corners=False)
        fused = torch.cat([low, high], dim=1)
        att = self.fc(fused)
        out = low * att + high * (1 - att)
        return self.out_conv(out)


class _Classifier(nn.Module):
    def __init__(self, in_ch=128, num_classes=2):
        super().__init__()
        self.cls = nn.Sequential(
            _DSConv(in_ch, in_ch),
            _DSConv(in_ch, in_ch),
            nn.ConvTranspose2d(in_ch, in_ch, 2, 2),
            nn.Conv2d(in_ch, num_classes, 1)
        )

    def forward(self, x):
        return self.cls(x)


class FastSCNN(nn.Module):
    """Fast-SCNN 简化实现（无预训练权重）"""
    def __init__(self, in_ch=3, out_ch=2):
        super().__init__()
        self.lds = _LearningDownsample()
        self.ffm = _FeatureFusion()
        self.cls = _Classifier(num_classes=out_ch)

    def forward(self, x):
        low, high = self.lds(x)
        fused = self.ffm(low, high)
        out = self.cls(fused)
        return F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)


if __name__ == "__main__":

    model = FastSCNN(in_ch=3, out_ch=1)

    input_tensor = torch.randn(1, 3, 256, 256)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    print("Output shape:", output.shape)
    flops, params = profile(model, inputs=(input_tensor, ), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")

    print(f"FLOPs: {flops}")
    print(f"Params: {params}")
