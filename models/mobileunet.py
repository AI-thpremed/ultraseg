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

import torch.nn as nn
import torch.nn.functional as F
from thop import profile, clever_format
import torch

class MobileBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU6(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class MobileUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=4):
        super().__init__()

        self.enc = nn.ModuleList([
            MobileBlock(in_ch, 16, 2),
            MobileBlock(16, 32, 2),
            MobileBlock(32, 64, 2),
            MobileBlock(64, 128, 2)
        ])

        self.up = nn.ModuleList([
            nn.ConvTranspose2d(128, 64, 2, 2),
            nn.ConvTranspose2d(64, 32, 2, 2),
            nn.ConvTranspose2d(32, 16, 2, 2),
            nn.ConvTranspose2d(16, out_ch, 2, 2)
        ])

    def forward(self, x):
        feats = []
        for m in self.enc:
            x = m(x)
            feats.append(x)

        for i, m in enumerate(self.up):
            x = m(x)
            if i < 3:
                x = x + F.interpolate(feats[-i-2], size=x.shape[-2:], mode='nearest')
        return x



if __name__ == "__main__":

    model = MobileUNet(in_ch=3, out_ch=1)

    input_tensor = torch.randn(1, 3, 256, 256)

    model.eval()
    with torch.no_grad():

        output = model(input_tensor)

    print("Output shape:", output.shape)
    flops, params = profile(model, inputs=(input_tensor, ), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")

    print(f"FLOPs: {flops}")
    print(f"Params: {params}")
