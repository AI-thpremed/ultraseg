import torch
import time
import numpy as np
import psutil
import torch, time, numpy as np
from models.unet import U_Net
from models.fastscnn import FastSCNN
from models.mobileunet import MobileUNet
from models.unet_s import UNet_S
from models.unet_t import UNet_T
import torch
import time
import numpy as np
from new_models.colonnet108k import ColonNet108
from new_models.colonnet130k import ColonNet130


def benchmark_cpu(model_class, name, x, num_threads=24):

    model = model_class().eval().cpu()
    
    torch.set_num_threads(num_threads)
    
    with torch.no_grad():
        for _ in range(50):
            _ = model(x)
    
    times = []
    with torch.no_grad():
        start = time.perf_counter()  
        for _ in range(1000):
            _ = model(x)
        end = time.perf_counter()
    
    # 计算
    total_time = end - start
    fps = 1000 / total_time
    latency_ms = total_time / 1000 * 1000  #
    
    cpu_percent = psutil.cpu_percent(interval=0.1) / 100.0
    if num_threads == 1:

        power_est = 125 * cpu_percent / 24
    else:
        power_est = 125 * cpu_percent
    
    print(f"{name:15} | {fps:6.1f} FPS | {latency_ms:5.1f} ms | {power_est:4.1f} W")
    return fps, latency_ms, power_est

x = torch.randn(1, 3, 256, 256).cpu()

print("="*60)
print(f"{'Model':15} | {'FPS':6} | {'Latency':7} | {'Power':5}")
print("="*60)

print("\n[24-Core Test (Intel i9-14900K)]")
models = {
    "ColonNet130": ColonNet130 ,
    "ColonNet108": ColonNet108 
    }

results_24core = {}
for name, model_class in models.items():
    torch.set_num_threads(24)
    results_24core[name] = benchmark_cpu(model_class, name, x, num_threads=24)

print("\n[Single-Core Test (Simulated Jetson Nano)]")
results_1core = {}
for name, model_class in models.items():
    torch.set_num_threads(1)
    results_1core[name] = benchmark_cpu(model_class, name, x, num_threads=1)

print("\n" + "="*80)
print("summary:")
print("="*80)
for name in models.keys():
    fps_24, lat_24, _ = results_24core[name]
    fps_1, lat_1, _ = results_1core[name]
    slowdown = fps_24 / fps_1
    print(f"{name:15} | {fps_24:5.1f} → {fps_1:5.1f} FPS ({slowdown:4.1f}x slower)")


