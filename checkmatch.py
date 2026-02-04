# check_match.py
import torch
import sys
from pathlib import Path
from new_models.tiny_lbATT_t6_large_3 import UNet_T_LB_DS_Test6_large_3

# --------------------------------------------------
# 1. 加载权重文件
# --------------------------------------------------
ckpt_path = Path('/home/ubuntu/Desktop/gaowh/cmd-seg/results-pass2/PolypGen21_UNet_T_LB_DS_Test6_large_2_40_train_20251124_163953/best.pth')
ckpt = torch.load(ckpt_path, map_location='cpu')

# 有的 checkpoint 会把模型 state_dict 再包一层，比如 {"model": state_dict, "epoch": ...}
if 'model' in ckpt:
    state_file = ckpt['model']
else:
    state_file = ckpt

# --------------------------------------------------
# 2. 导入并实例化你的模型
# --------------------------------------------------
# 假设 model.py 里定义的是 class UNet(nn.Module)
# 如果类名叫别的，把下面 UNet 换成你自己的
sys.path.insert(0, str(Path(__file__).parent))   # 保证能找到 model.py

model = UNet_T_LB_DS_Test6_large_3(in_ch=3, out_ch=2,key=3)          # <--- 如果构造函数需要参数，在这里传
state_model = model.state_dict()

# --------------------------------------------------
# 3. 对比
# --------------------------------------------------
keys_file = set(state_file.keys())
keys_model = set(state_model.keys())

missing_in_file = keys_model - keys_file
missing_in_model = keys_file - keys_model
shape_mismatch = []
for k in keys_model & keys_file:
    if state_model[k].shape != state_file[k].shape:
        shape_mismatch.append(
            f"{k}: model {state_model[k].shape} ≠ file {state_file[k].shape}"
        )

# --------------------------------------------------
# 4. 打印结果
# --------------------------------------------------
def print_list(title, lst):
    if lst:
        print(f"\n{title}:")
        for x in lst:
            print("  ", x)
    else:
        print(f"\n{title}: (空)")

print_list("只出现在模型里、文件没有的键", missing_in_file)
print_list("只出现在文件里、模型没有的键", missing_in_model)
print_list("形状不一致的键", shape_mismatch)

if not missing_in_file and not missing_in_model and not shape_mismatch:
    print("\n✅ 完全一致，checkpoint 与模型是配对的！")
else:
    print("\n❌ 不匹配，请检查模型结构或权重来源。")