# from backpack import extend
# from backpack.extensions import BatchGrad
# import torch.nn as nn
# import torch

# model = extend(nn.Sequential(
#     nn.Linear(10, 5),
#     nn.LayerNorm(5),
#     nn.ReLU(),
#     nn.Linear(5, 1),
# ))

# loss_fn = extend(nn.MSELoss())

# x = torch.randn(4, 10)
# y = torch.randn(4, 1)

# output = model(x)
# loss = loss_fn(output, y)

# from backpack import backpack

# with backpack(BatchGrad()):
#     loss.backward()

# # Access batch gradients
# for module in model.modules():
#     if hasattr(module, 'grad_batch'):
#         print(module, module.grad_batch.shape)
# def save_output_hook(module, input, output):
#     module.saved_output = output

# layer = model[0]  # for example, the first Linear layer
# layer.register_forward_hook(save_output_hook)

# # Then run forward pass again
# output = model(x)

# # Now you can access it
# print(layer.saved_output)


import torch
torch.manual_seed(0)

# tạo dữ liệu: B_total = 4, C=3, H=2, W=2, toàn 1
z = torch.ones(4, 3, 2, 2)
print("INPUT z:")
print(z)
print("shape:", z.shape)
print()

# Giả sử có 2 domain, mỗi domain có batch size = 2
domain_list = ['D1', 'D2']
len_minibatches = [2, 2]   # domain1: 2 mẫu, domain2: 2 mẫu
# muốn domain1 nhân 2, domain2 nhân 3
scales = [2.0, 3.0]

# === Cách A: xử lý bằng slicing block-wise (khi bạn concat các minibatch theo thứ tự domain) ===
def apply_scales_blockwise(z, len_minibatches, scales):
    z_out = z.clone()
    start = 0
    for bsize, scale in zip(len_minibatches, scales):
        end = start + bsize
        # z[start:end] vẫn là một mini-batch shape=(bsize, C, H, W)
        z_out[start:end] = z[start:end] * scale
        start = end
    return z_out

z_scaled_block = apply_scales_blockwise(z, len_minibatches, scales)
print("SCALED (block-wise):")
print(z_scaled_block)
print("shape:", z_scaled_block.shape)
print()

# === Cách B: xử lý bằng boolean mask (khi samples theo domain không ở vị trí liên tiếp) ===
# Tạo một vector tag per-sample theo thứ tự concat (ví dụ rải rác)
# Ở đây minh họa rải rác: sample order [D1, D2, D1, D2]
tags = ['D1', 'D2', 'D1', 'D2']
# Tạo mapping từ domain->scale
scale_map = {'D1': 2.0, 'D2': 3.0}

z_out_mask = z.clone()
for dom, scale in scale_map.items():
    # tìm indices của dom
    idxs = [i for i, t in enumerate(tags) if t == dom]
    if len(idxs) == 0:
        continue
    idxs_t = torch.tensor(idxs, dtype=torch.long)
    z_sub = z[idxs_t]            # (m, C, H, W)
    z_out_mask[idxs_t] = z_sub * scale

print("SCALED (mask / scattered order):")
print(z_out_mask)
print("shape:", z_out_mask.shape)
print()

# === minh hoạ chỉ lấy slice z[idxs] ===
print("Ví dụ: lấy slice first domain (block-wise): z[0:2]")
print(z[0:2])   # đây là 2 mẫu đầu, shape (2,3,2,2)
