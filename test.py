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
from models.swinjscc import SWINJSCC
model = 
model = 
ckpt = torch.load("C:\SemCom\Testbackpack\SemCom-Pytorch\out\checkpoints\CIFAR10_0.16666666666666666_Rayleigh0_swinjscc_16h52m53s_on_Aug_14_2025\epoch_0.pkl", map_location="cpu")
state_dict = ckpt.get('state_dict', ckpt)   # nếu checkpoint nested
print(">>> checkpoint keys (sample 200):")
for k in list(state_dict.keys())[:200]:
    print(k)
print(">>> total ckpt keys:", len(state_dict))

print("\n>>> model keys (sample 200):")
for k in list(model.state_dict().keys())[:200]:
    print(k)
print(">>> total model keys:", len(model.state_dict()))