from backpack import extend, backpack
from backpack.extensions import BatchGrad
import torch
import torch.nn as nn
from models.swinjscc import SWINJSCC

def save_output_hook(module, input, output):
    module.saved_output = output

class Args:
    base_snr = 20
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inv_cdim = 32
    var_cdim = 32
    bs = 1
    ds = "cifar10"
    snr_list = [10]
    ratio = 1/6
    algo = "swinjscc"
    channel_number = 32
    channel_type = "AWGN"
    image_dims = (3, 32, 32)
    downsample = 2
    encoder_kwargs = dict(
        img_size=(32, 32), patch_size=2, in_chans=3,
        embed_dims=[64, 128], depths=[2, 4], num_heads=[4, 8],
        C=32, window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, patch_norm=True
    )
    decoder_kwargs = dict(
        img_size=(32, 32),
        embed_dims=[128, 64], depths=[4, 2], num_heads=[8, 4],
        C=32, window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, patch_norm=True
    )
    pass_channel = True

args = Args()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SWINJSCC(args, 3, 10).to(device)
    
    model.change_channel(channel_type='AWGN', snr=13)
    loss_fn = extend(nn.MSELoss()).to(device )

    B, C, H, W = 4, 3, 32, 32
    x = torch.randn(B, C, H, W, device=device)
    y = x.clone()
#     model = extend(nn.Sequential(
#     nn.Linear(10, 5),
#     nn.LayerNorm(5),  # Thêm nn.LayerNorm sau tầng Linear đầu tiên
#     nn.ReLU(),
#     nn.Linear(5, 1),
# ))
    model = model.to(device)
    output, _, _ = model(x, 10)
    # x = torch.randn(B,10,device=device)
    # y =torch.randn(4,1 ).to(device)
    # output = model(x)
    loss = loss_fn(output, y)

    try:
        with backpack(BatchGrad()):
            loss.backward(create_graph=True)
    except Exception as e:
        print("❌ Lỗi khi chạy backward với BatchGrad():", e)
        return

    # Kiểm tra trạng thái extend dựa trên grad_batch
    print("\n🔍 Kiểm tra trạng thái extend (dựa trên grad_batch):")
    for name, param in model.decoder.named_parameters():
        status = "Extended" if hasattr(param, 'grad_batch') else "Not Extended"
        print(f"  {name:50s} → {status}")
        if hasattr(param, 'grad_batch'):
            print(f"    grad_batch shape: {tuple(param.grad_batch.shape)}")
    print("sdf", output)
    # Reset gradient để tránh rò rỉ bộ nhớ
    for param in model.parameters():
        param.grad = None

    print("\n✔ Kiểm tra hoàn thành.")

if __name__ == "__main__":
    main()