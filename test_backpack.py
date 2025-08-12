import torch
import torch.nn as nn
from backpack import extend, backpack
from backpack.extensions import BatchGrad

# ==== Model tối giản ====
class SimpleAE(nn.Module):
    def __init__(self, in_dim=3*32*32, latent_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, in_dim),
            nn.Sigmoid(),
        )

        # extend các module Linear trong decoder
        for module in self.decoder.modules():
            if isinstance(module, nn.Linear):
                extend(module)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

# ==== Trainer tối giản ====
def train_simple():
    in_dim = 3*32*32
    model = SimpleAE(in_dim).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # extend loss
    bce_loss = extend(nn.MSELoss(reduction='none'))

    x = torch.rand(8, 3, 32, 32).cuda()  # batch size = 8
    target = x.clone()                   # autoencoder nên target = input

    optimizer.zero_grad()
    out = model(x)                       # forward
    loss = bce_loss(out, target.view(out.shape)).sum()

    # bật tính grad_batch
    with backpack(BatchGrad()):
        loss.backward()

    # in grad_batch
    for name, param in model.decoder.named_parameters():
        if hasattr(param, 'grad_batch'):
            print(f"{name}: grad_batch.shape = {param.grad_batch.shape}")
        else:
            print(f"{name}: no grad_batch")

if __name__ == "__main__":
    train_simple()