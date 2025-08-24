import torch
import torch.nn as nn
import math

class Channel(nn.Module):
    def __init__(self, default_type='AWGN', default_snr=20):
        super().__init__()
        self.default_type = default_type
        self.default_snr = int(default_snr)

    def _power_4d(self, z_hat,snr_sub):
        # z: (m, C, H, W) -> power per sample shaped (m,1,1,1)
        k = z_hat[0].numel()
        sig_pwr = torch.sum(torch.abs(z_hat).square(), dim=(1, 2, 3), keepdim=True) / k
        noi_pwr = sig_pwr / (10 ** (snr_sub / 10))
        noise = torch.randn_like(z_hat) * torch.sqrt(noi_pwr / 2 )                                        # (m,1,1,1)

    # Noise ~ N(0, noi_pwr)
        noise = torch.randn_like(z_hat) * torch.sqrt(noi_pwr/2)
        return noise
        
    def _awgn_4d(self, z_sub, snr_sub):
        noise = self._power_4d(z_sub,snr_sub)                       # (m,1,1,1)
        return z_sub + noise

    def _rayleigh_4d(self, z_hat, snr_sub):
        B = z_hat.size(0)
        hr = torch.sqrt(
            torch.randn(B, device=z_hat.device).pow(2) +
            torch.randn(B, device=z_hat.device).pow(2)
        ) / math.sqrt(2)   
        hi = torch.sqrt(
            torch.randn(B, device=z_hat.device).pow(2) +
            torch.randn(B, device=z_hat.device).pow(2)
        ) / math.sqrt(2)

            # Broadcast thành (B,1,1,1)
        hr = hr.view(B, 1, 1, 1)
        hi = hi.view(B, 1, 1, 1)

        z_hat = z_hat.clone()
        z_hat[:, :z_hat.size(1) // 2] = hr * z_hat[:, :z_hat.size(1) // 2]
        z_hat[:, z_hat.size(1) // 2:] = hi * z_hat[:, z_hat.size(1) // 2:]
        return self._awgn_4d(z_hat, snr_sub)

    def _rician_4d(self, z_hat, snr_sub, K=5.0):
        B = z_hat.size(0)
        C = z_hat.size(1)
        K = 5.0  # Rician K-factor (power ratio of LOS to scattered components)
        los_component = torch.sqrt(torch.tensor(K/(K+1), device=z_hat.device))
        scatter_std = torch.sqrt(torch.tensor(1/(2*(K+1)), device=z_hat.device))
        hr = los_component + scatter_std * torch.randn(B, device=z_hat.device)
        hi = los_component + scatter_std * torch.randn(B, device=z_hat.device)
        hr = hr.view(B,1,1,1); hi = hi.view(B,1,1,1)
            # Áp fading giống hệt Rayleigh
        half = C // 2
        z_hat = z_hat.clone()
        z_hat[:, :half] *= hr
        z_hat[:, half:] *= hi

        return self._awgn_4d(z_hat, snr_sub)

    def _nakagami_4d(self, z_hat, snr_sub, m_param=2.0):
        m = 2.0    # Nakagami shape factor (m>=0.5; m=1 corresponds to Rayleigh fading)
        omega = 1.0  # Spread parameter (often normalized to 1)
        gamma_dist = torch.distributions.Gamma(m, m/omega)
            # Sample two independent coefficients and take the square root to obtain Nakagami-distributed amplitudes.
        h_n = torch.sqrt(gamma_dist.sample((2,))).to(z_hat.device)
        z_hat = z_hat.clone()
        half = z_hat.size(1) // 2
        z_hat[:, :half] = h_n[0] * z_hat[:, :half]
        z_hat[:, half:] = h_n[1] * z_hat[:, half:]
        return self._awgn_4d(z_hat, snr_sub)
    


    def forward_channel(self, z, chan_type, snr_val):
        if z.dim() != 4:
            raise ValueError("Channel.forward expects 4-D tensor (B,C,H,W)")
        if(chan_type == 'AWGN'):
            return self._awgn_4d(z, snr_val)
        elif(chan_type == 'Rayleigh'):
            return self._rayleigh_4d(z, snr_val)
        elif(chan_type == 'Rician'):
            return self._rician_4d(z, snr_val)
        elif(chan_type == 'Nakagami'):
            return self._nakagami_4d(z, snr_val)
        else:
            raise ValueError(f"Unknown channel type: {chan_type}")

    def forward(self, z, chan_types=None, snr_vals=None): #chan_list, snr_list
    
        if z.dim() != 4:
            raise ValueError("Channel.forward expects 4-D tensor (B,C,H,W)")

        B = z.size(0)
        device = z.device
        z_noisy = z.clone()
        z_1 = z[0:128]
        #z_2 = z[128:256]
        #print('Debugggg', z_noisy - z)
        z1 = self.forward_channel(z_1,chan_types[0], snr_vals[0])
        #z2 = self.forward_channel(z_2,chan_types[1], snr_vals[1])
        z_noisy[0:128] = z1
        #z_noisy[128:256] = z2
        return z_noisy