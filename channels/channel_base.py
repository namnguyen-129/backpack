# import torch
# import torch.nn as nn
# import math 

# class Channel(nn.Module):
#     def __init__(self, channel_type='AWGN', snr = 20):
#         if channel_type not in ['AWGN', 'Rayleigh', 'Rician', 'Nakagami']:
#             raise Exception('Unknown type of channel') 
#         super(Channel, self).__init__()
#         self.channel_type = channel_type
#         self.snr = snr

#     def forward(self, z_hat):
#         # print("sdf of channel ",self.snr)
#         #print("dsfw of channel",self.channel_type)
#         if z_hat.dim() not in {3, 4}:
#             raise ValueError('Input tensor must be 3D or 4D')

#         if z_hat.dim() == 3:
#             z_hat = z_hat.unsqueeze(0)
#         B, C, H, W = z_hat.shape
#         k = z_hat[0].numel()
#         sig_pwr = torch.sum(torch.abs(z_hat).square(), dim=(1, 2, 3), keepdim=True) / k
#         if self.snr is None:
#             noi_pwr = sig_pwr / (10 ** (self.snr / 10))
#         else:
#             noi_pwr = sig_pwr / (10 ** (self.snr / 10))
#         noise = torch.randn_like(z_hat) * torch.sqrt(noi_pwr / 2 )

#         if self.channel_type == 'Rayleigh':
#             hr = torch.sqrt(
#                 torch.randn(B, device=z_hat.device).pow(2) +
#                 torch.randn(B, device=z_hat.device).pow(2)
#             ) / math.sqrt(2)   
#             hi = torch.sqrt(
#                 torch.randn(B, device=z_hat.device).pow(2) +
#                 torch.randn(B, device=z_hat.device).pow(2)
#             ) / math.sqrt(2)

#             # Broadcast thành (B,1,1,1)
#             hr = hr.view(B, 1, 1, 1)
#             hi = hi.view(B, 1, 1, 1)
#             z_hat = z_hat.clone()
#             z_hat[:, :z_hat.size(1) // 2] = hr * z_hat[:, :z_hat.size(1) // 2]
#             z_hat[:, z_hat.size(1) // 2:] = hi * z_hat[:, z_hat.size(1) // 2:]
#         elif self.channel_type == "Rician":
#             # For Rician fading, add a deterministic LOS component plus a scattered (Gaussian) term.
#             K = 5.0  # Rician K-factor (power ratio of LOS to scattered components)
#             los_component = torch.sqrt(torch.tensor(K/(K+1), device=z_hat.device))
#             scatter_std = torch.sqrt(torch.tensor(1/(2*(K+1)), device=z_hat.device))
#             hr = los_component + scatter_std * torch.randn(B, device=z_hat.device)
#             hi = los_component + scatter_std * torch.randn(B, device=z_hat.device)
#             hr = hr.view(B,1,1,1); hi = hi.view(B,1,1,1)
#             # Áp fading giống hệt Rayleigh
#             half = C // 2
#             z_hat = z_hat.clone()
#             z_hat[:, :half] *= hr
#             z_hat[:, half:] *= hi
#         elif self.channel_type == "Nakagami":
#             # For Nakagami fading, generate fading amplitudes from a Gamma distribution.
#             m = 2.0    # Nakagami shape factor (m>=0.5; m=1 corresponds to Rayleigh fading)
#             omega = 1.0  # Spread parameter (often normalized to 1)
#             gamma_dist = torch.distributions.Gamma(m, m/omega)
#             # Sample two independent coefficients and take the square root to obtain Nakagami-distributed amplitudes.
#             h_n = torch.sqrt(gamma_dist.sample((2,))).to(z_hat.device)
#             z_hat = z_hat.clone()
#             half = z_hat.size(1) // 2
#             z_hat[:, :half] = h_n[0] * z_hat[:, :half]
#             z_hat[:, half:] = h_n[1] * z_hat[:, half:]
#         else:
#             pass
#         #print('Noise is', noise)
#         return z_hat + noise

#     def get_channel(self):
#         return self.channel_type, self.snr


# if __name__ == '__main__':
#     # test
#     channel = Channel(channel_type='AWGN', snr=10)
#     z_hat = torch.randn(64, 10, 5, 5)
#     z_hat = channel(z_hat)
#     print(f"AWGN: {z_hat}")

#     channel = Channel(channel_type='Rayleigh', snr=10)
#     z_hat = torch.randn(10, 5, 5)
#     z_hat = channel(z_hat)
#     print(f"Rayleigh: {z_hat}")

#     channel = Channel(channel_type='Rician', snr=10)
#     z_hat = torch.randn(10, 5, 5)
#     z_hat = channel(z_hat)
#     print(f"Rician: {z_hat}")

#     channel = Channel(channel_type='Nakagami', snr=10)
#     z_hat = torch.randn(10, 5, 5)
#     z_hat = channel(z_hat)
#     print(f"Nakagami: {z_hat}")




# import torch
# import torch.nn as nn
# import math

# class Channel(nn.Module):
#     """Channel nhận z (B,C,H,W) và danh sách chan_types + snr_vals length B.
#        Trả z_noisy cùng shape, giữ graph để backward/Backpack hoạt động."""
#     def __init__(self, default_type='AWGN', default_snr=20.0):
#         super().__init__()
#         self.default_type = default_type
#         self.default_snr = float(default_snr)

#     def _power_4d(self, z):
#         # z: (m, C, H, W) -> power per sample shaped (m,1,1,1)
#         k = z[0].numel()
#         return (z.abs().pow(2).sum(dim=(1,2,3)) / float(k)).view(-1,1,1,1)

#     def _awgn_4d(self, z_sub, snr_sub):
#         # z_sub: (m,C,H,W), snr_sub: (m,)
#         sig_pwr = self._power_4d(z_sub)                       # (m,1,1,1)
#         noi_pwr = sig_pwr / (10 ** (snr_sub.view(-1,1,1,1) / 10.0))
#         noise = torch.randn_like(z_sub) * torch.sqrt(noi_pwr)
#         return z_sub + noise

#     def _rayleigh_4d(self, z_sub, snr_sub):
#         m = z_sub.size(0)
#         hr = torch.sqrt(torch.randn(m, device=z_sub.device).pow(2) +
#                         torch.randn(m, device=z_sub.device).pow(2)) / math.sqrt(2)
#         hr = hr.view(m,1,1,1)
#         z_faded = z_sub * hr
#         return self._awgn_4d(z_faded, snr_sub)

#     def _rician_4d(self, z_sub, snr_sub, K=5.0):
#         m = z_sub.size(0)
#         los = math.sqrt(K/(K+1))
#         scatter_std = math.sqrt(1/(2*(K+1)))
#         hr = (los + scatter_std * torch.randn(m, device=z_sub.device)).view(m,1,1,1)
#         z_faded = z_sub * hr
#         return self._awgn_4d(z_faded, snr_sub)

#     def _nakagami_4d(self, z_sub, snr_sub, m_param=2.0):
#         m = z_sub.size(0)
#         omega = 1.0
#         gamma_dist = torch.distributions.Gamma(m_param, m_param/omega)
#         h_n = torch.sqrt(gamma_dist.sample((m,)).to(z_sub.device)).view(m,1,1,1)
#         z_faded = z_sub * h_n
#         return self._awgn_4d(z_faded, snr_sub)

#     def forward(self, z, chan_types=None, snr_vals=None):
#         """
#         z: (B, C, H, W)
#         chan_types: list/tuple length B of strings (e.g. 'AWGN','Rayleigh',...) or None
#         snr_vals: list/1D-tensor length B (dB) or None
#         """
#         if z.dim() != 4:
#             raise ValueError("Channel.forward expects 4-D tensor (B,C,H,W)")

#         B = z.size(0)
#         device = z.device

#         # prepare snr_vals tensor
#         if snr_vals is None:
#             snr_vals = torch.full((B,), float(self.default_snr), device=device)
#         else:
#             snr_vals = torch.tensor(snr_vals, dtype=torch.float, device=device).view(-1)

#         # prepare chan_types list
#         if chan_types is None:
#             chan_types = [self.default_type] * B
#         else:
#             if torch.is_tensor(chan_types):
#                 chan_types = [str(int(x)) for x in chan_types.view(-1).tolist()]
#             else:
#                 chan_types = [str(t) for t in chan_types]

#         # group by unique channel type to process vectorized
#         z_noisy = z.clone()
#         unique_types = list(dict.fromkeys(chan_types))  # preserve order
#         for t in unique_types:
#             idxs = [i for i, tt in enumerate(chan_types) if tt == t]
#             if len(idxs) == 0:
#                 continue
#             idxs_t = torch.tensor(idxs, dtype=torch.long, device=device)
#             z_sub = z[idxs_t]                  # (m, C, H, W)
#             snr_sub = snr_vals[idxs_t]        # (m,)

#             tl = t.lower()
#             if 'ray' in tl:
#                 out_sub = self._rayleigh_4d(z_sub, snr_sub)
#                 print('Rayyyy')
#             elif 'ric' in tl:
#                 out_sub = self._rician_4d(z_sub, snr_sub)
#             elif 'naka' in tl or 'nakagami' in tl:
#                 out_sub = self._nakagami_4d(z_sub, snr_sub)
#             else:  # default AWGN (catch 'awgn' too)
#                 print('AWGNNNN')
#                 out_sub = self._awgn_4d(z_sub, snr_sub)

#             z_noisy[idxs_t] = out_sub

#         return z_noisy



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
        snr_lin = torch.pow(10.0, snr_sub.view(-1,1,1,1) / 10.0)               # (m,1,1,1)
        noi_pwr = sig_pwr / snr_lin                                            # (m,1,1,1)

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

    def forward(self, z, chan_types=None, snr_vals=None):
    
        if z.dim() != 4:
            raise ValueError("Channel.forward expects 4-D tensor (B,C,H,W)")

        B = z.size(0)
        device = z.device

        # prepare snr_vals tensor
        if snr_vals is None:
            snr_vals = torch.full((B,), float(self.default_snr), device=device)
        else:
            snr_vals = torch.tensor(snr_vals, dtype=torch.float, device=device).view(-1)

        # prepare chan_types list
        if chan_types is None:
            chan_types = [self.default_type] * B
        else:
            if torch.is_tensor(chan_types):
                chan_types = [str(int(x)) for x in chan_types.view(-1).tolist()]
            else:
                chan_types = [str(t) for t in chan_types]

        # group by unique channel type to process vectorized
        z_noisy = z.clone()
        unique_types = list(dict.fromkeys(chan_types))  # preserve order
        for t in unique_types:
            idxs = [i for i, tt in enumerate(chan_types) if tt == t]
            if len(idxs) == 0:
                continue
            idxs_t = torch.tensor(idxs, dtype=torch.long, device=device)
            z_sub = z[idxs_t]                  # (m, C, H, W)
            snr_sub = snr_vals[idxs_t]        # (m,)

            tl = t.lower()
            if 'ray' in tl:
                out_sub = self._rayleigh_4d(z_sub, snr_sub)
            elif 'ric' in tl:
                out_sub = self._rician_4d(z_sub, snr_sub)
            elif 'naka' in tl or 'nakagami' in tl:
                out_sub = self._nakagami_4d(z_sub, snr_sub)
            else:  # default AWGN (catch 'awgn' too)
                out_sub = self._awgn_4d(z_sub, snr_sub)

            z_noisy[idxs_t] = out_sub

        return z_noisy