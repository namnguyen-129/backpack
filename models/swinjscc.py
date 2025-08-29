import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from random import choice
from channels.channel_base import Channel
from models.model_base import BaseModel
from modules.swinencoder import create_encoder
from modules.swindecoder import create_decoder
from collections import OrderedDict

from backpack import backpack, extend
from backpack.extensions import BatchGrad



class SWINJSCC(BaseModel):
    def __init__(self, args, in_channel, class_num):
        super(SWINJSCC, self).__init__(args, in_channel, class_num)
        self.args = args
        if isinstance(args.ratio, list):
            raise ValueError(f"args.ratio must be a single value, not a list: {args.ratio}")
        self.squared_difference = torch.nn.MSELoss(reduction='none')
        self.in_channel = in_channel
        self.class_num = class_num
        self.downsample = args.downsample
        encoder_kwargs = args.encoder_kwargs
        decoder_kwargs = args.decoder_kwargs
        
        self.encoder = create_encoder(**encoder_kwargs)
        self.decoder = create_decoder(**decoder_kwargs)
        self.channel = Channel('AWGN', 10)
        for  module in self.decoder.modules():
            if isinstance(module, nn.Linear) and len(list(module.parameters())) > 0:
                extend(module)

        self._register_hooks()
      
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.device else "cpu")
        self.pass_channel = args.pass_channel
        self.H = self.W = 0
        self.name = "SwinJSCC"
        self.channel_number = int(args.ratio * (2 * 3 * 2 ** (self.downsample * 2)))

    def _register_hooks(self):
        # Đăng ký hook cho các module nn.Linear
        for module in self.decoder.modules():
            if isinstance(module, nn.Linear):
                module.register_forward_hook(self.save_output)

    def save_output(self, module, input, output):
        module.output = output
        #print(f"Hook called for module {module}, output shape: {output.shape}")

    def feature_pass_channel(self, feature,chan_type, snr_chan,num_domain):
        noisy_feature = self.channel.forward(feature,chan_type, snr_chan,num_domain)  
        return noisy_feature


    def forward(self, input_image, chan_type,snr_chan):  # input_image: là x
        B, _, H, W = input_image.shape
        #print("Channel swin is", self.channel.get_channel())
        self.change_channel(channel_type=chan_type, snr=snr_chan)
        if H != self.H or W != self.W:
            self.encoder.update_resolution(H, W)
            self.decoder.update_resolution(H // (2 ** self.downsample), W // (2 ** self.downsample))
            self.H = H
            self.W = W
        feature, mask = self.encoder(input_image, snr_chan, self.channel_number)

        CBR = self.channel_number / (2 * 3 * 2 ** (self.downsample * 2))
        avg_pwr = torch.sum(feature ** 2) / mask.sum()

        if self.pass_channel:
            B, L, C = feature.shape
            H_patch = input_image.shape[2] // (2**self.downsample)
            W_patch = input_image.shape[3] // (2**self.downsample)
            assert H_patch * W_patch == L, (
            f"Mismatch tokens: L={L} nhưng H_patch×W_patch="
            f"{H_patch}×{W_patch}={H_patch*W_patch}"
            )
            feature_4D = feature.reshape(B, H_patch, W_patch, C).permute(0, 3, 1, 2)  # Chuyển đổi về (B, C, H, W)

            # Qua kênh
            noisy_feature_4D = self.feature_pass_channel(feature_4D)

            # Chuyển đổi noisy_feature về 3D để truyền vào decoder
            noisy_feature = noisy_feature_4D.flatten(2).permute(0, 2, 1)  # Chuyển đổi về (B, L, C)
        else:
            noisy_feature = feature

        noisy_feature = noisy_feature * mask
        # Decode
        recon_image = self.decoder(noisy_feature, snr_chan)

        return recon_image, CBR, snr_chan

    def get_latent(self, x,snr):
        enc,_ = self.encoder(x,snr,self.channel_number)
        enc = self.normalize_layer(enc)
        return enc

    def get_train_recon(self, x, base_snr):
        z = self.encoder(x)
        z = self.normalize_layer(z)
        z = self.channel(z)

        x_hat = self.decoder(z)
        return x_hat

    def get_latent_size(self, x):
        enc = self.encoder(x)
        enc = self.normalize_layer(enc)
        return enc.size()

    def normalize_layer(self, z):
        k = torch.tensor(1.0).to(self.device)  # torch.prod(torch.tensor(z.size()[1:], dtype=torch.float32))
        # Square root of k and P
        sqrt1 = torch.sqrt(k * self.P)
        sqrt2 = torch.sqrt(z*z + self.e)
        div = z / sqrt2
        z_out = div * sqrt1

        return z_out
    def change_channel(self, channel_type='AWGN', snr=None):
        if snr is None:
            self.channel = None
        else:
            self.channel = Channel(channel_type, snr)

    def get_channel(self):
        if hasattr(self, 'channel') and self.channel is not None:
            return self.channel.get_channel()
        return None
    
    def parse_domain(self, domain_str):
        """Extract channel name and SNR from domain string."""
        channel_name = ''.join([c for c in domain_str if not c.isdigit()])
        snr = ''.join([c for c in domain_str if c.isdigit()])
        return channel_name, int(snr)

    def parse_domain_list(self, domain_list):
        chan_list = []
        snr_list = []
        for _,d in enumerate(domain_list):
            c, s = self.parse_domain(d)
            chan_list.append(c)
            snr_list.append(s)
        return chan_list, snr_list
    
    def channel_perturb(self, input_image, domain_list, num_domain):
        B, _, H, W = input_image.shape
        batch_size = 128
        if H != self.H or W != self.W:
            self.encoder.update_resolution(H, W)
            self.decoder.update_resolution(H // (2 ** self.downsample), W // (2 ** self.downsample))
            self.H = H
            self.W = W
        chan_type_list, snr_chan_list = self.parse_domain_list(domain_list)
        all_after_encode, mask = self.encoder(input_image, snr_chan_list, self.channel_number)

        B, L, C = all_after_encode.shape
        H_patch = input_image.shape[2] // (2**self.downsample)
        W_patch = input_image.shape[3] // (2**self.downsample)
        assert H_patch * W_patch == L, (
        f"Mismatch tokens: L={L} nhưng H_patch×W_patch="
        f"{H_patch}×{W_patch}={H_patch*W_patch}"
        )
            #H = W = int(L**0.5)  # Giả định L là số lượng patch (H * W)
        feature_4D = all_after_encode.reshape(B, H_patch, W_patch, C).permute(0, 3, 1, 2)  # Chuyển đổi về (B, C, H, W)

            # Qua kênh
        #self.change_channel(channel_type=chan_type, snr=snr_chan)
            #print("Name of channel: ", self.channel.get_channel())
        noisy_feature_4D = self.feature_pass_channel(feature_4D,chan_type_list, snr_chan_list,num_domain)        
        noisy_feature = noisy_feature_4D.flatten(2).permute(0, 2, 1)


        noisy_feature = noisy_feature * mask

        all_out = self.decoder(all_after_encode, snr_chan_list)
        return all_out

