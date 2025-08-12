# from tqdm import tqdm
# import numpy as np
# import os
# import glob
# import time

# import torch
# from torch import nn
# from torch.optim import Adam

# from train.train_base import BaseTrainer
# from models.swinjscc import *
# from modules.distortion import Distortion

# import torch
# from torch.optim import Adam
# from tqdm import tqdm

# from models.swinjscc import SWINJSCC
# from collections import OrderedDict

# from backpack import backpack, extend
# from backpack.extensions import BatchGrad



# def extend_all(module):
#     extend(module)
#     for child in module.children():
#         extend_all(child)
# class SWINJSCCTrainer(BaseTrainer):
    
#     def __init__(self, args):
#         super().__init__(args)
#         # Khởi tạo model SwinJSCC với args
#         self.model = SWINJSCC(args, self.in_channel, self.class_num).to(self.device)
#         #self.model.decoder=extend(self.model.decoder)

#         #self.model._register_hooks()

#         self.optimizer = Adam(self.model.parameters(), lr=args.lr)
#         self.criterion = nn.MSELoss(reduction='mean')  
#         self.num_domains = len(args.domain_list)
#         self.penalty_weight = 100    
#         self.ema_decay      = 0.9         
#         self.penalty_anneal_iters  = 5
        
#         self.ema_per_domain = [
#             MovingAverage(ema=self.ema_decay, oneminusema_correction=True)
#             for _ in range(self.num_domains)
#         ]
#         for name, m in self.model.decoder.named_modules():
#             if isinstance(m, nn.Linear):
#                 print(f"  {name}: {m}")    
#         self.bce_extended = extend(nn.MSELoss(reduction='none'))
#         self.update_count = 0
#         self.domain_list = args.domain_list
#         print(self.domain_list)
#     def parse_domain(self, domain_str):
#         """Extract channel name and SNR from domain string."""
#         channel_name = ''.join([c for c in domain_str if not c.isdigit()])
#         snr = ''.join([c for c in domain_str if c.isdigit()])
#         return channel_name, int(snr)
    
#     def train(self):
#         domain_list = self.domain_list 
        
#         for epoch in range(self.args.out_e):
#             self.model.train()

#             epoch_train_loss = 0
#             epoch_val_loss = 0
#             for batch_idx, (x, y) in enumerate(tqdm(self.train_dl, desc=f"Epoch {epoch}")): 
#                 x, y = x.to(self.device), y.to(self.device)
#                 total_loss = 0
#                 all_in = []
#                 all_out = []
#                 len_minibatches = []
#                 for i, domain_str in enumerate(domain_list):
#                     chan_type, snr_chan = self.parse_domain(domain_str)
#                     #self.model.change_channel(channel_type=chan_type, snr=snr_chan)
#                     out = self.model.channel_perturb(x, chan_type, snr_chan)
#                     z = self.model.get_latent(x,snr_chan)
#                     z = self.model.channel(z)
#                     #out, _, _ = self.model(x, snr_chan)
#                     print("Out shape",out.shape)
#                     # FIXME the tensors should be flattened later
#                     all_in.append(x)  
#                     all_out.append(out) # output model (->encoder->channel->decoder->out)
#                     len_minibatches.append(x.shape[0])
                
#                 # for name, module in self.model.decoder.named_modules():
#                 #     if isinstance(module, nn.Linear):
#                 #         if hasattr(module, 'output'):
#                 #             print(f"{name} có thuộc tính 'output': {module.output.shape}")
#                 #         else:
#                 #             print(f"{name} không có thuộc tính 'output'")
#                 all_in = torch.cat(all_in, dim=0)
#                 all_out = torch.cat(all_out, dim=0)
#                 print("Shapeee",all_in.shape)
#                 print("Sh", all_out.shape)
#                 print("len_minibatch",len_minibatches)
#                 print("num_domain,", self.num_domains)
#                 penalty = self.compute_fishr_penalty(all_out, all_in, len_minibatches)
#                 loss = self.criterion(all_out, all_in) # la so thuc nen phai dung mse khong dung cross entropy 
#                 penalty_weight = 0.1 
             
#                 self.update_count += 1

#                 objective = loss + penalty_weight * penalty
#                 self.optimizer.zero_grad()
#                 objective.backward()
#                 self.optimizer.step()

#                 #return {'loss': objective.item(), 'nll': loss.item(), 'penalty': penalty.item()}
#                 # Backward
#                 total_loss += objective.item()
#             epoch_train_loss = total_loss / len(self.train_dl)
#             self.writer.add_scalar('train/loss', epoch_train_loss, epoch)
#             print(f"[Train] Epoch {epoch}: loss = {epoch_train_loss:.4f}")
           
#             self.save_model(epoch=epoch, model=self.model)

#         self.writer.close()
#         self.save_config()

#     def l2_between_dicts(self,dict_1, dict_2):
#         assert len(dict_1) == len(dict_2)
#         dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
#         dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
#         return (
#             torch.cat(tuple([t.view(-1) for t in dict_1_values])) -
#             torch.cat(tuple([t.view(-1) for t in dict_2_values]))
#         ).pow(2).mean()

#     def compute_fishr_penalty(self, all_out, all_in,  len_minibatches):
#         print("all_y shape:", all_in.shape)
#         dict_grads = self._get_grads(all_out, all_in)
#         grads_var_per_domain = self._get_grads_var_per_domain(dict_grads, len_minibatches)
#         return self._compute_distance_grads_var(grads_var_per_domain)

#     def _get_grads(self, logits,y):
#         self.optimizer.zero_grad()
#         loss = self.bce_extended(logits, y).sum()
#         print('asdfwoefjaojw')
#         with backpack(BatchGrad()):
#             loss.backward(
#                 inputs=list(self.model.decoder.parameters()), retain_graph=True, create_graph=True
#             )

#         # compute individual grads for all samples across all domains simultaneously
#         dict_grads = OrderedDict()
#         for name, weights in self.model.decoder.named_parameters():
#             print("nafdsfme: ",name)
#             print("weidsfasdght:", weights)   
#         return dict_grads

#     def _compute_distance_grads_var(self, grads_var_per_domain):

#         # compute gradient variances averaged across domains
#         grads_var = OrderedDict(
#             [
#                 (
#                     name,
#                     torch.stack(
#                         [
#                             grads_var_per_domain[domain_id][name]
#                             for domain_id in range(self.num_domains)
#                         ],
#                         dim=0
#                     ).mean(dim=0)
#                 )
#                 for name in grads_var_per_domain[0].keys()
#             ]
#         )

#         penalty = 0
#         for domain_id in range(self.num_domains):
#             penalty += self.l2_between_dicts(grads_var_per_domain[domain_id], grads_var)
#         return penalty / self.num_domains
    
#     def _get_grads_var_per_domain(self, dict_grads, len_minibatches):
#         # grads var per domain
#         grads_var_per_domain = [{} for _ in range(self.num_domains)]
#         for name, _grads in dict_grads.items():
#             all_idx = 0
#             for domain_id, bsize in enumerate(len_minibatches):
#                 env_grads = _grads[all_idx:all_idx + bsize]
#                 all_idx += bsize
#                 env_mean = env_grads.mean(dim=0, keepdim=True)
#                 env_grads_centered = env_grads - env_mean
#                 grads_var_per_domain[domain_id][name] = (env_grads_centered).pow(2).mean(dim=0)

#         # moving average
#         for domain_id in range(self.num_domains):
#             grads_var_per_domain[domain_id] = self.ema_per_domain[domain_id].update(
#                 grads_var_per_domain[domain_id]
#             )

#         return grads_var_per_domain
# class MovingAverage:

#     def __init__(self, ema, oneminusema_correction=True):
#         self.ema = ema
#         self.ema_data = {}
#         self._updates = 0
#         self._oneminusema_correction = oneminusema_correction

#     def update(self, dict_data):
#         ema_dict_data = {}
#         for name, data in dict_data.items():
#             data = data.view(1, -1)
#             if self._updates == 0:
#                 previous_data = torch.zeros_like(data)
#             else:
#                 previous_data = self.ema_data[name]

#             ema_data = self.ema * previous_data + (1 - self.ema) * data
#             if self._oneminusema_correction:
#                 # correction by 1/(1 - self.ema)
#                 # so that the gradients amplitude backpropagated in data is independent of self.ema
#                 ema_dict_data[name] = ema_data / (1 - self.ema)
#             else:
#                 ema_dict_data[name] = ema_data
#             self.ema_data[name] = ema_data.clone().detach()

#         self._updates += 1
#         return ema_dict_data
    









from tqdm import tqdm
import numpy as np
import os
import glob
import time

import torch
from torch import nn
from torch.optim import Adam

from train.train_base import BaseTrainer
from models.swinjscc import *
from modules.distortion import Distortion

import torch
from torch.optim import Adam
from tqdm import tqdm

from models.swinjscc import SWINJSCC
from collections import OrderedDict

from backpack import backpack, extend
from backpack.extensions import BatchGrad



def extend_all(module):
    extend(module)
    for child in module.children():
        extend_all(child)
class SWINJSCCTrainer(BaseTrainer):
    
    def __init__(self, args):
        super().__init__(args)
        # Khởi tạo model SwinJSCC với args
        self.model = SWINJSCC(args, self.in_channel, self.class_num).to(self.device)
        #self.model.decoder=extend(self.model.decoder)

        #self.model._register_hooks()

        self.optimizer = Adam(self.model.parameters(), lr=args.lr)
        self.criterion = nn.MSELoss(reduction='mean')  
        self.num_domains = len(args.domain_list)
        self.penalty_weight = 100    
        self.ema_decay      = 0.9         
        self.penalty_anneal_iters  = 5
        
        self.ema_per_domain = [
            MovingAverage(ema=self.ema_decay, oneminusema_correction=True)
            for _ in range(self.num_domains)
        ]
        for name, m in self.model.decoder.named_modules():
            if isinstance(m, nn.Linear):
                print(f"  {name}: {m}")    
        self.bce_extended = extend(nn.MSELoss(reduction='none'))
        self.update_count = 0
        self.domain_list = args.domain_list
        print(self.domain_list)
    def parse_domain(self, domain_str):
        """Extract channel name and SNR from domain string."""
        channel_name = ''.join([c for c in domain_str if not c.isdigit()])
        snr = ''.join([c for c in domain_str if c.isdigit()])
        return channel_name, int(snr)
    
    def train(self):
        domain_list = self.domain_list 
        
        for epoch in range(self.args.out_e):
            self.model.train()

            epoch_train_loss = 0
            epoch_val_loss = 0
            for batch_idx, (x, y) in enumerate(tqdm(self.train_dl, desc=f"Epoch {epoch}")): 
                x, y = x.to(self.device), y.to(self.device)
                total_loss = 0
                all_in = []
                all_out = []
                len_minibatches = []
                for i, domain_str in enumerate(domain_list):
                    chan_type, snr_chan = self.parse_domain(domain_str)

                    #self.model.change_channel(channel_type=chan_type, snr=snr_chan)
                    out,_,_ = self.model(x, chan_type, snr_chan)

                    #out, _, _ = self.model(x, snr_chan)
                    print("Out shape",out.shape)
                    # FIXME the tensors should be flattened later
                    all_in.append(x)  
                    all_out.append(out) # output model (->encoder->channel->decoder->out)
                    len_minibatches.append(x.shape[0])
                
                # for name, module in self.model.decoder.named_modules():
                #     if isinstance(module, nn.Linear):
                #         if hasattr(module, 'output'):
                #             print(f"{name} có thuộc tính 'output': {module.output.shape}")
                #         else:
                #             print(f"{name} không có thuộc tính 'output'")
                all_in = torch.cat(all_in, dim=0)
                all_out = torch.cat(all_out, dim=0)
                print("Shapeee",all_in.shape)
                print("Sh", all_out.shape)
                print("len_minibatch",len_minibatches)
                print("num_domain,", self.num_domains)
                penalty = self.compute_fishr_penalty(all_out, all_in, len_minibatches)
                loss = self.criterion(all_out, all_in) # la so thuc nen phai dung mse khong dung cross entropy 
                penalty_weight = 0.1 
             
                self.update_count += 1

                objective = loss + penalty_weight * penalty
                self.optimizer.zero_grad()
                objective.backward()
                self.optimizer.step()

                #return {'loss': objective.item(), 'nll': loss.item(), 'penalty': penalty.item()}
                # Backward
                total_loss += objective.item()
            epoch_train_loss = total_loss / len(self.train_dl)
            self.writer.add_scalar('train/loss', epoch_train_loss, epoch)
            print(f"[Train] Epoch {epoch}: loss = {epoch_train_loss:.4f}")

            self.save_model(epoch=epoch, model=self.model)

        self.writer.close()
        self.save_config()

    def l2_between_dicts(self,dict_1, dict_2):
        assert len(dict_1) == len(dict_2)
        dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
        dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
        return (
            torch.cat(tuple([t.view(-1) for t in dict_1_values])) -
            torch.cat(tuple([t.view(-1) for t in dict_2_values]))
        ).pow(2).mean()

    def compute_fishr_penalty(self, all_out, all_in,  len_minibatches):
        print("all_y shape:", all_in.shape)
        dict_grads = self._get_grads(all_out, all_in)
        grads_var_per_domain = self._get_grads_var_per_domain(dict_grads, len_minibatches)
        return self._compute_distance_grads_var(grads_var_per_domain)

    # def _get_grads(self, out, inp):
    #     self.optimizer.zero_grad()
    #     loss = self.bce_extended(inp, out).sum()
    #     with backpack(BatchGrad()):
    #         loss.backward(
    #             inputs=list(self.model.decoder.parameters()), retain_graph=True
    #         )


    #     dict_grads = OrderedDict()
    #     for name, weights in self.model.decoder.named_parameters():
    #         if hasattr(weights, "grad_batch") and weights.grad_batch is not None:
    #             dict_grads[name] = weights.grad_batch.clone().view(weights.grad_batch.size(0), -1)
    #         else:
    #             print(f"[WARNING] No grad_batch for {name} — possibly not supported by Backpack or not used in forward.")

    #     return dict_grads

    def _get_grads(self, out, inp):
    # giữ nguyên cách bạn tính loss
        self.optimizer.zero_grad()
        loss = self.bce_extended(inp, out).sum()

    # Thông tin ban đầu
        print("[BACKPACK] Starting BatchGrad backward")
        try:
            print(f"[BACKPACK] loss.requires_grad={loss.requires_grad}, loss.shape?={'scalar' if loss.dim()==0 else loss.shape}")
        except:
            pass

    # chạy Backpack và bắt lỗi để log
        try:
            with backpack(BatchGrad()):
                loss.backward()
        except Exception as e:
            print(f"[BACKPACK][ERROR] exception during backward: {e}")
        # nếu muốn tiếp tục chạy (không raise) thì comment dòng dưới, nhưng tốt nhất raise để debug
            raise

    # thu thập grad_batch và in log chi tiết cho mỗi param
        dict_grads = OrderedDict()
        any_gb = False
        for name, weights in self.model.decoder.named_parameters():
            has_gb = hasattr(weights, "grad_batch") and weights.grad_batch is not None
            print(f"[BACKPACK] param='{name}' has_grad_batch={has_gb}")
            if has_gb:
                gb = weights.grad_batch  # shape: (B_total, ...)
                any_gb = True
                try:
                    print(f"   grad_batch.shape={tuple(gb.shape)}, dtype={gb.dtype}, device={gb.device}")
                    flat = gb.detach().clone().view(gb.size(0), -1)  # (B, param_numel)
                # tóm tắt nhanh
                    mean = float(flat.mean()) if flat.numel() > 0 else float("nan")
                    std = float(flat.std()) if flat.numel() > 0 else float("nan")
                    print(f"   flat.shape={tuple(flat.shape)}, mean={mean:.6e}, std={std:.6e}")
                    dict_grads[name] = flat
                except Exception as ex:
                    print(f"[BACKPACK][WARN] failed to flatten grad_batch for {name}: {ex}")

        if not any_gb:
            print("[BACKPACK][WARN] NO grad_batch produced for ANY decoder parameter.")
            print("[BACKPACK][WARN] Possible reasons: decoder not used in forward, extend(...) missing, module types not supported by BackPACK, or graph mismatch between forward/backward.")
            print("[BACKPACK] Decoder modules and types:")
            for nm, m in self.model.decoder.named_modules():
                print(f"   - {nm}: {type(m).__name__}")
            print("[BACKPACK] Decoder parameters and shapes:")
            for nm, p in self.model.decoder.named_parameters():
                print(f"   * {nm}: shape={tuple(p.shape)}, requires_grad={p.requires_grad}")

        print("[BACKPACK] Done collecting grad_batch (returning dict_grads keys:", list(dict_grads.keys()), ")")
        return dict_grads
    

    def _get_grads_var_per_domain(self, dict_grads, len_minibatches):
        # grads var per domain
        grads_var_per_domain = [{} for _ in range(self.num_domains)]
        for name, _grads in dict_grads.items():
            all_idx = 0
            for domain_id, bsize in enumerate(len_minibatches):
                env_grads = _grads[all_idx:all_idx + bsize]
                all_idx += bsize
                env_mean = env_grads.mean(dim=0, keepdim=True)
                env_grads_centered = env_grads - env_mean
                grads_var_per_domain[domain_id][name] = (env_grads_centered).pow(2).mean(dim=0)

        # moving average
        for domain_id in range(self.num_domains):
            grads_var_per_domain[domain_id] = self.ema_per_domain[domain_id].update(
                grads_var_per_domain[domain_id]
            )
        for domain_id in range(self.num_domains):
            for k in grads_var_per_domain[domain_id].keys():
                print("Available keys:", k)
        return grads_var_per_domain

    def _compute_distance_grads_var(self, grads_var_per_domain):

        # compute gradient variances averaged across domains
        
        grads_var = OrderedDict(
            [
                (
                    name,
                    torch.stack(
                        [
                            grads_var_per_domain[domain_id][name]
                            
                            for domain_id in range(self.num_domains)
                        ],
                        dim=0
                    ).mean(dim=0)
                )
                for name in grads_var_per_domain[0].keys()
            ]
        )

        penalty = 0
        for domain_id in range(self.num_domains):
            penalty += self.l2_between_dicts(grads_var_per_domain[domain_id], grads_var)
        return penalty / self.num_domains
    

class MovingAverage:

    def __init__(self, ema, oneminusema_correction=True):
        self.ema = ema
        self.ema_data = {}
        self._updates = 0
        self._oneminusema_correction = oneminusema_correction

    def update(self, dict_data):
        ema_dict_data = {}
        for name, data in dict_data.items():
            data = data.view(1, -1)
            if self._updates == 0:
                previous_data = torch.zeros_like(data)
            else:
                previous_data = self.ema_data[name]

            ema_data = self.ema * previous_data + (1 - self.ema) * data
            if self._oneminusema_correction:
                # correction by 1/(1 - self.ema)
                # so that the gradients amplitude backpropagated in data is independent of self.ema
                ema_dict_data[name] = ema_data / (1 - self.ema)
            else:
                ema_dict_data[name] = ema_data
            self.ema_data[name] = ema_data.clone().detach()

        self._updates += 1
        return ema_dict_data
    