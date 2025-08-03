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
                    out = self.model.channel_perturb(x, chan_type, snr_chan)
                    z = self.model.get_latent(x,snr_chan)
                    z = self.model.channel(z)
                    #out, _, _ = self.model(x, snr_chan)
                    print("Out shape",out.shape)
                    # FIXME the tensors should be flattened later
                    all_in.append(x)  
                    all_out.append(out) # output model (->encoder->channel->decoder->out)
                    len_minibatches.append(x.shape[0])
                
                for name, module in self.model.decoder.named_modules():
                    if isinstance(module, nn.Linear):
                        if hasattr(module, 'output'):
                            print(f"{name} có thuộc tính 'output': {module.output.shape}")
                        else:
                            print(f"{name} không có thuộc tính 'output'")
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



        # domain_list = self.domain_list             
        # D = len(domain_list)
        # snr_vals = []
        # for dom in domain_list:
        #     _, snr = self.parse_domain(dom)
        #     snr_vals.append(snr)
        # snr_vals = torch.tensor(snr_vals, device=self.device) # (D,)

        # for epoch in range(self.args.out_e):
        #     total_loss = 0
        #     for x, y in tqdm(self.train_dl, desc=f"Epoch {epoch}"):
        #         x = x.to(self.device)   # (B, C, H, W)
        #         B = x.size(0)

        #     # 1) Tạo snr tensor (B, D)
        #         snrs = snr_vals.unsqueeze(0).expand(B, D)  # (B, D)

        #     # 2) Forward qua channel_perturb cho tất cả domains
        #         out = self.model.channel_perturb(x, snrs)   # (B, D, 3, H, W)

        #     # 3) Reshape để tính loss chung
        #         out_all = out.view(B*D, 3, x.size(2), x.size(3))
        #         x_all   = x.unsqueeze(1).expand(-1, D, -1, -1, -1).reshape(B*D, 3, x.size(2), x.size(3))

        #     # 4) Tính Fishr penalty (nếu cần) per‑domain
        #     #    reshape out_all → (B, D, 3, H, W) rồi tách domains
        #         penalty = self.compute_fishr_penalty(
        #             out_all.view(B, D, 3, *x_all.shape[2:]),
        #             x_all .view(B, D, 3, *x_all.shape[2:]),
        #             len_minibatches=[B]*D
        #         )

        #     # 5) Loss + backward
        #         loss = self.criterion(out_all, x_all)
        #         objective = loss + self.penalty_weight * penalty

        #         self.optimizer.zero_grad()
        #         objective.backward()
        #         self.optimizer.step()

        #         total_loss += objective.item()

        #     avg_loss = total_loss / len(self.train_dl)
        #     print(f"[Train] Epoch {epoch}: loss = {avg_loss:.4f}")
        #     self.writer.add_scalar('train/loss', avg_loss, epoch)
           
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

    # def _get_grads(self, out, input):
    #     self.optimizer.zero_grad()

    #     ##
    #     print("\n🔍 Danh sách module nn.Linear trước forward pass:")
    #     linear_modules = []
    #     for name, module in self.model.decoder.named_modules():
    #         if isinstance(module, nn.Linear):
    #             linear_modules.append((name, module))
    #             print(f"  {name:50s} → Has output: {hasattr(module, 'output')}")
    
    # # Forward pass để đảm bảo tất cả module được gọi
    #     self.model.eval()  # Chuyển sang eval để không ảnh hưởng trọng số
    #     try:
    #         for i, domain_str in enumerate(self.domain_list):
    #             chan, snr = self.parse_domain(domain_str)
    #             _ = self.model.channel_perturb(input[i*128:(i+1)*128], chan, snr)
    #         print("\n🔍 Danh sách module nn.Linear sau forward pass:")
    #         for name, module in linear_modules:
    #             has_output = hasattr(module, 'output')
    #             print(f"  {name:50s} → Has output: {has_output}")
    #             if has_output:
    #                 print(f"    Output shape: {module.output.shape}")
    #             else:
    #                 print(f"    [WARNING] Module {name} thiếu 'output'!")
    #     except Exception as e:
    #         print(f"Error in forward pass in _get_grads: {e}")
    #         return OrderedDict()
    #     ####
    #     self.model.train()
    #     loss = self.bce_extended(out, input).sum()
    #     try:
    #         with backpack(BatchGrad()):
    #             loss.backward(inputs=list(self.model.decoder.parameters()), create_graph=True)
    #         print("\n🔍 Kiểm tra sau backward pass:")
    #         for name, module in linear_modules:
    #             print(f"  {name:50s} → Has grad_batch: {hasattr(module.weight, 'grad_batch')}")
    #     except Exception as e:
    #         print(f"❌ Lỗi khi chạy backward với BatchGrad(): {e}")
    #         print("\n🔍 Các module nn.Linear thiếu 'output':")
    #         for name, module in linear_modules:
    #             if not hasattr(module, 'output'):
    #                 print(f"  {name:50s} → Thiếu 'output'")
    #         return OrderedDict()
    
    #     dict_grads = OrderedDict()
    #     for name, param in self.model.decoder.named_parameters():
    #         if hasattr(param, "grad_batch"):
    #             dict_grads[name] = param.grad_batch.clone().detach().view(param.grad_batch.size(0), -1)
    #             print(f"[INFO] Added grad for {name}, shape: {dict_grads[name].shape}")
    #         else:
    #             print(f"[INFO] Skipping {name} (no grad_batch)")
    #     return dict_grads

    # def _get_grads(self, out, inp):
    #     # out, inp are already concatenated tensors
    #     self.optimizer.zero_grad()
    #     for m in self.model.decoder.modules():
    #         if isinstance(m, nn.Linear) and hasattr(m, 'output'):
    #             del m.output
        
    #     chunk = inp.size(0) // 2
    #     for i, dom in enumerate(self.domain_list):
    #         inp_i = inp[i*chunk:(i+1)*chunk]
    #         _ = self.model.channel_perturb(inp_i, *self.parse_domain(dom))

    #     loss = self.bce_extended(out, inp).sum()
    #     # ensure hooks populated from previous forward
    #     with backpack(BatchGrad()):
    #         loss.backward(create_graph=True)

    #     grads = OrderedDict()
    #     for name, param in self.model.decoder.named_parameters():
    #         if hasattr(param, 'grad_batch'):
    #             grads[name] = param.grad_batch.detach().view(param.grad_batch.size(0), -1)
    #     return grads

    def _get_grads(self, out, inp):
        D = len(self.domain_list)
        B_total = inp.size(0)
        chunk = B_total // D

    # 0) Clear old outputs
        for m in self.model.decoder.modules():
            if isinstance(m, nn.Linear) and hasattr(m, 'output'):
                del m.output

    # 1) Log trước khi forward lại
        print(f"[_get_grads] B_total={B_total}, domains={D}, chunk={chunk}")

    # 2) Forward từng slice, logging tên domain + shape
        for i, dom in enumerate(self.domain_list):
            start, end = i*chunk, (i+1)*chunk
            inp_i = inp[start:end]
            print(f"  Domain {i}='{dom}': inp_i.shape={tuple(inp_i.shape)}")
            _ = self.model.channel_perturb(inp_i, *self.parse_domain(dom))

    # 3) Kiểm tra modules thiếu output
        missing = [
            name for name, module in self.model.decoder.named_modules()
            if isinstance(module, nn.Linear) and not hasattr(module, 'output')]
        print(f"[_get_grads] Modules missing output ({len(missing)}): {missing[:10]}{'...' if len(missing)>10 else ''}")

    # 4) Tính loss và backward
        loss = self.bce_extended(out, inp).sum()
        with backpack(BatchGrad()):
    # Thay vì backward, dùng autograd.grad sẽ không tạo cycle trong .grad
            loss.backward(create_graph=True)
            # grads_list = torch.autograd.grad(
            #     outputs=loss,
            #     inputs=list(self.model.decoder.parameters()),
            #     create_graph=True,
            #     retain_graph=True,
            #     allow_unused=True
            # )

    # 5) Thu grad_batch như cũ
        grads = OrderedDict()
        for name, param in self.model.decoder.named_parameters():
            if hasattr(param, 'grad_batch'):
                grads[name] = param.grad_batch.detach().view(param.grad_batch.size(0), -1)
        return grads


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
    


    #     self.optimizer.zero_grad()
    #     loss = self.bce_extended(logits, y).sum()
    #     with backpack(BatchGrad()):
    #         loss.backward(inputs=list(self.model.decoder.parameters()), retain_graph=True, create_graph=True)
    
    #     dict_grads = OrderedDict()
    #     for name, weights in self.model.decoder.named_parameters():
    #         if hasattr(weights, "grad_batch"):
    #             dict_grads[name] = weights.grad_batch.clone().view(weights.grad_batch.size(0), -1)
    #         else:
    #             print(f"[INFO] Skipping {name} as it has no grad_batch.")
    
    # # Reset gradient để tránh rò rỉ bộ nhớ
    #     for param in self.model.decoder.parameters():
    #         param.grad = None
    
    #     return dict_grads