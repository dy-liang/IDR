import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .aux_base import AUXBase
from common import utils
from module.rl_module import CFPredictor
from .transition_model import ProbabilisticTransitionModel, ProbabilisticStateModel
from common.utils import _handle_data

class CRESP(AUXBase):

    def __init__(self, aux_mode, cof_aux, action_shape, extr_latent_dim, nstep_of_rsd, hidden_dim,
                 output_dim, act_seq_out_dim, omg_seq_out_dim, l, rs_fc, extr_lr,
                 extr_beta, omega_opt_mode=None, num_sample=256, discount_of_rs=0.8,
                 temperature=0.1, opt_mode='min', opt_num=5, device='cpu', **kwargs):
        super().__init__()
        action_dim = action_shape[0]
        self.action_dim = action_dim
        act_seq_in_dim = nstep_of_rsd*action_dim
        # Initialize hyperparameters
        self.nstep_of_rsd = nstep_of_rsd
        self.rs_fc = rs_fc
        self.discount_of_rs = discount_of_rs
        self.pred_temp = temperature
        self.output_dim = output_dim
        self.opt_mode = opt_mode
        self.opt_num = opt_num
        self.device = device
        self.aux_mode = aux_mode
        self.cof_aux = cof_aux
        self.clipped_ratio = 1
        # Initialize modules
        self.network = CFPredictor(extr_latent_dim,
                                   act_seq_in_dim,
                                   nstep_of_rsd,
                                   hidden_dim,
                                    act_seq_out_dim,
                                   omg_seq_out_dim,
                                   output_dim, l,
                                   rs_fc=rs_fc,
                                   omega_opt_mode=omega_opt_mode,
                                   num_sample=num_sample).to(device)

        # Initialize optimizers
        # total = sum([param.nelement() for param in self.action_decoder.parameters()])
        # print("Number of parameter: %.2fM" % (total / 1e6))         
        print("Representation Mode:", self.aux_mode)
        self.action_decoder = nn.Sequential(
            nn.Linear(extr_latent_dim * 2, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, action_dim)).to(device)
        if self.aux_mode in ['a+x*sr','a+int*sr','a+c*sr']:               # 'a+int*sr'
            self.transition_model = nn.Sequential(
                nn.Linear(extr_latent_dim + action_dim, 1024),
                nn.LayerNorm(1024),
                nn.ReLU(),
                nn.Linear(1024, extr_latent_dim + 1)).to(device)
            self.model_optimizer = torch.optim.Adam(list(self.action_decoder.parameters())\
                +list(self.transition_model.parameters()), lr=extr_lr, betas=(extr_beta, 0.999))
            print("action_decoder:", self.action_decoder,"transition_model:",self.transition_model)
        elif self.aux_mode == 'r+a+s':
            self.state_decoder = nn.Sequential(
                nn.Linear(extr_latent_dim + action_dim, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, extr_latent_dim)).to(device)
            self.reward_decoder = nn.Sequential(
                nn.Linear(extr_latent_dim*2 +action_dim, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, 1)).to(device)
            self.model_optimizer = torch.optim.Adam(list(self.action_decoder.parameters())\
                +list(self.state_decoder.parameters())+list(self.reward_decoder.parameters()), lr=extr_lr, betas=(extr_beta, 0.999)) # r + a + p
        elif self.aux_mode == 'a':
            self.model_optimizer = torch.optim.Adam(self.action_decoder.parameters(), lr=extr_lr, betas=(extr_beta, 0.999))
        elif self.aux_mode == 'ras+fourier':
            self.action_decoder = nn.Sequential(
                nn.Linear(extr_latent_dim * 4, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, action_dim * 40)).to(device)

            # self.state_decoder = nn.Sequential(
            #     nn.Linear(extr_latent_dim + action_dim, 512),
            #     nn.LayerNorm(512),
            #     nn.ReLU(),
            #     nn.Linear(512, extr_latent_dim)).to(device)
            self.transition_decoder = ProbabilisticTransitionModel(extr_latent_dim, self.action_dim, 1024).to(device)
            self.state_decoder = ProbabilisticStateModel(extr_latent_dim, 1024).to(device)
            self.reward_decoder = nn.Sequential(
                nn.Linear(extr_latent_dim*4 +action_dim*4, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, 40)).to(device)
            self.model_optimizer = torch.optim.Adam(list(self.action_decoder.parameters())\
                +list(self.state_decoder.parameters())+list(self.reward_decoder.parameters())+list(self.transition_decoder.parameters()), lr=extr_lr, betas=(extr_beta, 0.999)) 
        if self.aux_mode == 'pad':
            self.pad_head = nn.Sequential(
                nn.Linear(extr_latent_dim * 2, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, action_dim)).to(device)
            self.pad_optimizer = torch.optim.Adam(self.pad_head.parameters(), lr=extr_lr, betas=(extr_beta, 0.999)) 
       
    def _prepare_data(self, data, num_aug):
        with torch.no_grad():
            traj_a, traj_r = data['traj_a'], data['traj_r']
            batch_size = traj_r.size(0)
            a_seq = traj_a.repeat(num_aug, 1, 1) # (batch_size*num_aug, rs_dim, a_dim)
            discount = (self.discount_of_rs ** torch.arange(
                self.nstep_of_rsd).to(traj_r.device)).unsqueeze(0)
            traj_r *= discount
            r_seq = traj_r.repeat(num_aug, 1) # (batch_size*num_aug, rs_dim)
        return a_seq, r_seq, batch_size

    def discrete_time_fourier_transform(self, time_domain_sequence): # Input: B, L
        with torch.no_grad():
            n_points = 20
            #omega = torch.linspace(-torch.pi, torch.pi, n_points) # (7,)
            zero = torch.tensor(0.00, requires_grad=False).to(time_domain_sequence.device)
            twiddle = torch.tensor([[torch.exp(-1j * omega * n) for n in [0,1,2]] \
                for omega in torch.linspace(-torch.pi, torch.pi, n_points)]).to(time_domain_sequence.device) #通用
            seq = torch.complex(time_domain_sequence,zero).transpose(1,0).to(time_domain_sequence.device)
            frequency_domain_signal = torch.mm(twiddle,seq)    #(n,m) @ (m,k) =(n,k)
            frequency_domain_signal = frequency_domain_signal.transpose(1,0) 
            return torch.abs(frequency_domain_signal), torch.angle(frequency_domain_signal) # Output: B, n_points

    def update_extr(self, data, extr, extr_targ, new_policy, aug, step, mode = None, use_targ = True, clip=True, up_intv=1000): # mode in {"a+sr", "a", "r+a+s"}
        #### Reward + Action inverse model + State Forward model 
        o, o2 = _handle_data(aug(data["obs"])), _handle_data(aug(data["obs2"]))
        s, s2 = extr(o), extr(o2)
        batch_size = s.shape[0]

        if mode == "ras+fourier":
            o3, o4 = _handle_data(aug(data["obs3"])), _handle_data(aug(data["obs4"]))
            s3, s4 = extr(o3), extr(o4) # o [b, 9, 84, 84] s [b, 50]
            a_seq, r_seq, batch_size = self._prepare_data(data, num_aug=1)

            s_seq = torch.cat((s, s2, s3, s4), dim =-1)  # 256, 150
            a_label = a_seq[:,0:3].transpose(2,1).reshape(batch_size * self.action_dim, -1) # b * an, 2
            r_label = r_seq[:,0:3] # b, 1, 2
            #print(a_label.shape, r_label.shape)
            saaa = torch.cat((s, (a_seq[:,0])), dim =-1)
            o_to_r = torch.cat((s_seq, a_seq[:,0:4].reshape(batch_size, -1)), dim =-1) # 256 168

            predict_a_magnitude, predict_a_phase = self.action_decoder(s_seq).reshape(-1, 40).chunk(2, -1) # b * 6, 20
            #print(predict_a_magnitude.shape, predict_a_phase.shape)
            predict_r_magnitude, predict_r_phase = self.reward_decoder(o_to_r).reshape(-1, 40).chunk(2, -1) # b, 20

            #predict_next_s2 = self.state_decoder(saaa) # b, 56

            a_magnitude, a_phase = self.discrete_time_fourier_transform(a_label) # b * 6, 20
            loss_a = F.smooth_l1_loss(predict_a_magnitude, a_magnitude, reduction='mean') \
                + F.smooth_l1_loss(predict_a_phase, a_phase, reduction='mean')

            r_magnitude, r_phase = self.discrete_time_fourier_transform(r_label) # b, 20
            loss_r = F.smooth_l1_loss(predict_r_magnitude, r_magnitude, reduction='mean') \
                + F.smooth_l1_loss(predict_r_phase, r_phase, reduction='mean')
            
            s_pre = self.state_decoder.sample_prediction(s)
            loss_state = F.smooth_l1_loss(s_pre, s, reduction='mean')

            mean_pre2, sigma_pre2 = self.transition_decoder(torch.cat((s, (a_seq[:,0])), dim =-1)) # s2
            mean2, sigma2 = self.state_decoder(s2)
            dist_pred2 = torch.distributions.Normal(mean_pre2, sigma_pre2)
            dist_targ2 = torch.distributions.Normal(mean2, sigma2)
            loss_s = torch.distributions.kl.kl_divergence(dist_pred2,dist_targ2) #[256, 50

            s3_pred = self.transition_decoder.sample_prediction(torch.cat((s2, (a_seq[:,1])), dim =-1)) # s3
            mean_pre4, sigma_pre4 = self.transition_decoder(torch.cat((s3_pred, (a_seq[:,2])), dim =-1)) # s4
            mean4, sigma4 = self.state_decoder(s4)
            dist_pred4 = torch.distributions.Normal(mean_pre4, sigma_pre4)
            dist_targ4 = torch.distributions.Normal(mean4, sigma4)
            loss_s4 = torch.distributions.kl.kl_divergence(dist_pred4,dist_targ4) #[256, 50

            with torch.no_grad():
                new_action = new_policy(s, deterministic=True, tanh=True, to_numpy=False).detach() #.reshape([batch_size,-1]) 
                rho = torch.abs(self.cof_aux /(new_action -a_seq[:,0])) 
                clipped_ratio  = torch.clamp(rho, min=0, max=1.5).detach() #256 1

            loss = loss_a + loss_r + ((loss_s+loss_s4) * clipped_ratio.mean(-1).unsqueeze(-1)).mean() + loss_state

            info_dict = dict(Loss_a=loss_a.mean().clone(), Loss_r = loss_r.mean().clone(), \
                             Loss_sf=loss_state.mean().clone(),Loss_1kl=loss_s.mean().clone(),Loss_4kl=loss_s4.mean().clone(),\
                                 Loss_aux = loss.clone(), Clipo = clipped_ratio.mean().clone())
        opt_dict = dict(opt_p=self.model_optimizer) 

        return loss, opt_dict, info_dict
    def update_pad_extr(self, data, aug, extr): 
        o, a, o2 = _handle_data(aug(data["obs"])), data['act'], _handle_data(aug(data["obs2"]))
        s, s2 = extr(o), extr(o2)
        pred_action = self.pad_head(torch.cat((s, s2), dim=-1))
        loss = F.mse_loss(pred_action, a)
        info_dict = dict(Loss_pad=loss.mean().clone())
        opt_dict = dict(opt_p=self.pad_optimizer) 
        return loss, opt_dict, info_dict
    
    def _save(self, model_dir, step):
        pass

    def _load(self, model_dir, step):
        pass

    def _print_log(self, logger):
        if self.aux_mode == 'ras+fourier':
            logger.log_tabular('Loss_aux', average_only=True)
            logger.log_tabular('Loss_a', average_only=True)
            logger.log_tabular('Loss_r', average_only=True)
            logger.log_tabular('Loss_sf', average_only=True)
            logger.log_tabular('Loss_1kl', average_only=True)
            logger.log_tabular('Loss_4kl', average_only=True)
            logger.log_tabular('Clipo', average_only=True) 
        if self.aux_mode == 'pad':
            logger.log_tabular('Loss_pad', average_only= True)