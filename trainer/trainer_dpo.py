"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import numpy as np
import torch
import time
import torch.nn.functional as F
from d4rl import infos
import swanlab as wandb
from utils import create_optimizer,create_scheduler
EXP_ADV_MAX = 100.
class StateMixingNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)
    def forward(self, states):
        # states: [B, T, state_dim]
        x = torch.relu(self.fc1(states))
        x = torch.sigmoid(self.fc2(x))
        return x.squeeze(-1)  # [B, T]

class StateLstmNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = torch.nn.LSTM(state_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = torch.nn.Linear(hidden_dim, 1)
    def forward(self, states):
        h, _ = self.lstm(states)         # [B, T, hidden_dim]
        x = torch.sigmoid(self.fc(h))    # [B, T, 1]
        return x.squeeze(-1)             # [B, T]

class IDT_Trainer:
    def __init__(
        self,
        env_name,
        policy,
        policy_optimizer,
        policy_scheduler,
        qf,
        q_loss_mean,
        state_mean,
        state_std,
        conditioning,
        q_scale,
        min_q,
        device="cuda",
        reward_scale=0.001,
        base_arch = "idt"
    ):
        self.env_name = env_name
        self.policy = policy
        self.policy_optimizer = policy_optimizer
        self.policy_scheduler = policy_scheduler
        
        self.qf = qf
        self.q_loss_mean = q_loss_mean
        self.state_mean = state_mean
        self.state_std = state_std
        self.conditioning = conditioning
        self.q_scale = q_scale
        self.min_q = min_q
        self.reward_scale = reward_scale
        self.device = device
        self.base_arch = base_arch
        # self.x_net = StateMixingNet(state_mean.shape[0]).to(self.device)
        self.x_net = StateLstmNet(state_mean.shape[0]).to(self.device)
        self.x_net_optimizer = create_optimizer(self.x_net)
        self.x_net_scheduler = create_scheduler(self.x_net_optimizer, warmup_steps=5000)
        self.start_time = time.time()

    def train_iteration(
        self,
        dataloader
    ):

        losses, action_losses, q_losses, weighted_q_losses = [], [], [], []
        logs = dict()
        train_start = time.time()

        self.policy.train()
        for i, trajs in enumerate(dataloader):
            loss, action_loss, q_loss, weighted_q_loss = self.train_step(trajs)
            losses.append(loss)
            action_losses.append(action_loss)
            q_losses.append(q_loss)
            weighted_q_losses.append(weighted_q_loss)

        logs["time/training"] = time.time() - train_start
        logs["training/action_loss_mean"] = np.mean(action_losses)
        logs["training/action_loss_std"] = np.std(action_losses)
        logs["training/q_loss_mean"] = np.mean(q_losses)
        logs["training/q_loss_std"] = np.std(q_losses)
        logs["training/loss_mean"] = np.mean(losses)
        logs["training/loss_std"] = np.std(losses)
        logs["training/weighted_q_loss_mean"] = np.mean(weighted_q_losses)
        logs["training/weighted_q_loss_std"] = np.std(weighted_q_losses)

        return logs
    
    def compute_q_based_rtg(self, states, actions, rtg, batch_size=64, gamma=0.99):
        # q_vals = self.q_function(states, actions)[0].detach()
        # q_vals = self.q_function(states, actions)[0].detach()
        state_mean_tensor = torch.from_numpy(self.state_mean).to(actions.device)
        state_std_tensor = torch.from_numpy(self.state_std).to(actions.device)
        states_flat = states.view(-1, states.shape[2])*state_std_tensor + state_mean_tensor
        action_flat = actions.view(-1, actions.shape[2])
        with torch.no_grad():  # 核心：禁用该块内所有操作的梯度追踪
            q1, q2 = self.qf(states_flat.to(dtype=torch.float32), action_flat.to(dtype=torch.float32))
            q_vals = torch.minimum(q1, q2).view(batch_size, -1)
        q_rtg = []

        # print(f"rtg shape: {rtg.shape}") [B, T+1, 1]
        # print(f"rtg top10: {rtg[:10]}")
        for i in range(q_vals.shape[0]):
            q_seq = q_vals[i]
            rtg = torch.zeros_like(q_seq)
            for t in reversed(range(len(q_seq))):
                if t == len(q_seq) - 1:
                    rtg[t] = q_seq[t]
                else:
                    rtg[t] = q_seq[t] + gamma * rtg[t + 1]
            q_rtg.append(rtg)
        based_q_rtg = torch.stack(q_rtg)
        based_q_rtg = based_q_rtg * self.reward_scale #归一化
        # print(f"based_q_rtg shape: {based_q_rtg.shape}")
        # print(f"based_q_rtg top10: {based_q_rtg[:10]}")
        
        
        return based_q_rtg.unsqueeze(-1) # [B, T, 1]，与rtg的形状一致
    def compute_combine_q_based_rtg(self, states, actions, rtg, batch_size=64, gamma=1.0):
        # q_vals = self.q_function(states, actions)[0].detach()
        # q_vals = self.q_function(states, actions)[0].detach()
        state_mean_tensor = torch.from_numpy(self.state_mean).to(actions.device)
        state_std_tensor = torch.from_numpy(self.state_std).to(actions.device)
        states_flat = states.view(-1, states.shape[2])*state_std_tensor + state_mean_tensor
        action_flat = actions.view(-1, actions.shape[2])
        with torch.no_grad():  # 核心：禁用该块内所有操作的梯度追踪
            q1, q2 = self.qf(states_flat.to(dtype=torch.float32), action_flat.to(dtype=torch.float32))
            q_vals = torch.minimum(q1, q2).view(batch_size, -1)
        q_rtg = []
        original_rtg = rtg.clone() # torch.Size([64, 9, 1])
        # print(f"rtg shape: {rtg.shape}")
        # print(f"rtg top10: {rtg[:10]}")
        for i in range(q_vals.shape[0]):
            q_seq = q_vals[i]
            rtg = torch.zeros_like(q_seq)
            for t in reversed(range(len(q_seq))):
                if t == len(q_seq) - 1:
                    rtg[t] = q_seq[t]
                else:
                    rtg[t] = q_seq[t] + gamma * rtg[t + 1]
            q_rtg.append(rtg)
        based_q_rtg = torch.stack(q_rtg)
        based_q_rtg = based_q_rtg * self.reward_scale #归一化 [B,T]
        based_q_rtg = based_q_rtg.unsqueeze(-1) # [B, T, 1]，与rtg的形状一致
        # print(f"based_q_rtg shape: {based_q_rtg.shape}")
        # print(f"based_q_rtg top10: {based_q_rtg[:10]}")
        x = self.x_net(states)   # [B, T]
        x = x.unsqueeze(-1) # [B, T, 1]
        original_rtg = original_rtg[:, :-1]  # 去掉最后一个时间步的RTG
        based_q_rtg = based_q_rtg[:, :]
        # print('original_rtg', original_rtg.shape)
        # print('based_q_rtg', based_q_rtg.shape)
        # print('x', x.shape)

        combined_rtg = original_rtg * x + based_q_rtg * (1 - x)
        
        
        return combined_rtg
    def train_step(self, trajs):
        (
            states,
            subgoals,
            actions,
            _,
            _,
            traj_returns,
            rtgs,
            timesteps,
            ordering,
            padding_mask,
        ) = [tensor.to(self.device) for tensor in trajs]

        action_target = actions.clone()
        state_mean_tensor = torch.from_numpy(self.state_mean).to(actions.device)
        state_std_tensor = torch.from_numpy(self.state_std).to(actions.device)
        batch_size, context_len = actions.shape[0], actions.shape[1]

        # rtgs = self.compute_q_based_rtg(states, actions, rtgs,batch_size)
        rtgs = self.compute_combine_q_based_rtg(states, actions, rtgs,batch_size)
        # print(f"rtgs shape: {rtgs.shape}")
        # print(f"rtgs top10: {rtgs[:10]}")
        
        if "antmaze" in self.env_name and self.conditioning == "subgoal":
            conditions = subgoals
        elif "idt" in self.base_arch or "idc" in self.base_arch:
            # print(f"直接使用rtgs作为条件:rtgs[:, :]")
            conditions = rtgs[:, :]
        else:
            # print(f"使用rtgs的前一时刻作为条件:rtgs[:, :-1]")
            conditions = rtgs[:, :-1]
            
        # Predict actions
        action_preds = self.policy.forward(
            states, conditions, actions, timesteps, ordering, padding_mask=padding_mask
        )

        # Compute action loss and update the act model
        act_dim = action_preds.shape[2]
        action_loss = F.mse_loss(action_preds, action_target, reduction="none")
        action_loss = action_loss.mean(dim=2)
        
        action_preds_flat = action_preds.view(-1, act_dim)
        states_flat = states.view(-1, states.shape[2])*state_std_tensor + state_mean_tensor
        q1, q2 = self.qf(states_flat.to(dtype=torch.float32), action_preds_flat.to(dtype=torch.float32))
        q_loss = -torch.minimum(q1, q2).view(batch_size, -1)

        reward_min = infos.REF_MIN_SCORE[f"{self.env_name}"]
        reward_max = infos.REF_MAX_SCORE[f"{self.env_name}"]
        
        normalized_returns = (traj_returns - reward_min) / (reward_max - reward_min)
        if "halfcheetah" in self.env_name:
            normalized_max_return = 0.9
        elif "hopper" in self.env_name or "walker2d" in self.env_name:
            normalized_max_return = 1.1
        else:
            normalized_max_return = 1.0
            
        q_alpha = self.q_scale * (normalized_max_return - normalized_returns) * 100
        q_alpha = torch.clamp(q_alpha, min=self.min_q).view(batch_size, -1)
        lmbda = q_alpha/self.q_loss_mean
        weighted_q_loss = lmbda * q_loss
            
        loss = action_loss + weighted_q_loss
        # loss = action_loss 
        loss = loss.view(-1, 1)[padding_mask.reshape(-1) > 0].mean()
        
        self.policy_optimizer.zero_grad(set_to_none=True)
        self.x_net_optimizer.zero_grad(set_to_none=True)

        loss.backward()
        self.policy_optimizer.step()
        self.policy_scheduler.step()
        self.x_net_optimizer.step()
        self.x_net_scheduler.step()
        
        return loss.item(), action_loss.mean().item(), q_loss.mean().item(), weighted_q_loss.mean().item()
    
    def train_dpo_epoch(self, preference_dataloader, reference_model_path, dpo_beta=1.0):
        """
        DPO训练一个epoch
        """
        # 1. 加载参考模型，只在第一次调用时加载
        if not hasattr(self, "_reference_model"):
            self._reference_model = self._load_reference_model(reference_model_path)
        self._reference_model.eval()
        for param in self._reference_model.parameters():
            param.requires_grad = False

        total_loss = 0
        total_samples = 0

        self.policy.train()
        for pref_batch, dispref_batch in preference_dataloader:
            # 保证在GPU
            s_p, a_p, rtg_p, t_p, m_p = [x.to(self.device) for x in pref_batch]
            s_d, a_d, rtg_d, t_d, m_d = [x.to(self.device) for x in dispref_batch]

            # 当前策略
            _, curr_action_p, _ = self.policy.forward(
                states=s_p, actions=a_p, returns_to_go=rtg_p, timesteps=t_p, attention_mask=m_p)
            _, curr_action_d, _ = self.policy.forward(
                states=s_d, actions=a_d, returns_to_go=rtg_d, timesteps=t_d, attention_mask=m_d)
            curr_dist_p = torch.distributions.Normal(curr_action_p, 1.0)
            curr_dist_d = torch.distributions.Normal(curr_action_d, 1.0)

            # 参考策略
            with torch.no_grad():
                _, ref_action_p, _ = self._reference_model(
                    states=s_p, actions=a_p, returns_to_go=rtg_p, timesteps=t_p, attention_mask=m_p)
                _, ref_action_d, _ = self._reference_model(
                    states=s_d, actions=a_d, returns_to_go=rtg_d, timesteps=t_d, attention_mask=m_d)
                ref_dist_p = torch.distributions.Normal(ref_action_p, 1.0)
                ref_dist_d = torch.distributions.Normal(ref_action_d, 1.0)

            # DPO损失
            log_diff_p = (curr_dist_p.log_prob(a_p) - ref_dist_p.log_prob(a_p)).sum(-1)
            log_diff_d = (curr_dist_d.log_prob(a_d) - ref_dist_d.log_prob(a_d)).sum(-1)
            loss = -torch.log(torch.sigmoid(dpo_beta * (log_diff_p - log_diff_d))).mean()

            # 优化
            self.policy_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.policy_optimizer.step()

            total_loss += loss.item() * len(a_p)
            total_samples += len(a_p)

        avg_loss = total_loss / (total_samples + 1e-6)
        return avg_loss

    def _load_reference_model(self, model_path):
        # 假定policy有init_kwargs用于重构，或可自定义
        reference_model = self.policy.__class__(**self.policy.init_kwargs).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        reference_model.load_state_dict(checkpoint["model_state_dict"])
        return reference_model    
