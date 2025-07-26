"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import gym
import numpy as np
import torch
# import wandb
import swanlab as wandb
from d4rl import infos

MAX_EPISODE_LEN = 1000


# def create_vec_eval_episodes_fn(
#     vec_env,
#     env_name,
#     eval_rtg,
#     state_dim,
#     subgoal_dim,
#     act_dim,
#     state_mean,
#     state_std,
#     reward_scale,
#     device,
#     stage,
# ):
#     def eval_episodes_fn(actor):
#         target_return = [eval_rtg * reward_scale] * vec_env.num_envs
#         returns, lengths, _ = vec_evaluate_seq(
#             vec_env,
#             state_dim,
#             subgoal_dim,
#             act_dim,
#             actor,
#             max_ep_len=MAX_EPISODE_LEN,
#             reward_scale=reward_scale,
#             target_return=target_return,
#             mode="normal",
#             state_mean=state_mean,
#             state_std=state_std,
#             device=device,
#         )
#         suffix = f"{stage} "
#         reward_min = infos.REF_MIN_SCORE[env_name]
#         reward_max = infos.REF_MAX_SCORE[env_name]
#         return {
#             f"{suffix}evaluation/{int(eval_rtg)} return_mean": np.mean(returns),
#             # f"{suffix}evaluation/{int(eval_rtg)} return_std": np.std(returns),
#             # f"{suffix}evaluation/{int(eval_rtg)} length_mean": np.mean(lengths),
#             # f"{suffix}evaluation/{int(eval_rtg)} length_std": np.std(lengths),
#             f"{suffix}evaluation/{int(eval_rtg)} d4rl_score": (np.mean(returns) - reward_min) * 100 / (reward_max - reward_min),
#         }

#     return eval_episodes_fn
def create_vec_eval_episodes_fn(
    vec_env,               # 向量环境（多环境并行）
    env_name,              # 环境名称（用于获取D4RL参考分数）
    eval_rtg,              # 评估时的目标回报（Return-to-Go）
    state_dim,             # 状态空间维度
    subgoal_dim,           # 子目标维度（若环境包含子目标）
    act_dim,               # 动作空间维度
    state_mean,            # 状态标准化的均值
    state_std,             # 状态标准化的标准差
    reward_scale,          # 回报缩放因子
    device,                # 计算设备（CPU/GPU）
    stage,                 # 当前训练阶段（用于日志标记）
):
    """
    创建用于评估向量环境中episode性能的函数
    
    返回值:
        eval_episodes_fn: 评估函数，调用后返回包含评估指标的字典
    """
    def eval_episodes_fn(actor):
        """
        评估策略网络（actor）在向量环境中的性能
        
        参数:
            actor: 待评估的策略网络
            
        返回值:
            包含平均回报、D4RL标准化分数等指标的字典
        """
        # 设置每个环境的目标回报（按回报缩放因子调整）
        target_return = [eval_rtg * reward_scale] * vec_env.num_envs
        
        # 调用向量环境评估函数，获取回报和长度数据
        returns, lengths, _ = vec_evaluate_seq(
            vec_env,               # 向量环境
            state_dim,             # 状态维度
            subgoal_dim,           # 子目标维度
            act_dim,               # 动作维度
            actor,                 # 策略网络
            max_ep_len=MAX_EPISODE_LEN,  # 最大episode长度
            reward_scale=reward_scale,   # 回报缩放因子
            target_return=target_return, # 目标回报列表
            mode="normal",         # 评估模式（正常模式）
            state_mean=state_mean, # 状态均值（用于标准化）
            state_std=state_std,   # 状态标准差（用于标准化）
            device=device,         # 计算设备
        )
        
        # 日志标记后缀（区分不同训练阶段）
        suffix = f"{stage} "
        # 获取环境的参考分数范围（用于计算D4RL标准化分数）
        reward_min = infos.REF_MIN_SCORE[env_name]  # 环境最小参考分数
        reward_max = infos.REF_MAX_SCORE[env_name]  # 环境最大参考分数
        
        # 返回评估指标字典
        return {
            # 平均回报（按目标RTG和阶段标记）
            f"{suffix}evaluation/{int(eval_rtg)} return_mean": np.mean(returns),
            # D4RL标准化分数（将原始回报映射到0-100范围）
            f"{suffix}evaluation/{int(eval_rtg)} d4rl_score": 
                (np.mean(returns) - reward_min) * 100 / (reward_max - reward_min),
            # f"{suffix}evaluation/{int(eval_rtg)} return_std": np.std(returns),
            # f"{suffix}evaluation/{int(eval_rtg)} length_mean": np.mean(lengths),
            # f"{suffix}evaluation/{int(eval_rtg)} length_std": np.std(lengths),
        }
 
    return eval_episodes_fn

# def vec_evaluate_seq(
#     vec_env,
#     state_dim,
#     subgoal_dim,
#     act_dim,
#     actor,
#     target_return: list,
#     value_buffer=None,
#     value_update=None,
#     max_ep_len=1000,
#     reward_scale=0.001,
#     state_mean=0.0,
#     state_std=1.0,
#     device="cuda",
#     mode="normal",
#     total_transitions_sampled=0,
# ):

#     actor.eval()
#     actor.to(device=device)

#     state_mean = torch.from_numpy(state_mean).to(device=device)
#     state_std = torch.from_numpy(state_std).to(device=device)

#     num_envs = vec_env.num_envs
#     state = vec_env.reset()
    
#     if subgoal_dim == 2:
#         target_goal = np.array(vec_env.get_attr("target_goal"))
    
#     # we keep all the histories on the device
#     # note that the latest action and reward will be "padding"
#     states = (
#         torch.from_numpy(state)
#         .reshape(num_envs, state_dim)
#         .to(device=device, dtype=torch.float32)
#     ).reshape(num_envs, -1, state_dim)
#     subgoals = torch.empty(num_envs, 0, subgoal_dim).to(device=device, dtype=torch.float32)
#     actions = torch.zeros(0, device=device, dtype=torch.float32)
#     rewards = torch.zeros(0, device=device, dtype=torch.float32)
#     state = torch.from_numpy(state).to(device=device).reshape(num_envs, -1, state_dim).to(dtype=torch.float32)
#     prev_state = states
    
#     ep_return = target_return
#     target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(
#         num_envs, -1, 1
#     )

#     timesteps = torch.tensor([0] * num_envs, device=device, dtype=torch.long).reshape(
#         num_envs, -1
#     )

#     # episode_return, episode_length = 0.0, 0
#     episode_return = np.zeros((num_envs, 1)).astype(float)
#     episode_length = np.full(num_envs, np.inf)

#     unfinished = np.ones(num_envs).astype(bool)
#     for t in range(max_ep_len):
#         # add padding
#         actions = torch.cat(
#             [
#                 actions,
#                 torch.zeros((num_envs, act_dim), device=device).reshape(
#                     num_envs, -1, act_dim
#                 ),
#             ],
#             dim=1,
#         )
#         rewards = torch.cat(
#             [
#                 rewards,
#                 torch.zeros((num_envs, 1), device=device).reshape(num_envs, -1, 1),
#             ],
#             dim=1,
#         )
        
#         with torch.no_grad():
#             if subgoal_dim == 2:
#                 subgoal = (
#                     torch.from_numpy(target_goal)
#                     .reshape(num_envs, subgoal_dim)
#                     .to(device=device, dtype=torch.float32)
#                 ).reshape(num_envs, -1, subgoal_dim)
#                 subgoals = torch.cat([subgoals, (subgoal)], dim=1)
#                 conditions= subgoals
#             else:
#                 conditions = target_return.to(dtype=torch.float32)
#             action = actor.get_action_predictions(
#                 (states.to(dtype=torch.float32) - state_mean) / state_std,
#                 conditions,
#                 actions.to(dtype=torch.float32),
#                 timesteps.to(dtype=torch.long),
#                 num_envs=num_envs,
#             )
            
#         action = action.clamp(*actor.action_range)

#         state, reward, done, _ = vec_env.step(action.detach().cpu().numpy())
        
#         # eval_env.step() will execute the action for all the sub-envs, for those where
#         # the episodes have terminated, the envs will be reset. Hence we use
#         # "unfinished" to track whether the first episode we roll out for each sub-env is
#         # finished. In contrast, "done" only relates to the current episode
#         episode_return[unfinished] += reward[unfinished].reshape(-1, 1)
        
#         torch_state = torch.from_numpy(state).to(device=device).reshape(num_envs, -1, state_dim)
#         torch_reward = torch.from_numpy(reward).to(device=device).reshape(num_envs, 1)
                
#         actions[:, -1] = action
#         states = torch.cat([states, (torch_state)], dim=1)
#         rewards[:, -1] = torch_reward
        
#         if mode != "delayed":
#             pred_return = target_return[:, -1] - (torch_reward * reward_scale)
#         else:
#             pred_return = target_return[:, -1]
#         target_return = torch.cat(
#             [target_return, pred_return.reshape(num_envs, -1, 1)], dim=1
#         )

#         timesteps = torch.cat(
#             [
#                 timesteps,
#                 torch.ones((num_envs, 1), device=device, dtype=torch.long).reshape(
#                     num_envs, 1
#                 )
#                 * (t + 1),
#             ],
#             dim=1,
#         )

#         if t == max_ep_len - 1:
#             done = np.ones(done.shape).astype(bool)

#         if np.any(done):
#             ind = np.where(done)[0]
#             unfinished[ind] = False
#             episode_length[ind] = np.minimum(episode_length[ind], t + 1)

#         if not np.any(unfinished):
#             break


#     trajectories = []
#     for ii in range(num_envs):
#         ep_len = episode_length[ii].astype(int)
#         terminals = np.zeros(ep_len)
#         terminals[-1] = 1
#         traj = {
#             "observations": states[ii].detach().cpu().numpy()[:ep_len],
#             "next_observations": states[ii].detach().cpu().numpy()[1:ep_len+1],
#             "actions": actions[ii].detach().cpu().numpy()[:ep_len],
#             "rewards": rewards[ii].detach().cpu().numpy()[:ep_len],
#             "terminals": terminals,
#         }
#         trajectories.append(traj)

#     return (
#         episode_return.reshape(num_envs),
#         episode_length.reshape(num_envs),
#         trajectories,
#     )
    
def vec_evaluate_seq(
    vec_env,                   # 向量环境（多环境并行）
    state_dim,                 # 状态空间维度
    subgoal_dim,               # 子目标维度（若有）
    act_dim,                   # 动作空间维度
    actor,                     # 策略网络（用于生成动作）
    target_return: list,       # 每个环境的目标回报列表
    value_buffer=None,         # 值函数缓存（可选）
    value_update=None,         # 值函数更新函数（可选）
    max_ep_len=1000,           # 最大episode长度
    reward_scale=0.001,        # 回报缩放因子
    state_mean=0.0,            # 状态均值（标准化用）
    state_std=1.0,             # 状态标准差（标准化用）
    device="cuda",             # 计算设备
    mode="normal",             # 评估模式（normal/delayed）
    total_transitions_sampled=0,  # 已采样的总转移次数（用于日志）
):
    """
    在向量环境中评估策略网络的序列决策性能，收集轨迹数据并计算回报
    
    参数:
        vec_env: 并行化的强化学习环境
        actor: 策略网络（Actor）
        target_return: 每个环境的目标回报值
        其他参数: 环境配置、标准化参数、计算设备等
        
    返回值:
        episode_return: 每个环境的最终回报
        episode_length: 每个环境的episode长度
        trajectories: 包含状态、动作、回报等的轨迹数据列表
    """
    # 将策略网络设为评估模式（关闭dropout等训练特有的层）
    actor.eval()
    # 将策略网络移动到指定计算设备
    actor.to(device=device)
 
    # 将状态标准化参数转换为张量并移动到设备
    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)
 
    # 获取向量环境中的环境数量
    num_envs = vec_env.num_envs
    # 重置所有环境，获取初始状态
    state = vec_env.reset()
    
    # 若子目标维度为2，获取环境的目标goal（特定环境如FetchReach）
    if subgoal_dim == 2:
        target_goal = np.array(vec_env.get_attr("target_goal"))
    
    # 将初始状态转换为张量并标准化（形状：[num_envs, 1, state_dim]）
    states = (
        torch.from_numpy(state)
        .reshape(num_envs, state_dim)
        .to(device=device, dtype=torch.float32)
    ).reshape(num_envs, -1, state_dim)
    # 初始化子目标张量（若有子目标）
    subgoals = torch.empty(num_envs, 0, subgoal_dim).to(device=device, dtype=torch.float32)
    # 初始化动作和回报张量（用于存储序列数据）
    actions = torch.zeros(0, device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    # 初始状态张量（用于循环中的状态更新）
    state = torch.from_numpy(state).to(device=device).reshape(num_envs, -1, state_dim).to(dtype=torch.float32)
    prev_state = states  # 历史状态（可能用于递归模型）
    
    # 设置目标回报（转换为张量并调整形状：[num_envs, 1, 1]）
    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(
        num_envs, -1, 1
    )
 
    # 初始化时间步张量（记录每个环境的当前步数）
    timesteps = torch.tensor([0] * num_envs, device=device, dtype=torch.long).reshape(
        num_envs, -1
    )
 
    # 初始化回报和长度记录数组
    episode_return = np.zeros((num_envs, 1)).astype(float)  # 每个环境的累积回报
    episode_length = np.full(num_envs, np.inf)  # 每个环境的episode长度（初始设为无穷大）
 
    # 跟踪未完成episode的环境（初始所有环境均未完成）
    unfinished = np.ones(num_envs).astype(bool)
    
    # 循环执行动作，直到达到最大episode长度或所有环境完成
    for t in range(max_ep_len):
        # 扩展动作张量（添加当前时间步的动作占位符）
        actions = torch.cat(
            [
                actions,
                torch.zeros((num_envs, act_dim), device=device).reshape(
                    num_envs, -1, act_dim
                ),
            ],
            dim=1,
        )
        # 扩展回报张量（添加当前时间步的回报占位符）
        rewards = torch.cat(
            [
                rewards,
                torch.zeros((num_envs, 1), device=device).reshape(num_envs, -1, 1),
            ],
            dim=1,
        )
        
        # 禁用梯度计算（评估阶段无需反向传播）
        with torch.no_grad():
            # 若子目标维度为2，获取当前目标goal并添加到子目标序列
            if subgoal_dim == 2:
                subgoal = (
                    torch.from_numpy(target_goal)
                    .reshape(num_envs, subgoal_dim)
                    .to(device=device, dtype=torch.float32)
                ).reshape(num_envs, -1, subgoal_dim)
                subgoals = torch.cat([subgoals, (subgoal)], dim=1)
                conditions = subgoals  # 以子目标作为条件
            else:
                conditions = target_return.to(dtype=torch.float32)  # 以目标回报作为条件
            
            # 调用策略网络生成动作（输入状态、条件、历史动作和时间步）
            action = actor.get_action_predictions(
                (states.to(dtype=torch.float32) - state_mean) / state_std,  # 标准化状态
                conditions,  # 条件（子目标或目标回报）
                actions.to(dtype=torch.float32),  # 历史动作序列
                timesteps.to(dtype=torch.long),  # 时间步
                num_envs=num_envs,  # 环境数量
            )
            
        # 将动作裁剪到策略网络定义的动作范围内
        action = action.clamp(*actor.action_range)
 
        # 在环境中执行动作，获取下一个状态、回报和结束标志
        state, reward, done, _ = vec_env.step(action.detach().cpu().numpy())
        
        # 更新未完成环境的累积回报（仅对未结束的episode累加回报）
        episode_return[unfinished] += reward[unfinished].reshape(-1, 1)
        
        # 将状态和回报转换为张量并调整形状
        torch_state = torch.from_numpy(state).to(device=device).reshape(num_envs, -1, state_dim)
        torch_reward = torch.from_numpy(reward).to(device=device).reshape(num_envs, 1)
                
        # 更新当前时间步的动作和状态序列
        actions[:, -1] = action  # 将生成的动作存入动作序列
        states = torch.cat([states, (torch_state)], dim=1)  # 追加新状态到状态序列
        rewards[:, -1] = torch_reward  # 存入当前回报
        
        # 根据模式更新目标回报（正常模式下减去当前回报的缩放值）
        if mode != "delayed":
            pred_return = target_return[:, -1] - (torch_reward * reward_scale)
        else:
            pred_return = target_return[:, -1]  # 延迟模式下不更新目标回报
        # 将新的预测回报追加到目标回报序列
        target_return = torch.cat(
            [target_return, pred_return.reshape(num_envs, -1, 1)], dim=1
        )
 
        # 更新时间步（当前步数+1）
        timesteps = torch.cat(
            [
                timesteps,
                torch.ones((num_envs, 1), device=device, dtype=torch.long).reshape(
                    num_envs, 1
                )
                * (t + 1),
            ],
            dim=1,
        )
 
        # 若达到最大episode长度，强制标记所有环境为结束
        if t == max_ep_len - 1:
            done = np.ones(done.shape).astype(bool)
 
        # 检查是否有环境结束，更新未完成标记和episode长度
        if np.any(done):
            ind = np.where(done)[0]  # 找到结束的环境索引
            unfinished[ind] = False  # 标记这些环境为已完成
            episode_length[ind] = np.minimum(episode_length[ind], t + 1)  # 记录实际长度
 
        # 若所有环境均已完成，提前退出循环
        if not np.any(unfinished):
            break
 
    # 整理轨迹数据（每个环境一个轨迹字典）
    trajectories = []
    for ii in range(num_envs):
        ep_len = episode_length[ii].astype(int)  # 当前环境的episode长度
        terminals = np.zeros(ep_len)  # 终止标志（最后一步为1）
        terminals[-1] = 1
        # 构建轨迹字典，包含状态、动作、回报等信息
        traj = {
            "observations": states[ii].detach().cpu().numpy()[:ep_len],  # 状态序列
            "next_observations": states[ii].detach().cpu().numpy()[1:ep_len+1],  # 下一个状态序列
            "actions": actions[ii].detach().cpu().numpy()[:ep_len],  # 动作序列
            "rewards": rewards[ii].detach().cpu().numpy()[:ep_len],  # 回报序列
            "terminals": terminals,  # 终止标志序列
        }
        trajectories.append(traj)
 
    # 返回每个环境的回报、长度和轨迹数据
    return (
        episode_return.reshape(num_envs),  # 形状：[num_envs]
        episode_length.reshape(num_envs),  # 形状：[num_envs]
        trajectories,  # 轨迹列表（长度为num_envs）
    )