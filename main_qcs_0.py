"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import argparse  # 导入命令行参数解析库
import time  # 导入时间库
import gym  # 导入Gym库，用于强化学习环境
import numpy as np  # 导入NumPy库，用于数值计算
import wandb  # 导入wandb库，用于实验跟踪和可视化

# 从stable_baselines3库中导入SubprocVecEnv，用于多进程环境
from stable_baselines3.common.vec_env import SubprocVecEnv
from trainer.main_trainer import Maintrainer  # 导入自定义的训练器
import utils  # 导入自定义的工具模块
from utils import (
    Logger,  # 导入日志记录器
    create_model,  # 用于创建模型的函数
    create_optimizer,  # 用于创建优化器的函数
    create_scheduler,  # 用于创建学习率调度器的函数
    get_env_spec,  # 用于获取环境的规格的函数
    get_env_builder,  # 用于构建环境的函数
    load_dataset,  # 用于加载数据集的函数
    initialize_q_network,  # 用于初始化Q网络的函数
    get_q_loss_mean  # 用于获取Q损失均值的函数
)

MAX_EPISODE_LEN = 1000  # 定义最大 episode 长度


class Experiment:
    def __init__(self, variant):
        self.variant = variant  # 初始化实验参数
        self.start_time = time.time()  # 记录实验开始时间
        self._initialize_experiment_parameters()  # 初始化实验参数
        self._initialize_environment()  # 初始化环境
        self._initialize_model()  # 初始化模型
        self._initialize_dataset()  # 初始化数据集
        self._initialize_training_components()  # 初始化训练组件
        self._initialize_logging()  # 初始化日志记录器

    def _initialize_experiment_parameters(self):
        self.iter = 0  # 初始化迭代次数
        # 根据环境设置奖励尺度
        env = self.variant.get("env", "")
        if "antmaze" in env:
            self.reward_scale = 1.0  # antmaze环境奖励尺度为1.0
        elif "kitchen" in env:
            self.reward_scale = 0.01  # kitchen环境奖励尺度为0.01
        else:
            self.reward_scale = 0.001  # 默认奖励尺度为0.001

        # 设置与架构相关的参数
        base_arch = self.variant.get('base_arch', "")
        if 'mlp' in base_arch:
            context_length = 1  # 如果是mlp架构，设置上下文长度为1
        elif 'dc' in base_arch:
            context_length = 8  # 如果是dc架构，设置上下文长度为8
        else:
            context_length = None  # 默认或错误处理

        # 更新上下文长度
        if context_length is not None:
            self.variant.update({
                'K': context_length,
                'eval_context_length': context_length
            })

        # 设置与环境相关的配置
        if "antmaze" in env:
            batch_size, lr, max_iters = 256, 0.0003, 1000  # antmaze环境的batch_size、学习率和最大迭代次数
        elif "pen" in env or "kitchen" in env:
            batch_size, lr, max_iters = 64, 0.0003, 500  # pen或kitchen环境的配置
        else:
            batch_size, lr, max_iters = 64, 0.0001, 500  # 其他环境的默认配置

        # 更新实验配置
        self.variant.update({
            'batch_size': batch_size,
            'learning_rate': lr,
            'max_iters': max_iters,
            'n_head': int(self.variant['embed_dim'] // 128)  # 计算头数
        })

    def _initialize_environment(self):
        # 获取环境的规格，包括状态维度、动作维度和动作范围
        self.state_dim, self.act_dim, self.action_range = get_env_spec(self.variant["env"])
        # 如果是antmaze环境，并且使用子目标条件化
        if "antmaze" in self.variant["env"] and self.variant["conditioning"] == "subgoal":
            self.subgoal_dim = 2  # 子目标维度为2
            self.condition_dim = self.subgoal_dim  # 条件维度等于子目标维度
        else:
            self.subgoal_dim = self.state_dim  # 否则，子目标维度与状态维度相同
            self.condition_dim = 1  # 条件维度为1
        self.device = self.variant.get("device", "cuda")  # 选择设备，默认为cuda
        self.target_entropy = -self.act_dim  # 目标熵

    def _initialize_model(self):
        # 创建模型
        self.policy = create_model(self, MAX_EPISODE_LEN)
        self.qf = None  # Q函数为空
        initialize_q_network(self)  # 初始化Q网络
        self.q_loss_mean = get_q_loss_mean(self)  # 获取Q损失均值

    def _initialize_dataset(self):
        # 加载离线数据集
        self.offline_trajs, self.state_mean, self.state_std, self.max_return = load_dataset(self.variant["env"], self.variant["conditioning"])
        if "antmaze" in self.variant["env"]:
            self.state_mean = np.zeros_like(self.state_mean)  # 如果是antmaze环境，状态均值为0
            self.state_std = np.ones_like(self.state_std)  # 状态标准差为1
            
    def _initialize_training_components(self):
        # 创建Q-aid优化器
        self.policy_optimizer = create_optimizer(self.policy, self.variant["learning_rate"], self.variant["weight_decay"])
        self.policy_scheduler = create_scheduler(self.policy_optimizer, self.variant["warmup_steps"])

        # 设置训练参数
        self.trainer_params = {
            "env_name": self.variant["env"],
            "policy": self.policy,
            "policy_optimizer": self.policy_optimizer,
            "policy_scheduler": self.policy_scheduler,
            "device": self.device,
            "qf": self.qf,
            "q_loss_mean": self.q_loss_mean,
            "q_scale": self.variant["q_scale"],
            "min_q": self.variant["min_q"],
            "state_mean": self.state_mean,
            "state_std": self.state_std,
            "conditioning": self.variant["conditioning"]
        }

        # 设置数据加载器参数
        self.common_dataloader_params = {
            "batch_size": self.variant["batch_size"],
            "max_len": self.variant["K"],
            "state_dim": self.state_dim,
            "subgoal_dim": self.subgoal_dim,
            "act_dim": self.act_dim,
            "state_mean": self.state_mean,
            "state_std": self.state_std,
            "reward_scale": self.reward_scale,
            "action_range": self.action_range
        }

    def _initialize_logging(self):
        # 初始化日志记录器
        self.logger = Logger(self.variant)

    def evaluate(self, eval_fns):
        # 评估模型性能
        eval_start = time.time()
        self.policy.eval()
        outputs = {}
        for eval_fn in eval_fns:
            o = eval_fn(self.policy)  # 执行评估函数
            outputs.update(o)
        outputs["time/evaluation"] = time.time() - eval_start  # 记录评估时间

        return outputs, None


    def __call__(self):
        # 检查是否使用subgoal条件化
        if self.variant.get('conditioning') == 'subgoal':
            # 如果使用了subgoal条件化，确保环境是antmaze
            if 'antmaze' not in self.variant["env"]:
                raise RuntimeError('Only the Antmaze domain can use subgoal conditioning.')

        if self.variant['log_to_wb']:
            # 如果启用了wandb日志记录
            wandb_name = f"{self.variant['env']}"
            group_name = f"{self.variant['base_arch']} ({self.variant['conditioning']})"
            
            env = self.variant['env']
            project_name = 'QCS'
            wandb.init(
                name=wandb_name,
                group=group_name,
                project=project_name,
                config=self.variant
            )

        print("\n\nMaking Eval Env.....")
        env_name = self.variant["env"]
        if "antmaze" in env_name:
            # 如果环境是antmaze，创建并获取目标
            env = gym.make(env_name)
            target_goal = env.target_goal
            env.close()
            print(f"Generated the fixed target goal: {target_goal}")
        else:
            target_goal = None
        
        # 创建多个评估环境
        eval_envs = SubprocVecEnv(
            [
                get_env_builder(i, env_name=env_name, target_goal=target_goal)
                for i in range(self.variant["num_eval_episodes"])
            ]
        )
    
        main_trainer = Maintrainer(self)  # 初始化主训练器
        main_trainer.train(eval_envs, self.variant['log_to_wb'], self.max_return)  # 开始训练

        eval_envs.close()  # 关闭评估环境


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)  # 设置随机种子
    parser.add_argument("--env", type=str, default="hopper-medium-v2")  # 设置环境类型

    # 模型配置选项
    parser.add_argument('--base_arch', type=str, default='dc')  # 设置基础架构
    parser.add_argument("--conditioning", type=str, default='rtg', help='rtg or subgoal')  # 设置条件化类型
    parser.add_argument("--K", type=int, default=20)  # 设置上下文长度
    parser.add_argument("--embed_dim", type=int, default=256)  # 设置嵌入维度
    parser.add_argument("--n_layer", type=int, default=4)  # 设置层数
    parser.add_argument("--n_head", type=int, default=2)  # 设置头数
    parser.add_argument("--activation_function", type=str, default="relu")  # 设置激活函数
    parser.add_argument("--dropout", type=float, default=0.1)  # 设置dropout比率
    parser.add_argument("--eval_context_length", type=int, default=8)  # 设置评估时上下文长度
    parser.add_argument("--use_action", action='store_true')  # 是否使用动作
    parser.add_argument("--ordering", type=int, default=0)  # 排序选项
    
    parser.add_argument("--q_scale", type=float, default=1.0)  # 设置Q函数的缩放因子
    parser.add_argument("--min_q", type=int, default=0.0)  # 设置最小Q值

    # 评估选项
    parser.add_argument("--num_eval_episodes", type=int, default=10)  # 设置评估回合数

    # 训练选项
    parser.add_argument("--batch_size", type=int, default=16)  # 设置批次大小
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)  # 设置学习率
    parser.add_argument("--weight_decay", "-wd", type=float, default=1e-4)  # 设置权重衰减
    parser.add_argument("--warmup_steps", type=int, default=10000)  # 设置预热步骤

    # 预训练选项
    parser.add_argument("--max_iters", type=int, default=500)  # 设置最大迭代次数
    parser.add_argument("--num_updates_per_iter", type=int, default=1000)  # 每个迭代的更新次数

    # 环境选项
    parser.add_argument("--device", type=str, default="cuda")  # 设置设备为cuda
    parser.add_argument("--log_to_wb", "-w", type=bool, default=True)  # 是否记录到wandb
    parser.add_argument("--save_dir", type=str, default="./exp/seq")  # 设置保存路径
    parser.add_argument("--exp_name", type=str, default="default")  # 设置实验名称
    
    # DC卷积选项
    parser.add_argument('--conv_window_size', type=int, default=4)  # 设置卷积窗口大小
    
    # IQL参数选项
    parser.add_argument("--iql_discount", type=float, default=0.99)  # IQL折扣因子
    parser.add_argument("--iql_expectile", type=float, default=0.7)  # IQL期望分位数
    parser.add_argument("--iql_layernorm", default=False, action='store_true')  # 是否使用层归一化
    parser.add_argument("--iql_q_hiddens", type=int, default=2)  # Q网络隐藏层数
    parser.add_argument("--iql_v_hiddens", type=int, default=2)  # V网络隐藏层数

    args = parser.parse_args()  # 解析参数

    utils.set_seed_everywhere(args.seed)  # 设置随机种子
    experiment = Experiment(vars(args))  # 初始化实验

    print("=" * 50)
    experiment()  # 运行实验
