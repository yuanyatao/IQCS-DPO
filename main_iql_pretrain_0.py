# 导入必要的库
import numpy as np
import torch
import gym  # 强化学习环境库
import argparse  # 命令行参数解析
import os  # 操作系统接口
import d4rl  # 离线强化学习数据集库
from tqdm import trange  # 进度条显示
from coolname import generate_slug  # 生成随机字符串用于实验命名
import wandb  # 实验跟踪和可视化工具
# import gymnasium as gym
# 导入自定义工具函数和IQL算法实现
import utils
from models.iql.base import IQL

# 设置计算设备（优先使用GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 主程序入口
if __name__ == "__main__":

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    
    #################################### 实验相关参数 ####################################
    parser.add_argument("--policy", default="IQL")                 # 使用的策略算法名称
    parser.add_argument("--env", default="Hopper-v3")   # 强化学习环境名称
    parser.add_argument("--seed", default=0, type=int)             # 随机种子（确保实验可重复）
    parser.add_argument("--save_freq", default=5e5, type=int)      # 模型保存频率（每多少步保存一次）
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # 最大训练步数
    parser.add_argument("--save_model", action="store_true", default=True)  # 是否保存模型参数
    parser.add_argument("--log_to_wb", "-w", type=bool, default=False)  # 是否使用wandb记录实验
    parser.add_argument("--normalize", default=False, action='store_true')  # 是否对状态进行标准化
    
    ##################################### IQL算法参数 ####################################
    parser.add_argument("--batch_size", default=16, type=int)     # 训练批次大小
    parser.add_argument("--hidden_dim", default=256, type=int)     # 神经网络隐藏层维度
    parser.add_argument("--expectile", default=0.7, type=float)    # IQL中的期望分位数（用于价值函数估计）
    parser.add_argument("--tau", default=0.005, type=float)        # 目标网络软更新系数
    parser.add_argument("--discount", default=0.99, type=float)    # 未来奖励折扣因子
    parser.add_argument("--q_hiddens", default=2, type=int)        # Q网络隐藏层数量
    parser.add_argument("--v_hiddens", default=2, type=int)        # V网络隐藏层数量
    parser.add_argument("--layernorm", default=False, action='store_true')  # 是否使用层标准化
    parser.add_argument("--save_dir", type=str, default="./exp")   # 模型保存目录
    
    # 解析命令行参数并生成随机实验名称
    args = parser.parse_args()
    args.cooldir = generate_slug(2)  # 生成易记的随机名称用于实验记录

    # 打印关键参数信息
    print("---------------------------------------")
    print(f"策略: {args.policy}, 环境: {args.env}, 随机种子: {args.seed}")
    print("---------------------------------------")

    # 创建强化学习环境
    env = gym.make(args.env)

    ############################## 设置随机种子（确保实验可重复）##############################
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 获取环境的状态和动作维度信息
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])  # 动作空间的最大值

    # 初始化经验回放缓冲区
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    # 加载D4RL离线数据集
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
    
    # 对特定环境（antmaze）进行特殊处理
    if 'antmaze' in args.env:
        replay_buffer.reward = replay_buffer.reward - 1.0  # 调整奖励（根据论文实现）
        args.q_hiddens = 3  # 调整Q网络结构
        args.v_hiddens = 3  # 调整V网络结构
    
    # 状态标准化处理
    if args.normalize:
        mean, std = replay_buffer.normalize_states()  # 计算并应用标准化参数
    else:
        mean, std = 0, 1  # 不使用标准化

    #################################### 实验记录配置 ####################################
    if args.log_to_wb:
        wandb_name = f"{args.env}"          # 实验名称
        group_name = "q-pretrain"           # 实验分组名称
        project_name = "QCS-Q-Pretrain"     # wandb项目名称
        
        # 初始化wandb配置
        wandb.init(
            name=wandb_name,
            group=group_name,
            project=project_name,
            config=args
        )

    #################################### 创建保存目录 ####################################
    utils.mkdir(args.save_dir)  # 创建根目录
    base_dir = os.path.join(args.save_dir, f"{(args.policy).lower()}")  # 策略子目录
    utils.mkdir(base_dir)
    args.work_dir = os.path.join(base_dir, args.env)  # 环境子目录
    utils.mkdir(args.work_dir)
    
    args.model_dir = os.path.join(args.work_dir, str(args.seed))  # 随机种子子目录
    utils.mkdir(args.model_dir)

    #################################### 初始化策略模型 ####################################
    # 构建模型参数字典
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "discount": args.discount,
        "tau": args.tau,
        "expectile": args.expectile,
        "hidden_dim": args.hidden_dim,
        "q_hiddens": args.q_hiddens,
        "v_hiddens": args.v_hiddens,
        "layernorm": args.layernorm,
    }

    # 实例化IQL策略
    policy = IQL(**kwargs)

    ###################################### 训练循环 ######################################
    # 使用trange创建带进度条的训练循环
    for t in trange(int(args.max_timesteps)):
        # 从经验池采样并训练模型
        policy.train(replay_buffer, args.batch_size, log_to_wb=args.log_to_wb)
        
        # 定期保存模型
        if (t + 1) % args.save_freq == 0:
            policy.save(args.model_dir)
    
    # 训练结束后保存最终模型
    policy.save(args.model_dir)

