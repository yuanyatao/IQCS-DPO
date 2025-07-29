"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import argparse
import time
import gym
import numpy as np
# import wandb
import swanlab as wandb
from stable_baselines3.common.vec_env import SubprocVecEnv
from trainer.main_trainer import Maintrainer
import utils
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import faiss
from torch.utils.data import Dataset, DataLoader
import torch
from utils import (
    Logger,
    create_model,
    create_optimizer,
    create_scheduler,
    get_env_spec,
    get_env_builder,
    load_dataset,
    initialize_q_network,
    get_q_loss_mean,
    discount_cumsum,
)

MAX_EPISODE_LEN = 1000
def dpo_collate_fn(batch):
    # batch: List[Tuple[dict, dict]]
    def dicts_to_tensor(dicts, key):
        return torch.stack([torch.tensor(d[key], dtype=torch.float32) for d in dicts])
    best_chunks, worst_chunks = zip(*batch)
    # K = best_chunks[0]['observations'].shape[0]
    s_p = dicts_to_tensor(best_chunks, 'observations')
    a_p = dicts_to_tensor(best_chunks, 'actions')
    rtg_p = dicts_to_tensor(best_chunks, 'rtg')
    # 可选，如果你有timestep/mask等字段可以补充
    t_p = torch.zeros_like(rtg_p)  # 若没有就全0
    m_p = torch.ones_like(rtg_p)   # 若没有就全1
    s_d = dicts_to_tensor(worst_chunks, 'observations')
    a_d = dicts_to_tensor(worst_chunks, 'actions')
    rtg_d = dicts_to_tensor(worst_chunks, 'rtg')
    t_d = torch.zeros_like(rtg_d)
    m_d = torch.ones_like(rtg_d)
    return (s_p, a_p, rtg_p, t_p, m_p), (s_d, a_d, rtg_d, t_d, m_d)

class PreferencePairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        # 返回best, worst两个chunk
        return self.pairs[idx]
    
def _extract_chunk(traj, start_idx, K, use_rtg=True, gamma=1.0):
    chunk = {}
    chunk['observations'] = traj['observations'][start_idx: start_idx + K]
    chunk['actions'] = traj['actions'][start_idx: start_idx + K]
    chunk['rewards'] = traj['rewards'][start_idx: start_idx + K]
    # 动态计算rtg，只在用到DPO/偏好采样时传use_rtg=True
    if use_rtg:
        rtg = discount_cumsum(traj["rewards"][start_idx:], gamma=gamma)[:K]
        chunk['rtg'] = rtg
    return chunk

class Experiment:
    def __init__(self, variant):
        self.variant = variant
        self.start_time = time.time()
        self._initialize_experiment_parameters()
        self._initialize_environment()
        self._initialize_model()
        self._initialize_dataset()
        self._initialize_training_components()
        self._initialize_logging()

    def _initialize_experiment_parameters(self):
        self.iter = 0
        # Set reward scale based on the environment
        env = self.variant.get("env", "")
        if "antmaze" in env:
            self.reward_scale = 1.0
        elif "kitchen" in env:
            self.reward_scale = 0.01
        else:
            self.reward_scale = 0.001

        # Set architecture-specific parameters
        base_arch = self.variant.get('base_arch', "")
        if 'mlp' in base_arch:
            context_length = 1
        elif 'dc' in base_arch or 'dt' in base_arch or 'idt' in base_arch or 'idc' in base_arch:
            context_length = 8
            # context_length = 20  #尝试修改参数为20
        else:
            context_length = None  # Default or error handling as needed

        if context_length is not None:
            self.variant.update({
                'K': context_length,
                'eval_context_length': context_length
            })

        # Set environment-specific configurations
        if "antmaze" in env:
            batch_size, lr, max_iters = 256, 0.0003, 1000
        elif "pen" in env or "kitchen" in env:
            batch_size, lr, max_iters = 64, 0.0003, 500
        else:
            batch_size, lr, max_iters = 64, 0.0001, 500

        self.variant.update({
            'batch_size': batch_size,
            'learning_rate': lr,
            'max_iters': max_iters,
            'n_head': int(self.variant['embed_dim'] // 128)
        })

    def _initialize_environment(self):
        self.state_dim, self.act_dim, self.action_range = get_env_spec(self.variant["env"])
        if "antmaze" in self.variant["env"] and self.variant["conditioning"] == "subgoal":
            self.subgoal_dim = 2
            self.condition_dim = self.subgoal_dim
        else:
            self.subgoal_dim = self.state_dim
            self.condition_dim = 1
        self.device = self.variant.get("device", "cuda")
        self.target_entropy = -self.act_dim

    def _initialize_model(self):
        self.policy = create_model(self, MAX_EPISODE_LEN)
        self.qf = None
        initialize_q_network(self)
        self.q_loss_mean = get_q_loss_mean(self)

    def _initialize_dataset(self):
        self.offline_trajs, self.state_mean, self.state_std, self.max_return = load_dataset(self.variant["env"], self.variant["conditioning"])
        if "antmaze" in self.variant["env"]:
            self.state_mean = np.zeros_like(self.state_mean)
            self.state_std = np.ones_like(self.state_std)
            
    def _initialize_training_components(self):
        # q aid
        self.policy_optimizer = create_optimizer(self.policy, self.variant["learning_rate"], self.variant["weight_decay"])
        self.policy_scheduler = create_scheduler(self.policy_optimizer, self.variant["warmup_steps"])

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
        self.logger = Logger(self.variant)

    def evaluate(self, eval_fns):
        eval_start = time.time()
        self.policy.eval()
        outputs = {}
        for eval_fn in eval_fns:
            o = eval_fn(self.policy)
            outputs.update(o)
        outputs["time/evaluation"] = time.time() - eval_start

        return outputs, None


    def __call__(self):
        
        if self.variant.get('conditioning') == 'subgoal':
            # Check for Antmaze domain when 'subgoal' conditioning is used
            if 'antmaze' not in self.variant["env"]:
                raise RuntimeError('Only the Antmaze domain can use subgoal conditioning.')
        
        if self.variant['log_to_wb']:
            wandb_name = f"{self.variant['env']}-{self.variant['base_arch']}"
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
            env = gym.make(env_name)
            target_goal = env.target_goal
            env.close()
            print(f"Generated the fixed target goal: {target_goal}")
        else:
            target_goal = None
        eval_envs = SubprocVecEnv(
            [
                get_env_builder(i, env_name=env_name, target_goal=target_goal)
                for i in range(self.variant["num_eval_episodes"])
            ]
        )
    
        main_trainer = Maintrainer(self)
        main_trainer.train(eval_envs, self.variant['log_to_wb'], self.max_return)

        eval_envs.close()


    # 微观偏好数据集生成函数
    def create_state_aligned_preference_dataset(self, 
                                                K=8, state_threshold=0.1, reward_ratio=0.7, 
                                                num_neighbors=50, max_pairs=10000, 
                                                similarity_metric='l2', action_diff_threshold=0.1):
        """
        生成微观偏好数据集：找到轨迹中状态相似的片段对，并构建偏好数据。
        """
        print("Creating micro preference dataset...")
        assert similarity_metric in ['l2', 'cosine'], "Unsupported similarity metric"
        
        dataset = []
        state_library = []  # 存储轨迹信息和状态
        state_dim = self.state_mean.shape[0]

        # 1. 构建状态索引库
        print("构建状态索引库...")
        with ThreadPoolExecutor() as executor:
            futures = []
            for traj_idx, traj in enumerate(self.offline_trajs):   # <== 适配你的数据变量名
                futures.append(executor.submit(
                    lambda traj, idx: [(idx, t, traj['observations'][t+K-1]) for t in range(len(traj['observations']) - K)],
                    traj, traj_idx
                ))
            for future in futures:
                state_library.extend(future.result())

        print(f"构建完成，共 {len(state_library)} 个状态.")

        # 2. 提取归一化后的状态向量
        state_vectors = np.array([
            (entry[2] - self.state_mean) / self.state_std
            for entry in state_library
        ], dtype=np.float32)

        # 3. 处理余弦相似度需要L2归一化
        if similarity_metric == 'cosine':
            faiss.normalize_L2(state_vectors)
        # 4. 构建FAISS索引
        if similarity_metric == 'l2':
            index = faiss.IndexFlatL2(state_dim)
        elif similarity_metric == 'cosine':
            index = faiss.IndexFlatIP(state_dim)
        index.add(state_vectors)

        # 5. 进行最近邻搜索
        print("开始搜索相似状态...")
        cnt = 0
        with tqdm(total=max_pairs, desc="开始生成偏好对") as pbar:
            for i in range(len(state_library)):
                traj_idx1, t1, query_state = state_library[i]
                query_vector = np.array([(query_state - self.state_mean) / self.state_std], dtype=np.float32)
                distances, indices = index.search(query_vector, num_neighbors)
                candidates = [state_library[j] for j in indices[0] if j != i]

                traj1 = self.offline_trajs[traj_idx1]
                chunk1 = _extract_chunk(traj1, t1, K)

                scores = []
                for traj_idx2, t2, cand_state in candidates:
                    if np.linalg.norm(query_state - cand_state) > state_threshold:
                        continue
                    traj2 = self.offline_trajs[traj_idx2]
                    chunk2 = _extract_chunk(traj2, t2, K)

                    # 确保最后一个动作不同
                    if np.allclose(chunk1['actions'][-1], chunk2['actions'][-1], atol=action_diff_threshold):
                        continue

                    # 分数（RTG和奖励加权）
                    score = reward_ratio * chunk2['rtg'][-1] + (1 - reward_ratio) * chunk2['rewards'][-1]
                    scores.append((score, chunk2))

                if scores:
                    scores.sort(key=lambda x: x[0])
                    best_chunk = scores[-1][1]
                    worst_chunk = scores[0][1]
                    dataset.append((best_chunk, worst_chunk))
                    cnt += 1
                    pbar.update(1)
                    if cnt >= max_pairs:
                        print(f"已经达到最大对数max_pairs: ({max_pairs}), 提前停止.")
                        break
        print(f"生成了 {len(dataset)} 个微观偏好对.")
        return dataset

    def create_preference_dataloader(self):
        """
        构建DPO用的pair数据集和DataLoader，参数从self.variant读取
        """
        # 全部从variant读取，主控统一
        pairs = self.create_state_aligned_preference_dataset(
            K=self.variant.get("dpo_K", 8),
            state_threshold=self.variant.get("dpo_state_threshold", 0.1),
            reward_ratio=self.variant.get("dpo_reward_ratio", 0.7),
            num_neighbors=self.variant.get("dpo_num_neighbors", 50),
            max_pairs=self.variant.get("dpo_max_pairs", 10000),
            similarity_metric=self.variant.get("dpo_similarity_metric", "l2"),
            action_diff_threshold=self.variant.get("dpo_action_diff_threshold", 0.1),
        )
        dataset = PreferencePairDataset(pairs)
        batch_size = self.variant.get("dpo_batch_size", 64)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size, 
            shuffle=True,
            num_workers=12,
            collate_fn=dpo_collate_fn)
        return dataloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env", type=str, default="hopper-medium-v2")

    # model options
    parser.add_argument('--base_arch', type=str, default='idt', help='mlp, dt, dc, idt')
    parser.add_argument("--conditioning", type=str, default='rtg', help='rtg or subgoal')
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=2)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--eval_context_length", type=int, default=8) #原 8
    parser.add_argument("--use_action", action='store_true')
    parser.add_argument("--ordering", type=int, default=0)
    
    parser.add_argument("--q_scale", type=float, default=1.0)
    parser.add_argument("--min_q", type=int, default=0.0)

    # shared evaluation options
    parser.add_argument("--num_eval_episodes", type=int, default=10)

    # shared training options
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", "-wd", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=10000)
   
    # pretraining options
    parser.add_argument("--max_iters", type=int, default=500)
    parser.add_argument("--num_updates_per_iter", type=int, default=1000)


    # environment options
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--log_to_wb", "-w", type=bool, default=True)
    parser.add_argument("--save_dir", type=str, default="./exp/seq")
    parser.add_argument("--exp_name", type=str, default="default")
    
    # DC convolution
    parser.add_argument('--conv_window_size', type=int, default=4)
    
    parser.add_argument("--iql_discount", type=float, default=0.99)
    parser.add_argument("--iql_expectile", type=float, default=0.7)
    parser.add_argument("--iql_layernorm", default=False, action='store_true')
    parser.add_argument("--iql_q_hiddens", type=int, default=2)
    parser.add_argument("--iql_v_hiddens", type=int, default=2)
    
    # IDT options
    parser.add_argument("--idt", action='store_true', help="Use IDT training or IDC")
    parser.add_argument("--dycombine", action='store_true', help="Use dynamic combination of IDT and IDC")
    parser.add_argument("--qaid", action='store_true', help="Use Q-Aid for IDT or IDC")
    # DPO options
    parser.add_argument("--use_dpo", action='store_true', help="Use DPO training")
    parser.add_argument("--dpo_model_path", type=str, default="./exp/old_models/hopper-expert-v2-idt-rtg-seed-0/model.pt", help="Path to the DPO model for loading")
    parser.add_argument("--dpo_K", type=int, default=8, help="DPO偏好片段长度")
    parser.add_argument("--dpo_batch_size", type=int, default=64, help="DPO偏好训练batch size")
    parser.add_argument("--dpo_state_threshold", type=float, default=0.05, help="DPO片段状态距离阈值")
    parser.add_argument("--dpo_reward_ratio", type=float, default=0.7, help="DPO分数rtg权重")
    parser.add_argument("--dpo_num_neighbors", type=int, default=50, help="DPO近邻数量")
    parser.add_argument("--dpo_max_pairs", type=int, default=10000, help="DPO最大pair数")
    parser.add_argument("--dpo_similarity_metric", type=str, default="l2", help="DPO相似度类型")
    parser.add_argument("--dpo_action_diff_threshold", type=float, default=0.1, help="DPO末动作判定阈值")
    parser.add_argument("--dpo_beta", type=float, default=0.5, help="DPO beta超参数")
    
    parser.add_argument("--save_model_name", type=str, default="dpo_model_st0_05_beta_0_5", help="DPO模型保存名称")
    
    
    
    args = parser.parse_args()

    utils.set_seed_everywhere(args.seed)
    experiment = Experiment(vars(args))

    print("=" * 50)
    experiment()
