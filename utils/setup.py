import torch
from models.decision_transformer.base import DecisionTransformer
from models.decision_convformer.base import DecisionConvFormer
from models.mlp_actor import MLPActor
from utils import Lamb, create_vec_eval_episodes_fn

def create_model(self, MAX_EPISODE_LEN):
    if 'dt' in self.variant['base_arch'] or 'idt' in self.variant['base_arch']:
        return DecisionTransformer(
            env_name=self.variant['env'],
            state_dim=self.state_dim,
            condition_dim=self.condition_dim,
            act_dim=self.act_dim,
            action_range=self.action_range,
            use_condition=self.variant['base_arch'] != 'dt-no-condition',
            use_action=self.variant["use_action"],
            max_length=self.variant["K"],
            eval_context_length=self.variant["eval_context_length"],
            max_ep_len=MAX_EPISODE_LEN,
            hidden_size=self.variant["embed_dim"],
            n_layer=self.variant["n_layer"],
            n_head=self.variant["n_head"],
            n_inner=4 * self.variant["embed_dim"],
            activation_function=self.variant["activation_function"],
            n_positions=1024,
            resid_pdrop=self.variant["dropout"],
            attn_pdrop=self.variant["dropout"],
            ordering=self.variant["ordering"],
            # plot_attention=self.variant['plot_attention']
        ).to(device=self.device)
    if 'dc' in self.variant['base_arch'] or 'idc' in self.variant['base_arch']:
        return DecisionConvFormer(
            env_name=self.variant['env'],
            state_dim=self.state_dim,
            condition_dim=self.condition_dim,
            act_dim=self.act_dim,
            action_range=self.action_range,
            use_condition=self.variant['base_arch'] != 'dc-no-condition',
            use_action=self.variant["use_action"],
            max_length=self.variant["K"],
            eval_context_length=self.variant["eval_context_length"],
            max_ep_len=MAX_EPISODE_LEN,
            hidden_size=self.variant["embed_dim"],
            n_layer=self.variant["n_layer"],
            n_head=self.variant["n_head"],
            n_inner=4 * self.variant["embed_dim"],
            activation_function=self.variant["activation_function"],
            drop_p=self.variant["dropout"],
            ordering=self.variant["ordering"],
            window_size=self.variant['conv_window_size']
        ).to(device=self.device)
        
    if 'mlp' in self.variant['base_arch']:
        return MLPActor(
            state_dim=self.state_dim,
            condition_dim=self.condition_dim,
            act_dim=self.act_dim,
            action_range=self.action_range,
            hidden_size=self.variant["embed_dim"],
            n_layer=self.variant["n_layer"],
            use_condition=self.variant['base_arch'] != 'mlp-no-condition',
            use_action=self.variant["use_action"],
            dropout=self.variant["dropout"],
            max_length=self.variant["K"],
        ).to(device=self.device)

def create_optimizer(model, learning_rate=1e-4, weight_decay=1e-4):
    """
    创建Lamb优化器实例
    
    功能：
        为深度学习模型配置Lamb（Layer-wise Adaptive Moments optimizer for Batch training）优化器，
        该优化器结合了Adam的自适应学习率特性和LARS（Layer-wise Adaptive Rate Scaling）的层自适应缩放策略，
        适用于大规模分布式训练场景。
    
    参数：
        model：训练的模型实例，需包含可训练参数（model.parameters()）
        learning_rate（float）：初始学习率，控制参数更新的步长
        weight_decay（float）：权重衰减系数，用于防止模型过拟合（L2正则化）
    
    返回：
        Lamb优化器实例，用于模型参数的梯度更新
    
    核心公式（Lamb优化器参数更新逻辑）：
        1. 计算参数梯度 g_t 和一阶动量 m_t（类似Adam的动量）：
            m_t = β1 * m_{t-1} + (1 - β1) * g_t
        2. 计算二阶动量 v_t（类似Adam的自适应缩放）：
            v_t = β2 * v_{t-2} + (1 - β2) * g_t^2
        3. 对动量和二阶动量进行偏差修正：
            m_t_hat = m_t / (1 - β1^t)，v_t_hat = v_t / (1 - β2^t)
        4. 计算参数更新量（结合LARS的层自适应缩放）：
            update_t = (m_t_hat / (√v_t_hat + eps)) * (||θ_t|| / (||g_t|| + weight_decay * ||θ_t||))
        5. 最终参数更新：θ_{t+1} = θ_t - learning_rate * update_t
        
        （注：eps为数值稳定性常数，此处固定为1e-8；β1、β2为Lamb默认超参数，通常为0.9和0.999）
    """
    return Lamb(
        model.parameters(),  # 模型的可训练参数
        lr=learning_rate,    # 初始学习率
        weight_decay=weight_decay,  # 权重衰减（L2正则化）
        eps=1e-8             # 数值稳定性参数，避免分母为0
    )

# def create_scheduler(optimizer, warmup_steps):
#     return torch.optim.lr_scheduler.LambdaLR(
#         optimizer, lambda steps: min((steps + 1) / warmup_steps, 1)
#     )
#     # return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_steps)
def create_scheduler(optimizer, warmup_steps):
    """
    创建线性预热学习率调度器
    
    功能：
        为优化器配置学习率调度策略，实现"线性预热"效果：在训练初期（预热阶段）学习率从0线性增长至初始学习率，
        预热结束后保持初始学习率不变，用于缓解模型训练初期的不稳定性。
    
    参数：
        optimizer：已初始化的优化器实例（如Lamb、Adam等）
        warmup_steps（int）：预热步数，即学习率增长到初始值所需的训练步数
    
    返回：
        LambdaLR调度器实例，用于动态调整学习率
    
    核心公式（学习率调整逻辑）：
        设当前训练步数为 steps（从0开始计数），则学习率缩放因子为：
            lr_scale = min( (steps + 1) / warmup_steps, 1 )
        实际学习率 = 初始学习率 * lr_scale
        
        逻辑说明：
            - 当 steps < warmup_steps 时：lr_scale = (steps + 1)/warmup_steps，学习率随步数线性增长（从0→初始学习率）
            - 当 steps ≥ warmup_steps 时：lr_scale = 1，学习率保持初始值不变
    """
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,  # 待调整学习率的优化器
        # 自定义学习率缩放函数：线性预热+恒定阶段
        lambda steps: min((steps + 1) / warmup_steps, 1)
    )
def create_eval_function(self, eval_envs, eval_rtg, stage):
    return create_vec_eval_episodes_fn(
        vec_env=eval_envs,
        env_name=self.variant["env"],
        eval_rtg=eval_rtg,
        state_dim=self.state_dim,
        subgoal_dim=self.subgoal_dim,
        act_dim=self.act_dim,
        state_mean=self.state_mean,
        state_std=self.state_std,
        reward_scale=self.reward_scale,
        device=self.device,
        stage=stage
    )
