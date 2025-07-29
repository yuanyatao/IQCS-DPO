import time
# import wandb
import swanlab as wandb
from utils import create_eval_function, create_dataloader, save_model,create_optimizer,create_scheduler
from trainer.seq_trainer import SequenceTrainer
from trainer.act_trainer import ActTrainer
from trainer.idt_trainer import IDT_Trainer
import copy
import torch
class Maintrainer:
    def __init__(self, experiment):
        self.experiment = experiment
        self.variant = experiment.variant
        self.iter = 0

    def train(self, eval_envs, log_to_wb, max_return,save_model_name="model"):
        self._initialize_trainer_and_eval_functions(eval_envs, max_return)
        if self.variant["use_dpo"]:
            print(">> 进入DPO训练模式")
            
            save_model_name = self.variant.get("save_model_name", "model")
            print(f"保存模型的名称后缀: {save_model_name}")
            preference_dataloader = self.experiment.create_preference_dataloader()
            
            reference_model_path = self.variant.get("dpo_model_path", None)
            assert reference_model_path is not None, "DPO模式需要指定dpo_model_path参数"
            print(f"加载预训练策略权重: {reference_model_path}")

            # 正确读取policy的state_dict
            checkpoint = torch.load(reference_model_path, map_location=self.variant.get("device", "cpu"))
            self.trainer.policy.load_state_dict(checkpoint["policy_state_dict"])

            # 复制参考模型
            self._reference_model = copy.deepcopy(self.trainer.policy)
            for p in self._reference_model.parameters():
                p.requires_grad_(False)
            self._reference_model.eval()

            # # 通知训练器
            # self.trainer.set_reference_model(reference_model)
            best_score = -float('inf')
            best_epoch = -1
            for epoch in range(self.variant["max_iters"]):
                # 1. 训练一个epoch
                dpo_loss = self.trainer.train_dpo_epoch(
                    preference_dataloader=preference_dataloader,
                    reference_model=self._reference_model,
                    dpo_beta=self.variant.get("dpo_beta", 1.0)
                )

                # 2. 验证性能
                eval_outputs, eval_reward = self.experiment.evaluate(self.eval_fns)
                outputs = {"dpo_loss": dpo_loss, "epoch": epoch}
                outputs.update(eval_outputs)
                if log_to_wb:
                    wandb.log(outputs, print_to_console=True)
                print(f"[DPO][Epoch {epoch}] loss={dpo_loss:.4f}, eval={eval_outputs}")

                # 3. 取某个关键评价指标作为最佳判据（如d4rl_score）
                # 用第一个d4rl_score为例
                d4rl_keys = [k for k in eval_outputs if "d4rl_score" in k]
                if len(d4rl_keys) > 0:
                    score1 = eval_outputs[d4rl_keys[0]]
                    score2 = eval_outputs[d4rl_keys[1]]
                    score = max(score1, score2)
                    if score > best_score:
                        best_score = score
                        best_epoch = epoch
                        save_model(self.experiment, path_prefix=self.experiment.logger.log_path, postfix=save_model_name)
                        print(f">> 当前最佳模型已保存：epoch {best_epoch} | {d4rl_keys[0]}={best_score:.2f}")

            print(f"DPO训练完成！最佳分数={best_score:.2f}，最佳epoch={best_epoch}")
        else:
            # 正常训练分支
            print(">> 进入正常训练模式")
            best_score = -float('inf')
            best_epoch = -1
            while self.iter < self.variant["max_iters"]:
                self.experiment.common_dataloader_params.update({
                    "trajectories": self.experiment.offline_trajs,
                    "num_iters": self.variant["num_updates_per_iter"]
                })
                dataloader = create_dataloader(**self.experiment.common_dataloader_params)

                train_outputs = self.trainer.train_iteration(dataloader=dataloader)
                eval_outputs, eval_reward = self.experiment.evaluate(self.eval_fns)

                outputs = self._prepare_outputs(train_outputs, eval_outputs)

                if log_to_wb:
                    wandb.log(data = outputs, print_to_console=True)

                self.iter += 1
                d4rl_keys = [k for k in eval_outputs if "d4rl_score" in k]
                if len(d4rl_keys) > 0:
                    score1 = eval_outputs[d4rl_keys[0]]
                    score2 = eval_outputs[d4rl_keys[1]]
                    score = max(score1, score2)
                    if score > best_score:
                        best_score = score
                        best_epoch = self.iter
                        save_model(self.experiment, path_prefix=self.experiment.logger.log_path, postfix=save_model_name)
                        print(f">> 当前最佳模型已保存：epoch {best_epoch} | {d4rl_keys[0]}={best_score:.2f}")

                # save_model(
                #     self.experiment,
                #     path_prefix=self.experiment.logger.log_path
                # )

    def _initialize_trainer_and_eval_functions(self, eval_envs, max_return):
        if "no-condition" in self.variant["base_arch"] or self.variant["conditioning"] == "subgoal":
            target_rtg = [max_return] 
        else:
            if "antmaze" in self.variant["env"]:
                target_rtg = [max_return, 100*max_return]
            else:
                target_rtg = [max_return, 2*max_return]
        self.eval_fns = [create_eval_function(self.experiment, eval_envs, tar, "OFFLINE") for tar in target_rtg]

        # wandb.log(
        #     data={
        #         "experiment/base_arch": self.variant["base_arch"],
        #     },
        #      print_to_console=True
        #     )
        if "mlp" in self.variant["base_arch"]:
            print(f"训练架构：{self.variant['base_arch']}")
            self.trainer = ActTrainer(**self.experiment.trainer_params)
        elif "idt" in self.variant["base_arch"] or "idc" in self.variant["base_arch"]:
            print(f"训练架构：{self.variant['base_arch']}")
            self.trainer = IDT_Trainer(**self.experiment.trainer_params)
        elif "dt" in self.variant["base_arch"] or "dc" in self.variant["base_arch"]:
            print(f"训练架构：{self.variant['base_arch']}")
            self.trainer = SequenceTrainer(**self.experiment.trainer_params)

    def _prepare_outputs(self, train_outputs, eval_outputs):
        outputs = {"time/total": time.time() - self.experiment.start_time}
        outputs.update(train_outputs)
        outputs.update(eval_outputs)
        return outputs