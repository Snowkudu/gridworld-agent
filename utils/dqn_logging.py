from __future__ import annotations


from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from agents.dqn import DQNAgent, OptimizationMetrics
from environment.evaluate_dqn import EvaluationResult
from training.metrics import save_json
from training.train_dqn_metrics import EpisodeMetrics, metric_as_dict




class DQNLogger:
    def __init__(
        self,
        run_dir: str | Path,
        run_config: dict,
    ):

        self.run_dir=Path("artifacts/p5_dqn") / run_dir
        self.run_dir.mkdir(parents=True,exist_ok=True)
        self.config=run_config
        self.tb_dir= self.run_dir / "tensorboard" / self.config["name"]
        self.checkpoint_path = self.run_dir / "checkpoint.pt"
        self.metric_path = self.run_dir/"metrics.json"
        self.writer=SummaryWriter(log_dir=self.tb_dir)
        self.episode_history: list[dict[str,object]] =[]


    def log_episode(self, metrics: EpisodeMetrics,optim_metrics: OptimizationMetrics | None = None,) -> None:

        loss_str = (
            f"{metrics.latest_loss:.4f}"
            if metrics.latest_loss is not None
            else "warmup"
        )

        q_str = (
            f"q={optim_metrics.predicted_q_mean:6.2f} "
            f"target={optim_metrics.bellman_target_mean:6.2f}"
            if optim_metrics is not None
            else "q=warmup target=warmup"
        )

        if metrics.success:
            print(
                f"[SUCCESS] ep={metrics.episode:03d} "
                f"return={metrics.episode_return:7.1f} "
                f"steps={metrics.steps:03d} "
                f"eps={metrics.epsilon:.3f} "
                f"{q_str}"
            )

        elif metrics.episode % 10 == 0:
            print(
                f"ep={metrics.episode:03d} "
                f"return={metrics.episode_return:7.1f} "
                f"steps={metrics.steps:03d} "
                f"success={metrics.success_rate:6.1%} "
                f"roll={metrics.rolling_success:6.1%} "
                f"avg_ret={metrics.rolling_return:7.1f} "
                f"eps={metrics.epsilon:.3f} "
                f"replay={metrics.replay_size:05d} "
                f"loss={loss_str} "
                f"{q_str}"
            )
        self.episode_history.append(metric_as_dict(metrics))
        self.writer.add_scalar("train/steps",metrics.steps,metrics.episode)
        self.writer.add_scalar("agent/epsilon",metrics.epsilon,metrics.episode)
        self.writer.add_scalars(
            "train/behavior",
        {
            "success_rate": metrics.success_rate,
            "rolling_success": metrics.rolling_success,
        },
             metrics.episode,
        )
        self.writer.add_scalars(
            "train/returns",
            {
                "episode_return": metrics.episode_return,
                "rolling_return": metrics.rolling_return,
            },
            metrics.episode,
        )
        self.writer.add_scalar(
            "train/success",
            int(metrics.success),
            metrics.episode,
        )

        self.writer.add_scalar(
            "train/timeout",
            int(metrics.timeout),
            metrics.episode,
        )


    def log_optimization(
        self,
        metrics: OptimizationMetrics,
    ) -> None:

       
        self.writer.add_scalar("optimization/td_loss",metrics.td_loss,metrics.optimization_step)
        self.writer.add_scalar("target/parameter_gap",metrics.parameter_gap,metrics.optimization_step)
        self.writer.add_scalar("target/synched",int(metrics.target_synced),metrics.optimization_step)
        self.writer.add_scalars(
            "optimization/q_values",
            {
                "predicted": metrics.predicted_q_mean,
                "bellman_target": metrics.bellman_target_mean,
                "target_next": metrics.target_next_q_mean,
            },
            metrics.optimization_step,
        )


    def log_validation(self, episode, result):
        self.writer.add_scalar(
            "validation/success_rate",
            result.success_rate,
            episode,
        )
        self.writer.add_scalar(
            "validation/two_cycle_rate",
            result.two_cycle_rate,
            episode,
        )
        self.writer.add_scalar(
            "validation/illegal_rate",
            result.illegal_rate,
            episode,
        )
        self.writer.add_scalar(
            "validation/mean_steps",
            result.mean_steps,
            episode,
        )

    def save_checkpoint(
        self,
        *,
        agent: DQNAgent,
        episode: int,
        validation: EvaluationResult,
    ) -> None:
        torch.save(
            {
                "episode": episode,
                "online_state_dict": agent.online.state_dict(),
                "target_state_dict": agent.target.state_dict(),
                "optimizer_state_dict": agent.optimizer.state_dict(),
                "epsilon": agent.epsilon,
                "validation": asdict(validation),
                "config": self.config,
            },
            self.checkpoint_path,
        )

    def build_payload(self)-> dict[str,object]:
        return {
            "config":self.config,
            "episodes": self.episode_history,            
        }


    
    def close(self) -> None:
        payload=self.build_payload()
        save_json(self.metric_path,payload)
        self.writer.flush()
        self.writer.close()


    