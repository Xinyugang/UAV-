from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import torch
import torch.nn as nn

from configs.defaults import EnvConfig, ModelConfig, PPOConfig, WirelessConfig
from envs.lf_uav_env import LowFidelityUAVEnv
from models.policy import MTCMAPPOAgent
from train.buffer import OnPolicyRolloutBuffer


def save_checkpoint(
    path: str | Path,
    agent: MTCMAPPOAgent,
    env_cfg: EnvConfig,
    ppo_cfg: PPOConfig,
    seed: int,
    extra_meta: Dict[str, Any] | None = None,
) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_cfg": asdict(agent.cfg),
        "env_cfg": asdict(env_cfg),
        "ppo_cfg": asdict(ppo_cfg),
        "seed": int(seed),
        "state_dict": agent.state_dict(),
        "param_breakdown": agent.parameter_breakdown(),
        "extra_meta": extra_meta or {},
    }
    torch.save(payload, out_path)
    return out_path


def load_checkpoint(
    path: str | Path, device: torch.device | str = "cpu"
) -> tuple[MTCMAPPOAgent, Dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    m_cfg = ModelConfig(**payload["model_cfg"])
    agent = MTCMAPPOAgent(m_cfg).to(device)
    agent.load_state_dict(payload["state_dict"])
    agent.eval()

    e_dict = dict(payload.get("env_cfg", {}))
    if "wireless" in e_dict and isinstance(e_dict["wireless"], dict):
        e_dict["wireless"] = WirelessConfig(**e_dict["wireless"])
    payload["env_cfg_obj"] = EnvConfig(**e_dict) if e_dict else EnvConfig()
    return agent, payload


class CTDEMAPPOTrainer:
    """Centralized Training with Decentralized Execution (CTDE) MAPPO Trainer.

    Implements Supplementary Algorithm S2 and Table S31 exactly:
      - Separate Adam optimizers (lr = 3e-4, betas = (0.9, 0.999), eps = 1e-5, weight_decay = 0)
      - Exact environment-step budget (final partial rollout truncated to hit exact budget)
      - Proper GAE B_t / C_t handling at episode resets and collection cutoffs (Eq. 39)
      - Recomputes encoders, attention, and prior during PPO epochs while holding stored
        extended action xi = (u, l, zeta) fixed (Eqs. 35-38)
      - Clipped value loss max((V - G)^2, (V_clipped - G)^2) with coef 0.5 (Table S31)
      - Separate global gradient-norm clipping at 0.5 for Actor and Critic.
    """

    def __init__(
        self,
        env_cfg: EnvConfig,
        model_cfg: ModelConfig,
        ppo_cfg: PPOConfig,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        self.env_cfg = env_cfg
        self.model_cfg = model_cfg
        self.ppo_cfg = ppo_cfg
        self.seed = int(seed)
        self.device = torch.device(device)

        torch.manual_seed(self.seed)
        np.random.seed(self.seed % (2 ** 31 - 1))
        self.rng = np.random.Generator(np.random.PCG64(self.seed))

        self.env = LowFidelityUAVEnv(cfg=self.env_cfg, training=True)
        self.agent = MTCMAPPOAgent(self.model_cfg).to(self.device)

        self.actor_optimizer = torch.optim.Adam(
            self.agent.actor_parameters(),
            lr=ppo_cfg.actor_lr,
            betas=ppo_cfg.adam_betas,
            eps=ppo_cfg.adam_eps,
            weight_decay=ppo_cfg.weight_decay,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.agent.critic_parameters(),
            lr=ppo_cfg.critic_lr,
            betas=ppo_cfg.adam_betas,
            eps=ppo_cfg.adam_eps,
            weight_decay=ppo_cfg.weight_decay,
        )
        self.buffer = OnPolicyRolloutBuffer(num_uavs=env_cfg.num_uavs, device=self.device)

    @torch.no_grad()
    def _eval_single_critic_value(
        self, critic_state: Dict[str, np.ndarray], uav_roles: np.ndarray
    ) -> float:
        c_t = {
            k: torch.as_tensor(v, dtype=torch.float32, device=self.device)
            for k, v in critic_state.items()
        }
        r_t = torch.as_tensor(uav_roles, dtype=torch.long, device=self.device)
        val = self.agent.critic(
            c_t["uav_features"], c_t["task_features"], c_t["obs_features"], uav_roles=r_t
        )
        return float(val.item())

    def _update_ppo(self) -> Dict[str, float]:
        self.agent.train()
        self.buffer.compute_gae_and_returns(
            gamma=self.ppo_cfg.gamma, gae_lambda=self.ppo_cfg.gae_lambda
        )

        policy_losses: List[float] = []
        value_losses: List[float] = []
        entropies: List[float] = []
        approx_kls: List[float] = []

        for _epoch in range(self.ppo_cfg.ppo_epochs):
            # 1. Actor + Shared Encoders + Attention updates (Eq. 38)
            for mb in self.buffer.iter_actor_minibatches(
                minibatch_size=self.ppo_cfg.minibatch_size,
                adv_norm_eps=self.ppo_cfg.adv_norm_eps,
                rng=self.rng,
            ):
                new_log_prob, entropy = self.agent.evaluate_extended_actions(
                    obs_batch=mb["obs"],
                    u_stored=mb["u_stored"],
                    l_stored=mb["l_stored"],
                    zeta_stored=mb["zeta_stored"],
                )
                log_ratio = new_log_prob - mb["old_log_prob"]
                ratio = torch.exp(torch.clamp(log_ratio, -20.0, 20.0))
                adv = mb["advantages"]

                surr1 = ratio * adv
                surr2 = torch.clamp(
                    ratio, 1.0 - self.ppo_cfg.clip_eps, 1.0 + self.ppo_cfg.clip_eps
                ) * adv
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_mean = entropy.mean()
                actor_total_loss = policy_loss - self.ppo_cfg.entropy_coef * entropy_mean

                self.actor_optimizer.zero_grad()
                actor_total_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.agent.actor_parameters(), self.ppo_cfg.max_grad_norm
                )
                self.actor_optimizer.step()

                policy_losses.append(float(policy_loss.item()))
                entropies.append(float(entropy_mean.item()))
                with torch.no_grad():
                    approx_kls.append(float(((ratio - 1.0) - log_ratio).mean().item()))

            # 2. Centralized Critic updates with clipped value objective (Eq. 40 & Table S31)
            for cmb in self.buffer.iter_critic_minibatches(
                minibatch_size=self.ppo_cfg.minibatch_size,
                rng=self.rng,
            ):
                v_pred = self.agent.evaluate_values(
                    cmb["critic_batch"], uav_roles=cmb["uav_roles"]
                )
                v_old = cmb["v_old"]
                ret = cmb["returns"]

                v_clipped = v_old + torch.clamp(
                    v_pred - v_old, -self.ppo_cfg.clip_eps, self.ppo_cfg.clip_eps
                )
                loss_unclipped = (v_pred - ret) ** 2
                loss_clipped = (v_clipped - ret) ** 2
                v_loss = self.ppo_cfg.value_loss_coef * torch.max(
                    loss_unclipped, loss_clipped
                ).mean()

                self.critic_optimizer.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.agent.critic_parameters(), self.ppo_cfg.max_grad_norm
                )
                self.critic_optimizer.step()

                value_losses.append(float(v_loss.item()))

        return {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
        }

    def train(self, max_steps_override: int | None = None) -> Dict[str, Any]:
        """Run full CTDE-MAPPO training loop up to the exact joint environment step budget."""
        budget = (
            self.ppo_cfg.total_env_steps
            if max_steps_override is None
            else int(max_steps_override)
        )
        ep_counter = 0
        actor_obs, critic_state = self.env.reset(seed=self.seed * 100_000 + ep_counter)

        collected_total = 0
        completed_ep_metrics: List[Dict[str, float]] = []
        rollout_logs: List[Dict[str, float]] = []

        while collected_total < budget:
            rollout_len = min(self.ppo_cfg.rollout_steps, budget - collected_total)
            self.buffer.clear()
            self.agent.eval()

            for step_in_roll in range(rollout_len):
                act_out = self.agent.act(actor_obs, deterministic=False, device=self.device)
                v_old = self._eval_single_critic_value(critic_state, actor_obs["uav_roles"])

                next_actor_obs, next_critic_state, reward, done, info = self.env.step(
                    act_out["motion_cmds"], act_out["slot_powers"]
                )
                is_rollout_boundary = step_in_roll == (rollout_len - 1)

                if done:
                    # True terminal state (including 1500-step horizon per Eq. 39): B_t = 0, C_t = 0
                    bootstrap_b = 0.0
                    continuation_c = 0.0
                    v_next_pre = 0.0
                elif is_rollout_boundary:
                    # Artificial collection cutoff on unfinished episode: B_t = 1, C_t = 0
                    bootstrap_b = 1.0
                    continuation_c = 0.0
                    v_next_pre = self._eval_single_critic_value(
                        next_critic_state, next_actor_obs["uav_roles"]
                    )
                else:
                    bootstrap_b = 1.0
                    continuation_c = 1.0
                    v_next_pre = self._eval_single_critic_value(
                        next_critic_state, next_actor_obs["uav_roles"]
                    )

                self.buffer.add_step(
                    actor_obs=actor_obs,
                    critic_state=critic_state,
                    u_sampled=act_out["u_sampled"],
                    l_sampled=act_out["l_sampled"],
                    zeta_stored=act_out["zeta_stored"],
                    old_log_prob=act_out["old_log_prob"],
                    old_value=v_old,
                    next_pre_reset_value=v_next_pre,
                    reward=reward,
                    bootstrap_b=bootstrap_b,
                    continuation_c=continuation_c,
                )

                collected_total += 1
                if done:
                    if "episode_metrics" in info:
                        completed_ep_metrics.append(info["episode_metrics"])
                    ep_counter += 1
                    actor_obs, critic_state = self.env.reset(
                        seed=self.seed * 100_000 + ep_counter
                    )
                else:
                    actor_obs, critic_state = next_actor_obs, next_critic_state

            upd_stats = self._update_ppo()
            upd_stats["env_steps"] = float(collected_total)
            rollout_logs.append(upd_stats)

        self.agent.eval()
        return {
            "total_env_steps": collected_total,
            "episodes_completed": len(completed_ep_metrics),
            "recent_full_success": float(
                np.mean([m["full_success"] for m in completed_ep_metrics[-20:]])
            )
            if completed_ep_metrics
            else 0.0,
            "rollout_logs": rollout_logs,
        }
