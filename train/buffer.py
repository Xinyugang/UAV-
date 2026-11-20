from __future__ import annotations

from typing import Any, Dict, Iterator, List
import numpy as np
import torch


class OnPolicyRolloutBuffer:
    """On-policy Rollout Buffer for CTDE-MAPPO (Algorithm S2 Steps 3-5 & Table S31).

    Stores raw decentralized Actor observations, ordered neighbor identities & masks,
    extended actions xi_i^t = (u_i^t, l_i^t, zeta_i^t), old log-densities, centralized Critic
    inputs, old value predictions, rewards, and exact GAE bootstrap/continuation indicators
    B_t and C_t (Eqs. 39-40).
    """

    def __init__(self, num_uavs: int, device: torch.device | str = "cpu") -> None:
        self.num_uavs = num_uavs
        self.device = torch.device(device)
        self.clear()

    def clear(self) -> None:
        self.actor_obs_list: List[Dict[str, np.ndarray]] = []
        self.critic_state_list: List[Dict[str, np.ndarray]] = []
        self.u_list: List[torch.Tensor] = []
        self.l_list: List[torch.Tensor] = []
        self.zeta_list: List[torch.Tensor] = []
        self.old_log_prob_list: List[torch.Tensor] = []
        self.old_values: List[float] = []
        self.next_pre_reset_values: List[float] = []
        self.rewards: List[float] = []
        self.bootstrap_flags: List[float] = []      # B_t in Eq. (39)
        self.continuation_flags: List[float] = []   # C_t in Eq. (39)

        self.advantages_step: np.ndarray = np.zeros(0, dtype=np.float32)
        self.returns_step: np.ndarray = np.zeros(0, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.rewards)

    def add_step(
        self,
        actor_obs: Dict[str, np.ndarray],
        critic_state: Dict[str, np.ndarray],
        u_sampled: torch.Tensor,
        l_sampled: torch.Tensor,
        zeta_stored: torch.Tensor,
        old_log_prob: torch.Tensor,
        old_value: float,
        next_pre_reset_value: float,
        reward: float,
        bootstrap_b: float,
        continuation_c: float,
    ) -> None:
        self.actor_obs_list.append({k: np.copy(v) for k, v in actor_obs.items()})
        self.critic_state_list.append({k: np.copy(v) for k, v in critic_state.items()})
        self.u_list.append(u_sampled.detach().cpu())
        self.l_list.append(l_sampled.detach().cpu())
        self.zeta_list.append(zeta_stored.detach().cpu())
        self.old_log_prob_list.append(old_log_prob.detach().cpu())
        self.old_values.append(float(old_value))
        self.next_pre_reset_values.append(float(next_pre_reset_value))
        self.rewards.append(float(reward))
        self.bootstrap_flags.append(float(bootstrap_b))
        self.continuation_flags.append(float(continuation_c))

    def compute_gae_and_returns(self, gamma: float = 0.99, gae_lambda: float = 0.95) -> None:
        """Compute backward GAE advantages and fixed return targets (Eqs. 39-40).

        delta_t = R_t + gamma * B_t * V_old(s^{t+1}) - V_old(s^t)
        A_hat_t = delta_t + gamma * lambda * C_t * A_hat_{t+1}
        G_hat_t = A_hat_t + V_old(s^t)
        """
        t_len = len(self.rewards)
        adv = np.zeros(t_len, dtype=np.float64)
        v_old = np.asarray(self.old_values, dtype=np.float64)
        v_next = np.asarray(self.next_pre_reset_values, dtype=np.float64)
        rews = np.asarray(self.rewards, dtype=np.float64)
        b_flags = np.asarray(self.bootstrap_flags, dtype=np.float64)
        c_flags = np.asarray(self.continuation_flags, dtype=np.float64)

        gae = 0.0
        for t in range(t_len - 1, -1, -1):
            delta_t = rews[t] + gamma * b_flags[t] * v_next[t] - v_old[t]
            gae = delta_t + gamma * gae_lambda * c_flags[t] * gae
            adv[t] = gae

        ret = adv + v_old
        self.advantages_step = adv.astype(np.float32)
        self.returns_step = ret.astype(np.float32)

    def iter_actor_minibatches(
        self,
        minibatch_size: int = 512,
        adv_norm_eps: float = 1e-8,
        rng: np.random.Generator | None = None,
    ) -> Iterator[Dict[str, Any]]:
        """Flatten (T, N, ..) -> (T*N, ..), normalize advantages with population SD, and yield batches."""
        t_len = len(self.rewards)
        if t_len == 0:
            return

        # Stack raw actor observations along time and flatten (T * N, ...)
        obs_flat: Dict[str, torch.Tensor] = {}
        keys = list(self.actor_obs_list[0].keys())
        for k in keys:
            stacked = np.concatenate([step_obs[k] for step_obs in self.actor_obs_list], axis=0)
            if stacked.dtype == bool:
                obs_flat[k] = torch.as_tensor(stacked, dtype=torch.bool, device=self.device)
            elif np.issubdtype(stacked.dtype, np.integer):
                obs_flat[k] = torch.as_tensor(stacked, dtype=torch.long, device=self.device)
            else:
                obs_flat[k] = torch.as_tensor(stacked, dtype=torch.float32, device=self.device)

        u_flat = torch.cat(self.u_list, dim=0).to(self.device)
        l_flat = torch.cat(self.l_list, dim=0).to(self.device)
        zeta_flat = torch.cat(self.zeta_list, dim=0).to(self.device)
        old_lp_flat = torch.cat(self.old_log_prob_list, dim=0).to(self.device)

        # Shared team advantage replicated to each UAV sample at step t (Manuscript p. 15)
        adv_uav = np.repeat(self.advantages_step, self.num_uavs)
        # Table S31: center and scale advantages over valid rollout entries with population SD
        adv_mean = float(np.mean(adv_uav))
        adv_pop_sd = float(np.std(adv_uav, ddof=0))
        adv_norm = (adv_uav - adv_mean) / (adv_pop_sd + adv_norm_eps)
        adv_t = torch.as_tensor(adv_norm, dtype=torch.float32, device=self.device)

        total_samples = u_flat.shape[0]
        indices = np.arange(total_samples)
        if rng is not None:
            rng.shuffle(indices)
        else:
            np.random.shuffle(indices)

        for start in range(0, total_samples, minibatch_size):
            mb_idx = torch.as_tensor(indices[start : start + minibatch_size], dtype=torch.long, device=self.device)
            mb_obs = {k: v.index_select(0, mb_idx) for k, v in obs_flat.items()}
            yield {
                "obs": mb_obs,
                "u_stored": u_flat.index_select(0, mb_idx),
                "l_stored": l_flat.index_select(0, mb_idx),
                "zeta_stored": zeta_flat.index_select(0, mb_idx),
                "old_log_prob": old_lp_flat.index_select(0, mb_idx),
                "advantages": adv_t.index_select(0, mb_idx),
            }

    def iter_critic_minibatches(
        self,
        minibatch_size: int = 512,
        rng: np.random.Generator | None = None,
    ) -> Iterator[Dict[str, Any]]:
        """Yield shuffled centralized Critic value-stream minibatches (Eq. 40, Table S31)."""
        t_len = len(self.rewards)
        if t_len == 0:
            return

        uav_feat = torch.as_tensor(
            np.stack([c["uav_features"] for c in self.critic_state_list], axis=0),
            dtype=torch.float32,
            device=self.device,
        )
        task_feat = torch.as_tensor(
            np.stack([c["task_features"] for c in self.critic_state_list], axis=0),
            dtype=torch.float32,
            device=self.device,
        )
        obs_feat = torch.as_tensor(
            np.stack([c["obs_features"] for c in self.critic_state_list], axis=0),
            dtype=torch.float32,
            device=self.device,
        )
        roles = torch.as_tensor(
            np.stack([a["uav_roles"] for a in self.actor_obs_list], axis=0),
            dtype=torch.long,
            device=self.device,
        )
        v_old_t = torch.as_tensor(self.old_values, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(self.returns_step, dtype=torch.float32, device=self.device)

        # Match critic minibatch count to step stream
        step_mb_size = max(1, minibatch_size // max(1, self.num_uavs))
        indices = np.arange(t_len)
        if rng is not None:
            rng.shuffle(indices)
        else:
            np.random.shuffle(indices)

        for start in range(0, t_len, step_mb_size):
            mb_idx = torch.as_tensor(indices[start : start + step_mb_size], dtype=torch.long, device=self.device)
            yield {
                "critic_batch": {
                    "uav_features": uav_feat.index_select(0, mb_idx),
                    "task_features": task_feat.index_select(0, mb_idx),
                    "obs_features": obs_feat.index_select(0, mb_idx),
                },
                "uav_roles": roles.index_select(0, mb_idx),
                "v_old": v_old_t.index_select(0, mb_idx),
                "returns": ret_t.index_select(0, mb_idx),
            }
