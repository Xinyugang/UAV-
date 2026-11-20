from __future__ import annotations

import math
from typing import Any, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn

from configs.defaults import ModelConfig
from models.attention import build_attention_module
from models.encoders import SharedDescriptorEncoders


_LOG_2PI = math.log(2.0 * math.pi)


def _diag_gaussian_log_prob(
    x: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor
) -> torch.Tensor:
    """Elementwise log N(x; mean, exp(2*log_std))."""
    var = torch.exp(2.0 * log_std)
    return -0.5 * (((x - mean) ** 2) / var + 2.0 * log_std + _LOG_2PI)


def _diag_gaussian_entropy(log_std: torch.Tensor) -> torch.Tensor:
    """Elementwise differential entropy 0.5 * (1 + log(2*pi)) + log_std."""
    return 0.5 * (1.0 + _LOG_2PI) + log_std


def project_power_budget(
    candidate_powers: torch.Tensor,
    valid_mask: torch.Tensor,
    p_max: float = 1.0,
) -> torch.Tensor:
    """Project candidate power vector onto feasible simplex-ball P_i (Manuscript p. 13, Alg. S1 Step 8).

    P_i = {P >= 0 : sum_j P_ij <= P_i^max}, with invalid-neighbor power forced to zero.
    Rescales ONLY when total valid power exceeds P_i^max.
    """
    valid_f = valid_mask.to(candidate_powers.dtype)
    p_nonneg = torch.clamp(candidate_powers, min=0.0) * valid_f
    total = p_nonneg.sum(dim=-1, keepdim=True)
    scale = torch.where(total > p_max, p_max / total.clamp_min(1e-12), torch.ones_like(total))
    return p_nonneg * scale


def _actor_hyperparams(cfg: ModelConfig) -> Tuple[int, int, int, bool]:
    """Return (h_enc, in_trunk, h_head, has_context_slot) aligned with Supplementary Table S7/S3."""
    method = cfg.method.lower()
    if method == "mtc_mappo":
        if cfg.ablate_task_attention:
            return 76, 197, 64, False
        return 72, 261, 76, True
    if method == "morph_gat":
        return 72, 261, 74, True
    if method in ("gat_mappo", "tarmac_b1"):
        return 68, 261, 68, True
    if method == "mappo_morph":
        return 76, 197, 64, False
    if method == "role_mappo":
        return 76, 261, 64, True
    if method == "mappo":
        return 79, 197, 64, False
    raise ValueError(f"Unknown method: {cfg.method}")


def _critic_hyperparams(cfg: ModelConfig) -> Tuple[int, int, int, int]:
    """Return (uav_in_dim, h_uav, d_uav, h_task) aligned with Supplementary Table S7/S3."""
    method = cfg.method.lower()
    if method == "mtc_mappo":
        return 11, 56, 64, 48   # 132,721 -> 132,700
    if method == "morph_gat":
        return 11, 52, 64, 48   # 132,417 -> 132,400
    if method in ("gat_mappo", "tarmac_b1"):
        return 11, 70, 48, 80   # 126,793 -> 126,800
    if method == "mappo_morph":
        return 11, 106, 32, 64  # 117,881 -> 117,900
    if method == "role_mappo":
        return 10, 64, 44, 48   # Role-MAPPO total -> 236,900
    if method == "mappo":
        return 5, 58, 36, 48    # MAPPO total -> 219,400
    raise ValueError(f"Unknown method: {cfg.method}")


class DecentralizedActor(nn.Module):
    """Decentralized Actor Network with 3 Diagonal-Gaussian Action Branches (Eqs. 30-32, 34-37)."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        h_enc, in_trunk, h_head, self.has_context_slot = _actor_hyperparams(cfg)

        # Role encoder specific to Role-MAPPO (discrete 5-role latent embedding)
        if cfg.method.lower() == "role_mappo":
            self.role_encoder: nn.Module | None = nn.Sequential(
                nn.Linear(5, 59),
                nn.Tanh(),
                nn.Linear(59, cfg.embed_dim),
            )
        else:
            self.role_encoder = None

        # Obstacle encoder f_obs (5 nearest obstacles x 4 features = 20D -> 64D)
        self.f_obs = nn.Sequential(
            nn.Linear(cfg.obs_cap * 4, h_enc),
            nn.Tanh(),
            nn.Linear(h_enc, cfg.embed_dim),
            nn.Tanh(),
        )
        # Full task-scene encoder f_task_obs (3 tasks x 8 features = 24D -> 64D)
        self.f_task_obs = nn.Sequential(
            nn.Linear(cfg.num_tasks * 8, h_enc),
            nn.Tanh(),
            nn.Linear(h_enc, cfg.embed_dim),
            nn.Tanh(),
        )

        # Two-layer MLP trunk with 128 hidden units (Manuscript Section 4.3, Table 3)
        self.trunk = nn.Sequential(
            nn.Linear(in_trunk, cfg.actor_hidden),
            nn.Tanh(),
            nn.Linear(cfg.actor_hidden, cfg.actor_hidden),
            nn.Tanh(),
        )

        # Three diagonal-Gaussian parameter heads (mean & clamped log-std)
        self.motion_head = nn.Sequential(
            nn.Linear(cfg.actor_hidden, h_head),
            nn.Tanh(),
            nn.Linear(h_head, 4),  # [mu_ux, mu_uy, log_std_ux, log_std_uy]
        )
        self.gate_head = nn.Sequential(
            nn.Linear(cfg.actor_hidden, h_head),
            nn.Tanh(),
            nn.Linear(h_head, 2),  # [mu_l, log_std_l]
        )
        self.alloc_head = nn.Sequential(
            nn.Linear(cfg.actor_hidden + cfg.embed_dim, h_head),
            nn.Tanh(),
            nn.Linear(h_head, 2),  # per-neighbor [mu_b,j, log_std_b,j]
        )

    def forward_distributions(
        self,
        x_self: torch.Tensor,
        c_self: torch.Tensor,
        c_nbr: torch.Tensor | None,
        r_actor: torch.Tensor,
        task_obs: torch.Tensor,
        obs_feats: torch.Tensor,
        c_nbr_self: torch.Tensor,
        nbr_mask: torch.Tensor,
        uav_roles: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute Gaussian distribution parameters for motion, gate, and per-neighbor residuals."""
        h_obs = self.f_obs(obs_feats)
        h_task = r_actor + self.f_task_obs(task_obs)

        if self.role_encoder is not None:
            if uav_roles is None:
                r_oh = torch.zeros((x_self.shape[0], 5), device=x_self.device, dtype=x_self.dtype)
            else:
                r_oh = torch.nn.functional.one_hot(uav_roles.long() % 5, num_classes=5).to(
                    dtype=x_self.dtype
                )
            ctx_feat = self.role_encoder(r_oh)
            if c_nbr is not None:
                ctx_feat = ctx_feat + c_nbr
            trunk_in = torch.cat([x_self, c_self, ctx_feat, h_task, h_obs], dim=-1)
        elif self.has_context_slot:
            assert c_nbr is not None
            trunk_in = torch.cat([x_self, c_self, c_nbr, h_task, h_obs], dim=-1)
        else:
            # Attention-free methods (MAPPO, MAPPO-morph, w/o task-conditioned attention):
            # Pool valid neighbor self-embeddings uniformly into c_self
            valid_f = nbr_mask.to(x_self.dtype)
            cnt = valid_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
            pooled_nbr = (c_nbr_self * valid_f[:, :, None]).sum(dim=1) / cnt
            has_any = nbr_mask.any(dim=-1, keepdim=True)
            c_combined = c_self + torch.where(has_any, 0.5 * pooled_nbr, torch.zeros_like(pooled_nbr))
            trunk_in = torch.cat([x_self, c_combined, h_task, h_obs], dim=-1)

        h_trunk = self.trunk(trunk_in)  # (B, 128)

        mot_raw = self.motion_head(h_trunk)
        # Scale mean smoothly to [-8, 8] m/s nominal velocity range while preserving full R^2 support
        mu_u = 8.0 * torch.tanh(mot_raw[:, :2] / 4.0)
        log_std_u = torch.clamp(mot_raw[:, 2:], self.cfg.log_std_min, self.cfg.log_std_max)

        gate_raw = self.gate_head(h_trunk)
        mu_l = gate_raw[:, 0]
        log_std_l = torch.clamp(gate_raw[:, 1], self.cfg.log_std_min, self.cfg.log_std_max)

        m_cap = c_nbr_self.shape[1]
        trunk_exp = h_trunk[:, None, :].expand(-1, m_cap, -1)
        alloc_raw = self.alloc_head(torch.cat([trunk_exp, c_nbr_self], dim=-1))  # (B, M, 2)
        mu_b = alloc_raw[:, :, 0]
        log_std_b = torch.clamp(alloc_raw[:, :, 1], self.cfg.log_std_min, self.cfg.log_std_max)

        return {
            "mu_u": mu_u,
            "log_std_u": log_std_u,
            "mu_l": mu_l,
            "log_std_l": log_std_l,
            "mu_b": mu_b,
            "log_std_b": log_std_b,
        }


class CentralizedCritic(nn.Module):
    """Centralized State-Value Critic V_psi(S_UAV^t, Z^t, D^t, O_obs^t) (Eq. 33 & Table S7).

    Uses permutation-invariant set encoders (mean+max over UAVs, mean over tasks & obstacles)
    followed by a two-layer 256-unit MLP trunk so parameter dimensions remain invariant to
    team size N and obstacle count N_obs while matching Table S7 exactly.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        uav_in_dim, h_uav, d_uav, h_task = _critic_hyperparams(cfg)
        self.uav_in_dim = uav_in_dim

        self.uav_encoder = nn.Sequential(
            nn.Linear(uav_in_dim, h_uav),
            nn.Tanh(),
            nn.Linear(h_uav, d_uav),
            nn.Tanh(),
        )
        self.task_encoder = nn.Sequential(
            nn.Linear(8, h_task),
            nn.Tanh(),
            nn.Linear(h_task, 64),
            nn.Tanh(),
        )
        self.obs_encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
        )

        d_trunk_in = 2 * d_uav + 64 + 32
        self.trunk = nn.Sequential(
            nn.Linear(d_trunk_in, cfg.critic_hidden),
            nn.Tanh(),
            nn.Linear(cfg.critic_hidden, cfg.critic_hidden),
            nn.Tanh(),
            nn.Linear(cfg.critic_hidden, 1),
        )

    def _prepare_uav_input(
        self, uav_features: torch.Tensor, uav_roles: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Select/format per-UAV global features (.., N, 11) -> (.., N, uav_in_dim)."""
        method = self.cfg.method.lower()
        x_part = uav_features[..., :5]
        z_part = uav_features[..., 5:11]
        if method == "mappo":
            return x_part
        if method == "role_mappo":
            n_uav = uav_features.shape[-2]
            if uav_roles is None:
                idx = torch.arange(n_uav, device=uav_features.device) % 5
                r_oh = torch.nn.functional.one_hot(idx, num_classes=5).to(uav_features.dtype)
                while r_oh.ndim < x_part.ndim:
                    r_oh = r_oh.unsqueeze(0)
                r_oh = r_oh.expand(*x_part.shape[:-1], 5)
            else:
                r_oh = torch.nn.functional.one_hot(uav_roles.long() % 5, num_classes=5).to(
                    uav_features.dtype
                )
            return torch.cat([r_oh, x_part], dim=-1)
        if method in ("gat_mappo", "tarmac_b1") or self.cfg.ablate_capability_descriptor:
            return torch.cat([x_part, torch.zeros_like(z_part)], dim=-1)
        return uav_features

    def forward(
        self,
        uav_features: torch.Tensor,
        task_features: torch.Tensor,
        obs_features: torch.Tensor,
        uav_roles: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate scalar state value V_psi. Accepts single state (N,..) or batch (B, N,..)."""
        single = uav_features.ndim == 2
        if single:
            uav_features = uav_features.unsqueeze(0)
            task_features = task_features.unsqueeze(0)
            obs_features = obs_features.unsqueeze(0)
            if uav_roles is not None and uav_roles.ndim == 1:
                uav_roles = uav_roles.unsqueeze(0)

        u_in = self._prepare_uav_input(uav_features, uav_roles=uav_roles)
        h_u = self.uav_encoder(u_in)  # (B, N, d_uav)
        u_mean = h_u.mean(dim=1)
        u_max = h_u.max(dim=1).values

        h_t = self.task_encoder(task_features).mean(dim=1)  # (B, 64)
        h_o = self.obs_encoder(obs_features).mean(dim=1)    # (B, 32)

        feat = torch.cat([u_mean, u_max, h_t, h_o], dim=-1)
        val = self.trunk(feat).squeeze(-1)  # (B,)
        return val.squeeze(0) if single else val


class MTCMAPPOAgent(nn.Module):
    """Complete CTDE Policy & Value Architecture for MTC-MAPPO, Baselines, and Ablations.

    Mutually exclusive module groups (Supplementary Table S7):
      - `shared_encoders`: SharedDescriptorEncoders (f_cap + F_C + F_D)
      - `attention`:       MTCTaskCapabilityAttention / MorphGAT / GAT / TarMAC / None
      - `actor`:           DecentralizedActor
      - `critic`:          CentralizedCritic
    """

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else ModelConfig()
        self.shared_encoders = SharedDescriptorEncoders(self.cfg)
        self.attention = build_attention_module(self.cfg)
        self.actor = DecentralizedActor(self.cfg)
        self.critic = CentralizedCritic(self.cfg)

    def parameter_breakdown(self) -> Dict[str, int]:
        """Return exact and Table-S7 nearest-100 rounded parameter counts across the 4 groups."""
        n_actor = sum(p.numel() for p in self.actor.parameters())
        n_critic = sum(p.numel() for p in self.critic.parameters())
        n_attn = 0 if self.attention is None else sum(p.numel() for p in self.attention.parameters())
        n_shared = sum(p.numel() for p in self.shared_encoders.parameters())
        n_total = n_actor + n_critic + n_attn + n_shared

        r100 = lambda x: int(round(x / 100.0) * 100)
        return {
            "actor_exact": n_actor,
            "critic_exact": n_critic,
            "attn_exact": n_attn,
            "shared_exact": n_shared,
            "total_exact": n_total,
            "actor_s7": r100(n_actor),
            "critic_s7": r100(n_critic),
            "attn_s7": r100(n_attn),
            "shared_s7": r100(n_shared),
            "total_s7": r100(n_actor) + r100(n_critic) + r100(n_attn) + r100(n_shared),
        }

    def actor_parameters(self) -> list[nn.Parameter]:
        """All parameters updated by the PPO Actor objective (Actor + Attention + Shared encoders)."""
        params = list(self.shared_encoders.parameters()) + list(self.actor.parameters())
        if self.attention is not None:
            params.extend(list(self.attention.parameters()))
        return params

    def critic_parameters(self) -> list[nn.Parameter]:
        """All parameters updated by the centralized Critic value objective."""
        return list(self.critic.parameters())

    def _compute_features_and_prior(
        self,
        obs_t: Dict[str, torch.Tensor],
        slot_suppress_mask: torch.Tensor | None = None,
        attention_mode: str = "normal",
        prior_uniform_mix_override: float | None = None,
    ) -> Dict[str, Any]:
        """Shared feature, attention, and prior forward pass used by both act() and PPO epochs."""
        x_self = obs_t["x_self"]          # (B, 5)
        z_self = obs_t["z_self"]          # (B, 6)
        uav_roles = obs_t.get("uav_roles")
        demand_mix = obs_t["demand_mix"]  # (B, 5)
        task_obs = obs_t["task_obs"]      # (B, 24)
        obs_feats = obs_t["obs_feats"]    # (B, 20)
        nbr_x = obs_t["nbr_x"]            # (B, M, 5)
        nbr_z = obs_t["nbr_z"]            # (B, M, 6)
        nbr_rel = obs_t["nbr_rel"]        # (B, M, 7)
        nbr_dist = obs_t["nbr_dist_cost"] # (B, M)
        nbr_hist = obs_t["nbr_hist_cost"] # (B, M)
        nbr_ener = obs_t["nbr_energy_cost"] # (B, M)
        nbr_mask = obs_t["nbr_mask"].bool() # (B, M)
        nbr_ids = obs_t.get("nbr_ids")

        bsz, m_cap, _ = nbr_x.shape
        c_self = self.shared_encoders.encode_capability(z_self, x_self, role_ids=uav_roles)

        nbr_role_ids = None
        if nbr_ids is not None:
            nbr_role_ids = torch.clamp(nbr_ids, min=0) % 5
        c_nbr_self = self.shared_encoders.encode_capability(
            nbr_z, nbr_x, role_ids=nbr_role_ids
        )
        c_nbr_self = c_nbr_self * nbr_mask[:, :, None].to(c_nbr_self.dtype)

        r_actor, r_query = self.shared_encoders.encode_demand(demand_mix)

        active_mask = nbr_mask if slot_suppress_mask is None else (nbr_mask & (~slot_suppress_mask))

        if self.attention is not None:
            attn_out = self.attention(
                r_query=r_query,
                c_self=c_self,
                c_nbr_self=c_nbr_self,
                nbr_rel=nbr_rel,
                nbr_dist_cost=nbr_dist,
                nbr_hist_cost=nbr_hist,
                nbr_energy_cost=nbr_ener,
                nbr_mask=nbr_mask,
                slot_suppress_mask=slot_suppress_mask,
                attention_mode=attention_mode,
                prior_uniform_mix_override=prior_uniform_mix_override,
            )
            c_nbr = attn_out["c_nbr"]
            pi_prior = attn_out["pi_prior"]
            omega_bar = attn_out["omega_bar"]
        else:
            c_nbr = None
            active_f = active_mask.to(x_self.dtype)
            cnt = active_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
            pi_prior = torch.where(
                active_mask.any(dim=-1, keepdim=True),
                active_f / cnt,
                torch.zeros_like(active_f),
            )
            omega_bar = pi_prior

        # Eq. (35): log-prior offset a_theta^prior(o, M) = log(pi^prior + eps_log) on active slots
        a_prior = torch.where(
            active_mask,
            torch.log(pi_prior + self.cfg.eps_log),
            torch.zeros_like(pi_prior),
        )

        dist_params = self.actor.forward_distributions(
            x_self=x_self,
            c_self=c_self,
            c_nbr=c_nbr,
            r_actor=r_actor,
            task_obs=task_obs,
            obs_feats=obs_feats,
            c_nbr_self=c_nbr_self,
            nbr_mask=active_mask,
            uav_roles=uav_roles,
        )
        return {
            "active_mask": active_mask,
            "pi_prior": pi_prior,
            "a_prior": a_prior,
            "omega_bar": omega_bar,
            "dist_params": dist_params,
        }

    @staticmethod
    def obs_dict_to_tensors(
        obs: Dict[str, Any], device: torch.device | str = "cpu"
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for k, v in obs.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(device=device)
            else:
                arr = np.asarray(v)
                if arr.dtype == bool:
                    out[k] = torch.as_tensor(arr, dtype=torch.bool, device=device)
                elif np.issubdtype(arr.dtype, np.integer):
                    out[k] = torch.as_tensor(arr, dtype=torch.long, device=device)
                else:
                    out[k] = torch.as_tensor(arr, dtype=torch.float32, device=device)
        return out

    @torch.no_grad()
    def act(
        self,
        obs: Dict[str, Any],
        deterministic: bool = False,
        slot_suppress_mask: torch.Tensor | np.ndarray | None = None,
        attention_mode: str = "normal",
        prior_uniform_mix_override: float | None = None,
        device: torch.device | str = "cpu",
    ) -> Dict[str, Any]:
        """Execute Algorithm S1 action generation for a team/batch of UAVs.

        Returns dictionary with:
          - 'motion_cmds': np.ndarray (B, 2), raw high-level motion commands u_i^t
          - 'slot_powers': np.ndarray (B, M_cap), budget-projected transmit powers P_ij^t
          - 'u_sampled':   torch.Tensor (B, 2)
          - 'l_sampled':   torch.Tensor (B,)
          - 'b_sampled':   torch.Tensor (B, M_cap)
          - 'zeta_stored': torch.Tensor (B, M_cap), extended-action allocation coord a_prior + b
          - 'old_log_prob': torch.Tensor (B,), log p_theta(xi | o, M) per Eq. (36)
          - 'omega_bar':   np.ndarray (B, M_cap), head-averaged attention priority
          - 'pi_prior':    np.ndarray (B, M_cap), smoothed communication prior
          - 'gate_mu':     np.ndarray (B,), sigmoid gate mu_i^t (0 where candidate set empty)
        """
        obs_t = self.obs_dict_to_tensors(obs, device=device)
        sup_t = None
        if slot_suppress_mask is not None:
            sup_t = torch.as_tensor(slot_suppress_mask, dtype=torch.bool, device=device)

        fwd = self._compute_features_and_prior(
            obs_t,
            slot_suppress_mask=sup_t,
            attention_mode=attention_mode,
            prior_uniform_mix_override=prior_uniform_mix_override,
        )
        active_mask: torch.Tensor = fwd["active_mask"]  # (B, M)
        a_prior: torch.Tensor = fwd["a_prior"]          # (B, M)
        dp = fwd["dist_params"]

        mu_u, log_std_u = dp["mu_u"], dp["log_std_u"]
        mu_l, log_std_l = dp["mu_l"], dp["log_std_l"]
        mu_b, log_std_b = dp["mu_b"], dp["log_std_b"]

        if deterministic:
            u_act = mu_u
            l_act = mu_l
            b_act = mu_b
        else:
            u_act = mu_u + torch.exp(log_std_u) * torch.randn_like(mu_u)
            l_act = mu_l + torch.exp(log_std_l) * torch.randn_like(mu_l)
            b_act = mu_b + torch.exp(log_std_b) * torch.randn_like(mu_b)

        has_any = active_mask.any(dim=-1)  # (B,)
        active_f = active_mask.to(u_act.dtype)

        # Zero out absent communication coordinates on empty candidate sets & invalid slots
        l_stored = torch.where(has_any, l_act, torch.zeros_like(l_act))
        b_valid = b_act * active_f
        zeta_stored = (a_prior + b_valid) * active_f

        # Eqs. (30)-(31): Sigmoid gate, allocation softmax over active valid slots, & projection
        gate_val = torch.where(has_any, torch.sigmoid(l_stored), torch.zeros_like(l_stored))
        zeta_masked = zeta_stored.masked_fill(~active_mask, -1e9)
        alloc_shares = torch.softmax(zeta_masked, dim=-1) * active_f
        alloc_shares = torch.where(has_any[:, None], alloc_shares, torch.zeros_like(alloc_shares))

        cand_powers = gate_val[:, None] * self.cfg.p_max * alloc_shares
        exec_powers = project_power_budget(cand_powers, active_mask, p_max=self.cfg.p_max)

        # Eq. (36) & Alg. S1 Step 3/9: Log-density over valid extended-action coordinates ONLY
        lp_motion = _diag_gaussian_log_prob(u_act, mu_u, log_std_u).sum(dim=-1)
        lp_gate = torch.where(
            has_any,
            _diag_gaussian_log_prob(l_stored, mu_l, log_std_l),
            torch.zeros_like(lp_motion),
        )
        lp_alloc = (_diag_gaussian_log_prob(b_valid, mu_b, log_std_b) * active_f).sum(dim=-1)
        old_log_prob = lp_motion + lp_gate + lp_alloc

        return {
            "motion_cmds": u_act.cpu().numpy(),
            "slot_powers": exec_powers.cpu().numpy(),
            "u_sampled": u_act.detach().cpu(),
            "l_sampled": l_stored.detach().cpu(),
            "b_sampled": b_valid.detach().cpu(),
            "zeta_stored": zeta_stored.detach().cpu(),
            "old_log_prob": old_log_prob.detach().cpu(),
            "omega_bar": fwd["omega_bar"].detach().cpu().numpy(),
            "pi_prior": fwd["pi_prior"].detach().cpu().numpy(),
            "gate_mu": gate_val.detach().cpu().numpy(),
        }

    def evaluate_extended_actions(
        self,
        obs_batch: Dict[str, torch.Tensor],
        u_stored: torch.Tensor,
        l_stored: torch.Tensor,
        zeta_stored: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recompute current density log p_theta(xi_stored | o, M) and entropy (Eqs. 35-37, Alg. S2).

        Crucially:
          - Recomputes current features, attention, and log-prior a_theta^prior(o, M) from raw obs.
          - Holds stored extended coordinate zeta_stored FIXED and recovers implied residual:
                b_theta = zeta_stored - a_theta^prior(o, M)
            so policy gradients backpropagate through BOTH Actor heads AND the attention prior.
          - Empty neighbor sets evaluate ONLY motion-branch log-density and entropy.
          - Invalid padded slots contribute neither log-probability nor entropy.
        """
        fwd = self._compute_features_and_prior(obs_batch)
        active_mask: torch.Tensor = fwd["active_mask"]
        a_prior: torch.Tensor = fwd["a_prior"]
        dp = fwd["dist_params"]

        mu_u, log_std_u = dp["mu_u"], dp["log_std_u"]
        mu_l, log_std_l = dp["mu_l"], dp["log_std_l"]
        mu_b, log_std_b = dp["mu_b"], dp["log_std_b"]

        has_any = active_mask.any(dim=-1)
        active_f = active_mask.to(u_stored.dtype)

        # Implied residual under current theta (Eq. 36)
        b_theta = (zeta_stored - a_prior) * active_f

        lp_motion = _diag_gaussian_log_prob(u_stored, mu_u, log_std_u).sum(dim=-1)
        lp_gate = torch.where(
            has_any,
            _diag_gaussian_log_prob(l_stored, mu_l, log_std_l),
            torch.zeros_like(lp_motion),
        )
        lp_alloc = (_diag_gaussian_log_prob(b_theta, mu_b, log_std_b) * active_f).sum(dim=-1)
        log_prob = lp_motion + lp_gate + lp_alloc

        ent_motion = _diag_gaussian_entropy(log_std_u).sum(dim=-1)
        ent_gate = torch.where(
            has_any,
            _diag_gaussian_entropy(log_std_l),
            torch.zeros_like(ent_motion),
        )
        ent_alloc = (_diag_gaussian_entropy(log_std_b) * active_f).sum(dim=-1)
        entropy = ent_motion + ent_gate + ent_alloc
        return log_prob, entropy

    def evaluate_values(
        self,
        critic_batch: Dict[str, torch.Tensor],
        uav_roles: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate centralized Critic state value V_psi (Eq. 33)."""
        return self.critic(
            uav_features=critic_batch["uav_features"],
            task_features=critic_batch["task_features"],
            obs_features=critic_batch["obs_features"],
            uav_roles=uav_roles,
        )


# ============================================================================
# Supplementary Table S32: Comparator Controls B2 and B3
# ============================================================================
def _project_simplex_l1_ball_np(v: np.ndarray, budget: float = 1.0) -> np.ndarray:
    """Euclidean projection onto {p >= 0 : sum(p) <= budget}."""
    p_nonneg = np.maximum(0.0, np.asarray(v, dtype=np.float64))
    if float(np.sum(p_nonneg)) <= budget:
        return p_nonneg
    # Duchi et al. (2008) projection onto simplex of sum = budget
    u = np.sort(p_nonneg)[::-1]
    cssv = np.cumsum(u) - budget
    ind = np.arange(1, u.size + 1)
    cond = u - cssv / ind > 0.0
    if not np.any(cond):
        return np.full_like(p_nonneg, budget / max(1, p_nonneg.size))
    rho = int(ind[cond][-1])
    theta = float(cssv[rho - 1] / rho)
    return np.maximum(0.0, p_nonneg - theta)


class B2LocalPowerHeuristic:
    """Comparator B2: Local Projected-Gradient Power Heuristic with Fixed MTC Motion (Table S32).

    Maximizes sum_{j in N_i^t} log(1 + SINR_hat_{ij}) over P_j >= 0, sum_j P_j <= 1 W using
    historical channel gains and other senders' previous-frame total transmit powers:
      - Up to 20 projected-gradient steps of step size 0.02
      - Early stop when objective gain < 1e-6
      - Uniform-power fallback on nonfinite objective/gradient or failure; zero when M = 0.
    """

    def __init__(
        self,
        frozen_mtc_agent: MTCMAPPOAgent,
        p_max: float = 1.0,
        step_size: float = 0.02,
        max_iters: int = 20,
        tol: float = 1e-6,
        noise_power_w: float = 4e-15,
    ) -> None:
        self.agent = frozen_mtc_agent
        self.p_max = p_max
        self.step_size = step_size
        self.max_iters = max_iters
        self.tol = tol
        self.noise_power_w = noise_power_w

    def act(
        self,
        obs: Dict[str, Any],
        prev_channel_gains: np.ndarray,
        prev_tx_powers: np.ndarray,
        device: torch.device | str = "cpu",
    ) -> Tuple[np.ndarray, np.ndarray]:
        base_out = self.agent.act(obs, deterministic=True, device=device)
        motion_cmds = base_out["motion_cmds"]

        nbr_ids = np.asarray(obs["nbr_ids"], dtype=np.int64)
        nbr_mask = np.asarray(obs["nbr_mask"], dtype=bool)
        n_uav, m_cap = nbr_ids.shape
        slot_powers = np.zeros((n_uav, m_cap), dtype=np.float64)

        p_other_tot = np.sum(prev_tx_powers, axis=1)  # (N,)
        for i in range(n_uav):
            valid_slots = np.where(nbr_mask[i])[0]
            m_valid = valid_slots.size
            if m_valid == 0:
                continue

            uids = nbr_ids[i, valid_slots]
            # Effective historical channel-to-interference+noise ratio alpha_j = G_{ij}^{t-1} / (N0B + I_{ij}^{t-1})
            alphas = np.zeros(m_valid, dtype=np.float64)
            for idx_m, j_uid in enumerate(uids):
                g_ij = max(float(prev_channel_gains[i, j_uid]), 1e-15)
                i_other = 0.0
                for ell in range(n_uav):
                    if ell != i and ell != j_uid:
                        i_other += float(p_other_tot[ell]) * float(prev_channel_gains[ell, j_uid])
                alphas[idx_m] = g_ij / max(self.noise_power_w + i_other, 1e-18)

            p_vec = np.full(m_valid, self.p_max / m_valid, dtype=np.float64)
            if not np.all(np.isfinite(alphas)):
                slot_powers[i, valid_slots] = p_vec
                continue

            prev_obj = float(np.sum(np.log1p(alphas * p_vec)))
            failed = not np.isfinite(prev_obj)
            for _it in range(self.max_iters):
                if failed:
                    break
                grad = alphas / (1.0 + alphas * p_vec)
                if not np.all(np.isfinite(grad)):
                    failed = True
                    break
                p_next = _project_simplex_l1_ball_np(p_vec + self.step_size * grad, self.p_max)
                if not np.all(np.isfinite(p_next)):
                    failed = True
                    break
                new_obj = float(np.sum(np.log1p(alphas * p_next)))
                if not np.isfinite(new_obj):
                    failed = True
                    break
                gain = new_obj - prev_obj
                p_vec = p_next
                prev_obj = new_obj
                if abs(gain) < self.tol:
                    break

            if failed:
                p_vec = np.full(m_valid, self.p_max / m_valid, dtype=np.float64)
            slot_powers[i, valid_slots] = p_vec

        return motion_cmds, slot_powers


class B3VelocitySafetyFilter:
    """Comparator B3: Nearest-Feasible One-Step Velocity Safety Filter on MTC-MAPPO (Table S32).

    - Prediction horizon dt = 0.2 s
    - Inter-UAV center clearance >= 0.6 m; obstacle-center clearance >= 0.7 m
    - Enforces platform speed v_i^max and acceleration a_i^max limits
    - If infeasible, targets zero velocity with acceleration-limited deceleration
    - Transmit powers remain identical to MTC-MAPPO.
    """

    def __init__(
        self,
        frozen_mtc_agent: MTCMAPPOAgent,
        dt: float = 0.2,
        uav_margin_m: float = 0.6,
        obs_margin_m: float = 0.7,
    ) -> None:
        self.agent = frozen_mtc_agent
        self.dt = dt
        self.uav_margin_m = uav_margin_m
        self.obs_margin_m = obs_margin_m

    def act(
        self,
        obs: Dict[str, Any],
        uav_pos: np.ndarray,
        uav_vel: np.ndarray,
        uav_caps: np.ndarray,
        obstacle_centers: np.ndarray,
        device: torch.device | str = "cpu",
    ) -> Tuple[np.ndarray, np.ndarray]:
        base_out = self.agent.act(obs, deterministic=True, device=device)
        raw_cmds = np.asarray(base_out["motion_cmds"], dtype=np.float64)
        slot_powers = np.asarray(base_out["slot_powers"], dtype=np.float64)

        n_uav = uav_pos.shape[0]
        filtered_cmds = np.zeros_like(raw_cmds)
        pred_other_pos = uav_pos + self.dt * uav_vel

        # Precompute polar search grid around desired velocity within acceleration disk
        angles = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
        radii_frac = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)

        for i in range(n_uav):
            v_curr = uav_vel[i]
            v_max_i = float(uav_caps[i, 2])
            a_max_i = float(uav_caps[i, 3])
            max_dv = a_max_i * self.dt

            # Project nominal command into speed & acceleration ball first
            u_des = raw_cmds[i].copy()
            un = float(np.linalg.norm(u_des))
            if un > v_max_i:
                u_des = u_des * (v_max_i / un)
            dv = u_des - v_curr
            dvn = float(np.linalg.norm(dv))
            if dvn > max_dv:
                u_des = v_curr + dv * (max_dv / dvn)

            def _is_safe(v_cand: np.ndarray) -> bool:
                if float(np.linalg.norm(v_cand)) > v_max_i + 1e-9:
                    return False
                if float(np.linalg.norm(v_cand - v_curr)) > max_dv + 1e-9:
                    return False
                p_pred = uav_pos[i] + self.dt * v_cand
                if np.any(p_pred < 0.0) or np.any(p_pred > 100.0):
                    return False
                for j in range(n_uav):
                    if j != i:
                        if float(np.linalg.norm(p_pred - pred_other_pos[j])) < self.uav_margin_m:
                            return False
                if obstacle_centers.shape[0] > 0:
                    d_obs = np.linalg.norm(obstacle_centers - p_pred[None, :], axis=-1)
                    if float(np.min(d_obs)) < self.obs_margin_m:
                        return False
                return True

            if _is_safe(u_des):
                filtered_cmds[i] = u_des
                continue

            # Search nearest feasible velocity in reachable acceleration disk around v_curr
            best_v: np.ndarray | None = None
            best_dist_sq = float("inf")
            for r_f in radii_frac:
                r = r_f * max_dv
                if r == 0.0:
                    cands = [v_curr]
                else:
                    cands = [
                        v_curr + np.array([r * np.cos(ang), r * np.sin(ang)]) for ang in angles
                    ]
                for vc in cands:
                    vn = float(np.linalg.norm(vc))
                    if vn > v_max_i:
                        vc = vc * (v_max_i / vn)
                    if _is_safe(vc):
                        dsq = float(np.sum((vc - u_des) ** 2))
                        if dsq < best_dist_sq:
                            best_dist_sq = dsq
                            best_v = vc

            if best_v is not None:
                filtered_cmds[i] = best_v
            else:
                # Table S32 fallback: target zero velocity with acceleration-limited deceleration
                vc_norm = float(np.linalg.norm(v_curr))
                if vc_norm <= max_dv:
                    filtered_cmds[i] = np.zeros(2, dtype=np.float64)
                else:
                    filtered_cmds[i] = v_curr * (1.0 - max_dv / vc_norm)

        return filtered_cmds, slot_powers
