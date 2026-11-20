from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn

from configs.defaults import ModelConfig, NEUTRAL_QUERY_DEMAND


class SharedDescriptorEncoders(nn.Module):
    """Shared capability and task-demand encoders across UAVs (Eqs. 22, 24 & Table S7).

    Architecture (Manuscript Section 4.2):
      - Capability encoder f_cap: 2-layer Tanh MLP (cap_in_dim -> 96 -> 64)
      - Self-capability linear projection F_C: Linear(64 -> 64)
      - Demand encoder F_D: 2-layer Tanh MLP (5 -> 96 -> 64)

    For MTC-MAPPO, Morph-GAT, GAT-MAPPO, and MAPPO-morph (cap_in_dim = 11):
      Exact parameter count = (11*96 + 96 + 96*64 + 64) + (64*64 + 64) + (5*96 + 96 + 96*64 + 64)
                            = 7,360 + 4,160 + 6,784 = 18,304 -> 18,300 at Table S7 precision.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        method = cfg.method.lower()

        if method == "mappo":
            self.cap_in_dim = 5   # State x_i^t only (no capability descriptor)
        elif method == "role_mappo":
            self.cap_in_dim = 10  # 5D discrete role one-hot + 5D state x_i^t
        else:
            self.cap_in_dim = 11  # 6D capability z_i + 5D state x_i^t

        self.f_cap = nn.Sequential(
            nn.Linear(self.cap_in_dim, 96),
            nn.Tanh(),
            nn.Linear(96, cfg.embed_dim),
            nn.Tanh(),
        )
        self.f_c = nn.Linear(cfg.embed_dim, cfg.embed_dim)

        self.f_demand = nn.Sequential(
            nn.Linear(5, 96),
            nn.Tanh(),
            nn.Linear(96, cfg.embed_dim),
            nn.Tanh(),
        )

        self.register_buffer(
            "neutral_demand_vec",
            torch.tensor(NEUTRAL_QUERY_DEMAND, dtype=torch.float32),
            persistent=False,
        )

    def encode_capability(
        self,
        z_vec: torch.Tensor,
        x_state: torch.Tensor,
        role_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute c_i^{self,t} = F_C(f_cap(z_i, x_i^t)) (Eq. 22). Supports arbitrary leading dims."""
        method = self.cfg.method.lower()
        if method == "mappo":
            inp = x_state
        elif method == "role_mappo":
            if role_ids is None:
                role_oh = torch.zeros_like(x_state[..., :5])
            else:
                role_oh = torch.nn.functional.one_hot(role_ids.long() % 5, num_classes=5).to(
                    dtype=x_state.dtype
                )
            inp = torch.cat([role_oh, x_state], dim=-1)
        else:
            if method in ("gat_mappo", "tarmac_b1") or self.cfg.ablate_capability_descriptor:
                z_eff = torch.zeros_like(z_vec)
            else:
                z_eff = z_vec
            inp = torch.cat([z_eff, x_state], dim=-1)

        h = self.f_cap(inp)
        return self.f_c(h)

    def encode_demand(
        self, demand_mix: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute r_i^t = F_D(D_i^t) for Actor and r_i^{query,t} for Query (Eq. 24 & QUERY control)."""
        r_actor = self.f_demand(demand_mix)
        if self.cfg.neutral_query_override:
            neutral = self.neutral_demand_vec.expand_as(demand_mix)
            r_query = self.f_demand(neutral)
        else:
            r_query = r_actor
        return r_actor, r_query
