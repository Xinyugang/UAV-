from __future__ import annotations

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.defaults import ModelConfig


def _apply_slot_interventions_and_prior(
    omega_heads: torch.Tensor,
    v_heads: torch.Tensor,
    nbr_mask: torch.Tensor,
    w_o: nn.Linear,
    context_ffn: nn.Module,
    eps_s: float,
    uniform_prior: bool = False,
    prior_uniform_mix: float = 0.0,
    slot_suppress_mask: torch.Tensor | None = None,
    attention_mode: str = "normal",
    rand_gen: torch.Generator | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Aggregate multi-head neighbor contexts (Eqs. 27-28) and form smoothed prior (Eq. 29).

    Args:
        omega_heads: shape (B, H, M) normalized per-head attention weights
        v_heads: shape (B, M, H, d_h) per-head neighbor values
        nbr_mask: shape (B, M) bool mask of valid candidates
        slot_suppress_mask: optional shape (B, M) bool mask of intervened/masked slots
            whose context and directed transmit power are suppressed (Fig. 5, S45-S46).
        attention_mode: 'normal' or 'randomize' (for Fig. 5 'Randomize attention' control).

    Returns:
        c_nbr: shape (B, d_emb) aggregated neighbor context (exact 0 where active_mask is empty)
        pi_prior: shape (B, M) smoothed communication prior over active valid neighbors
        omega_bar: shape (B, M) head-averaged attention weights
    """
    bsz, n_heads, m_cap = omega_heads.shape

    if attention_mode == "randomize":
        rand_logits = torch.randn(
            omega_heads.shape, device=omega_heads.device, dtype=omega_heads.dtype, generator=rand_gen
        )
        rand_logits = rand_logits.masked_fill(~nbr_mask[:, None, :], -1e9)
        omega_heads = torch.softmax(rand_logits, dim=-1)
        omega_heads = omega_heads * nbr_mask[:, None, :].to(omega_heads.dtype)

    if slot_suppress_mask is not None:
        active_mask = nbr_mask & (~slot_suppress_mask)
        # Renormalize remaining candidate weights per head
        omega_masked = omega_heads * active_mask[:, None, :].to(omega_heads.dtype)
        denom_h = omega_masked.sum(dim=-1, keepdim=True)
        omega_eff = torch.where(
            denom_h > 1e-12,
            omega_masked / denom_h.clamp_min(1e-12),
            torch.zeros_like(omega_masked),
        )
    else:
        active_mask = nbr_mask
        omega_eff = omega_heads * active_mask[:, None, :].to(omega_heads.dtype)

    has_any = active_mask.any(dim=-1)  # (B,)

    # Eq. (27): c_{i,h}^{nbr,t} = sum_j omega_{ij,h}^t * V_{j,h}^t
    # omega_eff: (B, H, M), v_heads: (B, M, H, d_h) -> permute v to (B, H, M, d_h)
    v_perm = v_heads.permute(0, 2, 1, 3)
    c_heads = torch.matmul(omega_eff.unsqueeze(2), v_perm).squeeze(2)  # (B, H, d_h)
    c_cat = c_heads.reshape(bsz, -1)  # (B, H * d_h)
    c_proj = w_o(c_cat)
    c_nbr = c_proj + context_ffn(c_proj)
    c_nbr = torch.where(has_any[:, None], c_nbr, torch.zeros_like(c_nbr))

    # Eq. (28): head-averaged priority bar_omega_{ij}^t = H^{-1} sum_h omega_{ij,h}^t
    omega_bar = omega_eff.mean(dim=1)  # (B, M)
    omega_bar = torch.where(has_any[:, None], omega_bar, torch.zeros_like(omega_bar))

    # Eq. (29): smoothed communication prior over active valid neighbors
    active_f = active_mask.to(omega_heads.dtype)
    valid_counts = active_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
    if uniform_prior:
        pi_prior = active_f / valid_counts
    else:
        smoothed = (omega_bar + eps_s) * active_f
        pi_prior = smoothed / smoothed.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    if prior_uniform_mix > 0.0:
        mix = float(max(0.0, min(1.0, prior_uniform_mix)))
        unif = active_f / valid_counts
        pi_prior = (1.0 - mix) * pi_prior + mix * unif

    pi_prior = torch.where(has_any[:, None], pi_prior, torch.zeros_like(pi_prior))
    return c_nbr, pi_prior, omega_bar


class MTCTaskCapabilityAttention(nn.Module):
    """Multi-head Task-Demand Query vs. Platform-Capability Key Attention (Eqs. 24-29).

    Parameter count = 3*(64*128+128) + (128*64+64) + (7*72+72 + 72*128+128) + (64*138+138 + 138*64+64)
                    = 24,960 + 8,256 + 9,848 + 17,938 = 61,002 -> 61,000 (Table S7).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.num_heads  # 2
        self.d_head = cfg.embed_dim   # 64 per head -> 128 concat

        self.w_q = nn.Linear(cfg.embed_dim, self.n_heads * self.d_head)
        self.w_k = nn.Linear(cfg.embed_dim, self.n_heads * self.d_head)
        self.w_v = nn.Linear(cfg.embed_dim, self.n_heads * self.d_head)
        self.rel_encoder = nn.Sequential(
            nn.Linear(7, 72),
            nn.Tanh(),
            nn.Linear(72, self.n_heads * self.d_head),
        )
        self.w_o = nn.Linear(self.n_heads * self.d_head, cfg.embed_dim)
        self.context_ffn = nn.Sequential(
            nn.Linear(cfg.embed_dim, 138),
            nn.Tanh(),
            nn.Linear(138, cfg.embed_dim),
        )

    def forward(
        self,
        r_query: torch.Tensor,
        c_self: torch.Tensor,
        c_nbr_self: torch.Tensor,
        nbr_rel: torch.Tensor,
        nbr_dist_cost: torch.Tensor,
        nbr_hist_cost: torch.Tensor,
        nbr_energy_cost: torch.Tensor,
        nbr_mask: torch.Tensor,
        slot_suppress_mask: torch.Tensor | None = None,
        attention_mode: str = "normal",
        prior_uniform_mix_override: float | None = None,
    ) -> Dict[str, torch.Tensor]:
        bsz, m_cap, _ = c_nbr_self.shape
        has_any = nbr_mask.any(dim=-1)

        # Fast-path Eq. (29) / Alg. S1 Step 3 when all batch items have empty candidate sets
        if not bool(has_any.any()):
            zeros_c = torch.zeros((bsz, self.cfg.embed_dim), device=r_query.device, dtype=r_query.dtype)
            zeros_m = torch.zeros((bsz, m_cap), device=r_query.device, dtype=r_query.dtype)
            return {"c_nbr": zeros_c, "pi_prior": zeros_m, "omega_bar": zeros_m, "scores": zeros_m}

        # Eqs. (24)-(25): Head-specific Q, K, V with L2 normalization on Q and K
        q = self.w_q(r_query).view(bsz, self.n_heads, self.d_head)
        k = self.w_k(c_nbr_self).view(bsz, m_cap, self.n_heads, self.d_head)
        v = (self.w_v(c_nbr_self) + self.rel_encoder(nbr_rel)).view(
            bsz, m_cap, self.n_heads, self.d_head
        )

        q_norm = F.normalize(q, p=2.0, dim=-1, eps=1e-8)
        k_norm = F.normalize(k, p=2.0, dim=-1, eps=1e-8)

        # Cosine similarity divided by temperature tau_a: (B, H, M)
        cos_sim = torch.einsum("bhd,bmhd->bhm", q_norm, k_norm) / max(self.cfg.tau_a, 1e-6)

        # Eq. (26): Subtract distance, link-history, and neighbor energy-deficiency costs
        if self.cfg.ablate_link_energy_mod:
            penalties = torch.zeros_like(nbr_dist_cost)
        else:
            penalties = (
                self.cfg.alpha_d * nbr_dist_cost
                + self.cfg.alpha_c * nbr_hist_cost
                + self.cfg.alpha_e * nbr_energy_cost
            )
        scores = cos_sim - penalties[:, None, :]  # (B, H, M)

        # Mask invalid padded slots before softmax
        masked_scores = scores.masked_fill(~nbr_mask[:, None, :], -1e9)
        omega_heads = torch.softmax(masked_scores, dim=-1)
        omega_heads = omega_heads * nbr_mask[:, None, :].to(scores.dtype)

        mix = (
            self.cfg.prior_uniform_mix
            if prior_uniform_mix_override is None
            else float(prior_uniform_mix_override)
        )
        c_nbr, pi_prior, omega_bar = _apply_slot_interventions_and_prior(
            omega_heads=omega_heads,
            v_heads=v,
            nbr_mask=nbr_mask,
            w_o=self.w_o,
            context_ffn=self.context_ffn,
            eps_s=self.cfg.eps_s,
            uniform_prior=self.cfg.ablate_comm_prior,
            prior_uniform_mix=mix,
            slot_suppress_mask=slot_suppress_mask,
            attention_mode=attention_mode,
        )
        return {
            "c_nbr": c_nbr,
            "pi_prior": pi_prior,
            "omega_bar": omega_bar,
            "scores": scores.mean(dim=1),
        }


class MorphGATAttention(nn.Module):
    """Parameter-matched Morph-GAT Attention Control (Section 4.2 & Supplementary Table S7).

    Replaces Query-Key task-capability cosine matching with generic 2-head GAT scoring
    e_ij^GAT = LeakyReLU(a^T [W h_i || W h_j]) while retaining identical descriptor inputs,
    distance/history/energy feasibility modulation, and prior-guided power interface.
    Exact parameter count = 57,324 -> 57,300 (Table S7).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.num_heads  # 2
        self.d_head = cfg.embed_dim   # 64

        self.w_node = nn.Linear(cfg.embed_dim, self.n_heads * self.d_head)
        self.attn_vec = nn.Parameter(torch.empty(self.n_heads, 2 * self.d_head))
        nn.init.xavier_uniform_(self.attn_vec.unsqueeze(0))

        self.w_v = nn.Linear(cfg.embed_dim, self.n_heads * self.d_head)
        self.rel_encoder = nn.Sequential(
            nn.Linear(7, 72),
            nn.Tanh(),
            nn.Linear(72, self.n_heads * self.d_head),
        )
        self.w_o = nn.Linear(self.n_heads * self.d_head, cfg.embed_dim)
        self.context_ffn = nn.Sequential(
            nn.Linear(cfg.embed_dim, 172),
            nn.Tanh(),
            nn.Linear(172, cfg.embed_dim),
        )
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(
        self,
        r_query: torch.Tensor,
        c_self: torch.Tensor,
        c_nbr_self: torch.Tensor,
        nbr_rel: torch.Tensor,
        nbr_dist_cost: torch.Tensor,
        nbr_hist_cost: torch.Tensor,
        nbr_energy_cost: torch.Tensor,
        nbr_mask: torch.Tensor,
        slot_suppress_mask: torch.Tensor | None = None,
        attention_mode: str = "normal",
        prior_uniform_mix_override: float | None = None,
    ) -> Dict[str, torch.Tensor]:
        bsz, m_cap, _ = c_nbr_self.shape
        has_any = nbr_mask.any(dim=-1)
        if not bool(has_any.any()):
            zeros_c = torch.zeros((bsz, self.cfg.embed_dim), device=c_self.device, dtype=c_self.dtype)
            zeros_m = torch.zeros((bsz, m_cap), device=c_self.device, dtype=c_self.dtype)
            return {"c_nbr": zeros_c, "pi_prior": zeros_m, "omega_bar": zeros_m, "scores": zeros_m}

        h_i = self.w_node(c_self + 0.5 * r_query).view(bsz, 1, self.n_heads, self.d_head)
        h_j = self.w_node(c_nbr_self).view(bsz, m_cap, self.n_heads, self.d_head)
        h_cat = torch.cat([h_i.expand(-1, m_cap, -1, -1), h_j], dim=-1)  # (B, M, H, 2*d_h)
        gat_raw = self.leaky_relu(torch.einsum("bmhd,hd->bhm", h_cat, self.attn_vec))

        penalties = (
            self.cfg.alpha_d * nbr_dist_cost
            + self.cfg.alpha_c * nbr_hist_cost
            + self.cfg.alpha_e * nbr_energy_cost
        )
        scores = gat_raw - penalties[:, None, :]
        masked_scores = scores.masked_fill(~nbr_mask[:, None, :], -1e9)
        omega_heads = torch.softmax(masked_scores, dim=-1) * nbr_mask[:, None, :].to(scores.dtype)

        v = (self.w_v(c_nbr_self) + self.rel_encoder(nbr_rel)).view(
            bsz, m_cap, self.n_heads, self.d_head
        )
        mix = (
            self.cfg.prior_uniform_mix
            if prior_uniform_mix_override is None
            else float(prior_uniform_mix_override)
        )
        c_nbr, pi_prior, omega_bar = _apply_slot_interventions_and_prior(
            omega_heads=omega_heads,
            v_heads=v,
            nbr_mask=nbr_mask,
            w_o=self.w_o,
            context_ffn=self.context_ffn,
            eps_s=self.cfg.eps_s,
            uniform_prior=False,
            prior_uniform_mix=mix,
            slot_suppress_mask=slot_suppress_mask,
            attention_mode=attention_mode,
        )
        return {
            "c_nbr": c_nbr,
            "pi_prior": pi_prior,
            "omega_bar": omega_bar,
            "scores": scores.mean(dim=1),
        }


class GATAttention(nn.Module):
    """Standard Generic Graph Attention Baseline without Descriptors (GAT-MAPPO, Table S7).

    Exact parameter count = 47,885 -> 47,900 (Table S7).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.num_heads  # 2
        self.d_head = cfg.embed_dim   # 64

        self.w_node = nn.Linear(cfg.embed_dim, self.n_heads * self.d_head)
        self.attn_vec = nn.Parameter(torch.empty(self.n_heads, 2 * self.d_head))
        nn.init.xavier_uniform_(self.attn_vec.unsqueeze(0))

        self.w_v = nn.Linear(cfg.embed_dim, self.n_heads * self.d_head)
        self.rel_encoder = nn.Sequential(
            nn.Linear(7, 32),
            nn.Tanh(),
            nn.Linear(32, self.n_heads * self.d_head),
        )
        self.w_o = nn.Linear(self.n_heads * self.d_head, cfg.embed_dim)
        self.context_ffn = nn.Sequential(
            nn.Linear(cfg.embed_dim, 141),
            nn.Tanh(),
            nn.Linear(141, cfg.embed_dim),
        )
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(
        self,
        r_query: torch.Tensor,
        c_self: torch.Tensor,
        c_nbr_self: torch.Tensor,
        nbr_rel: torch.Tensor,
        nbr_dist_cost: torch.Tensor,
        nbr_hist_cost: torch.Tensor,
        nbr_energy_cost: torch.Tensor,
        nbr_mask: torch.Tensor,
        slot_suppress_mask: torch.Tensor | None = None,
        attention_mode: str = "normal",
        prior_uniform_mix_override: float | None = None,
    ) -> Dict[str, torch.Tensor]:
        bsz, m_cap, _ = c_nbr_self.shape
        has_any = nbr_mask.any(dim=-1)
        if not bool(has_any.any()):
            zeros_c = torch.zeros((bsz, self.cfg.embed_dim), device=c_self.device, dtype=c_self.dtype)
            zeros_m = torch.zeros((bsz, m_cap), device=c_self.device, dtype=c_self.dtype)
            return {"c_nbr": zeros_c, "pi_prior": zeros_m, "omega_bar": zeros_m, "scores": zeros_m}

        h_i = self.w_node(c_self).view(bsz, 1, self.n_heads, self.d_head)
        h_j = self.w_node(c_nbr_self).view(bsz, m_cap, self.n_heads, self.d_head)
        h_cat = torch.cat([h_i.expand(-1, m_cap, -1, -1), h_j], dim=-1)
        scores = self.leaky_relu(torch.einsum("bmhd,hd->bhm", h_cat, self.attn_vec))
        masked_scores = scores.masked_fill(~nbr_mask[:, None, :], -1e9)
        omega_heads = torch.softmax(masked_scores, dim=-1) * nbr_mask[:, None, :].to(scores.dtype)

        v = (self.w_v(c_nbr_self) + self.rel_encoder(nbr_rel)).view(
            bsz, m_cap, self.n_heads, self.d_head
        )
        mix = (
            self.cfg.prior_uniform_mix
            if prior_uniform_mix_override is None
            else float(prior_uniform_mix_override)
        )
        c_nbr, pi_prior, omega_bar = _apply_slot_interventions_and_prior(
            omega_heads=omega_heads,
            v_heads=v,
            nbr_mask=nbr_mask,
            w_o=self.w_o,
            context_ffn=self.context_ffn,
            eps_s=self.cfg.eps_s,
            uniform_prior=False,
            prior_uniform_mix=mix,
            slot_suppress_mask=slot_suppress_mask,
            attention_mode=attention_mode,
        )
        return {
            "c_nbr": c_nbr,
            "pi_prior": pi_prior,
            "omega_bar": omega_bar,
            "scores": scores.mean(dim=1),
        }


class TarMACAttention(nn.Module):
    """B1 TarMAC-style Targeted Message Attention Adaptation (Das et al. [42], Table S32).

    One communication round per policy step, 64-dim messages, two attention heads, signature-key
    vs. sender-query message aggregation, and uniform power prior (no MTC demand-capability prior).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.num_heads  # 2
        self.d_head = 32              # 2 * 32 = 64-dim message

        self.msg_gen = nn.Sequential(
            nn.Linear(cfg.embed_dim + 7, 64),
            nn.Tanh(),
            nn.Linear(64, cfg.embed_dim),
        )
        self.q_proj = nn.Linear(cfg.embed_dim, self.n_heads * self.d_head)
        self.sig_proj = nn.Linear(cfg.embed_dim, self.n_heads * self.d_head)
        self.val_proj = nn.Linear(cfg.embed_dim, self.n_heads * self.d_head)
        self.w_o = nn.Linear(self.n_heads * self.d_head, cfg.embed_dim)
        self.context_ffn = nn.Sequential(
            nn.Linear(cfg.embed_dim, 64),
            nn.Tanh(),
            nn.Linear(64, cfg.embed_dim),
        )

    def forward(
        self,
        r_query: torch.Tensor,
        c_self: torch.Tensor,
        c_nbr_self: torch.Tensor,
        nbr_rel: torch.Tensor,
        nbr_dist_cost: torch.Tensor,
        nbr_hist_cost: torch.Tensor,
        nbr_energy_cost: torch.Tensor,
        nbr_mask: torch.Tensor,
        slot_suppress_mask: torch.Tensor | None = None,
        attention_mode: str = "normal",
        prior_uniform_mix_override: float | None = None,
    ) -> Dict[str, torch.Tensor]:
        bsz, m_cap, _ = c_nbr_self.shape
        has_any = nbr_mask.any(dim=-1)
        if not bool(has_any.any()):
            zeros_c = torch.zeros((bsz, self.cfg.embed_dim), device=c_self.device, dtype=c_self.dtype)
            zeros_m = torch.zeros((bsz, m_cap), device=c_self.device, dtype=c_self.dtype)
            return {"c_nbr": zeros_c, "pi_prior": zeros_m, "omega_bar": zeros_m, "scores": zeros_m}

        messages = self.msg_gen(torch.cat([c_nbr_self, nbr_rel], dim=-1))  # (B, M, 64)
        q = self.q_proj(c_self + r_query).view(bsz, self.n_heads, self.d_head)
        k_sig = self.sig_proj(messages).view(bsz, m_cap, self.n_heads, self.d_head)
        v = self.val_proj(messages).view(bsz, m_cap, self.n_heads, self.d_head)

        scale = float(self.d_head) ** -0.5
        scores = torch.einsum("bhd,bmhd->bhm", q, k_sig) * scale
        masked_scores = scores.masked_fill(~nbr_mask[:, None, :], -1e9)
        omega_heads = torch.softmax(masked_scores, dim=-1) * nbr_mask[:, None, :].to(scores.dtype)

        # Table S32: B1 omits the MTC demand-capability prior (uniform_prior=True)
        c_nbr, pi_prior, omega_bar = _apply_slot_interventions_and_prior(
            omega_heads=omega_heads,
            v_heads=v,
            nbr_mask=nbr_mask,
            w_o=self.w_o,
            context_ffn=self.context_ffn,
            eps_s=self.cfg.eps_s,
            uniform_prior=True,
            prior_uniform_mix=0.0,
            slot_suppress_mask=slot_suppress_mask,
            attention_mode=attention_mode,
        )
        return {
            "c_nbr": c_nbr,
            "pi_prior": pi_prior,
            "omega_bar": omega_bar,
            "scores": scores.mean(dim=1),
        }


def build_attention_module(cfg: ModelConfig) -> nn.Module | None:
    """Factory returning the method-specific attention module (or None for attention-free methods)."""
    method = cfg.method.lower()
    if method == "mtc_mappo":
        if cfg.ablate_task_attention:
            return None
        return MTCTaskCapabilityAttention(cfg)
    if method == "morph_gat":
        return MorphGATAttention(cfg)
    if method == "gat_mappo":
        return GATAttention(cfg)
    if method == "tarmac_b1":
        return TarMACAttention(cfg)
    if method in ("mappo_morph", "role_mappo", "mappo"):
        return None
    raise ValueError(f"Unsupported model method: {cfg.method}")
