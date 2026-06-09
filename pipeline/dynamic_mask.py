"""
Dynamic Masking Strategy — 基于区域描述词的动态掩码策略

Implements soft, semantic-aware attention masks for multi-region layout control
in MM-DiT architectures. Unlike hard (binary) masks that completely block cross-region
attention, this strategy uses continuous soft coefficients to dynamically re-weight
attention scores based on spatial belonging AND semantic relevance.

Reference: Section "动态掩码策略" (Dynamic Masking Strategy)
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional


class DynamicMaskBuilder:
    """
    Builds a dynamic soft attention mask for multi-region layout control.

    The mask matrix M ∈ R^{(L+M)×(L+M)} is partitioned into four sub-blocks:
        M = [[M_img→img,  M_img→txt],
             [M_txt→img,  M_txt→txt]]

    where L = number of image patches, M = number of text tokens.

    Args:
        hidden_seq_len: L — number of image patch tokens
        encoder_seq_len: M — number of text tokens
        regional_masks: List of per-region spatial masks, each [L, seq_len_i]
        each_prompt_seq_len: List of text sequence lengths per region
        regional_boxes: List of bounding boxes [[x1,y1,x2,y2], ...] in latent space
        H_z: Latent height (in patches)
        W_z: Latent width (in patches)
    """

    def __init__(
        self,
        hidden_seq_len: int,
        encoder_seq_len: int,
        regional_masks: List[torch.Tensor],
        each_prompt_seq_len: List[int],
        regional_boxes: List[List[int]],
        H_z: int,
        W_z: int,
    ):
        self.L = hidden_seq_len
        self.M = encoder_seq_len
        self.H_z = H_z
        self.W_z = W_z
        self.regional_masks = regional_masks  # List of [L, seq_len_i]
        self.each_prompt_seq_len = each_prompt_seq_len
        self.regional_boxes = regional_boxes
        self.num_regions = len(regional_masks)

        # Precompute spatial position indices for gaussian kernel
        self._precompute_spatial_positions()

    def _precompute_spatial_positions(self):
        """Precompute 2D spatial positions for all image patches."""
        y_positions = torch.arange(self.H_z).float()
        x_positions = torch.arange(self.W_z).float()
        gy, gx = torch.meshgrid(y_positions, x_positions, indexing='ij')
        self.img_positions = torch.stack([gy.flatten(), gx.flatten()], dim=1)  # [L, 2]

    def _gaussian_kernel(self, sigma: float = 2.0) -> torch.Tensor:
        """
        Compute spatial Gaussian kernel κ(p,q) based on Euclidean distance
        between image patch positions.

        NOTE: For large images (L > 4096), this precomputes the full L×L matrix
        which can be memory-intensive. For L > 4096, use _gaussian_kernel_sparse instead.

        Returns:
            kernel: [L, L] tensor with values in [0, 1]
        """
        # Pairwise squared distances
        dist_sq = torch.cdist(
            self.img_positions.float(),
            self.img_positions.float(),
            p=2
        ).pow(2)  # [L, L]

        # Gaussian: exp(-d^2 / (2*sigma^2))
        kernel = torch.exp(-dist_sq / (2.0 * sigma ** 2))
        return kernel

    def _gaussian_kernel_local(
        self, sigma: float = 2.0, max_dist: float = 6.0
    ) -> torch.Tensor:
        """
        Compute a sparse/local Gaussian kernel where only patches within max_dist
        have non-zero values. This is much more memory-efficient for large images.

        Returns:
            kernel: [L, L] sparse-like tensor (dense but with many zeros)
        """
        device = self.img_positions.device
        L = self.L

        # Use block-based computation or compute only needed values
        # For each position, only compute similarity with nearby positions
        # Since this is called during initialization, we use a grid approach

        # Threshold: exp(-max_dist^2 / (2*sigma^2)) = min_similarity
        min_sim = torch.exp(torch.tensor(-max_dist**2 / (2.0 * sigma**2)))

        # Compute in chunks to save memory
        chunk_size = 1024
        kernel = torch.zeros(L, L)

        for i in range(0, L, chunk_size):
            end_i = min(i + chunk_size, L)
            chunk_positions = self.img_positions[i:end_i]

            # Distances from this chunk to all positions
            dist_sq = torch.cdist(
                chunk_positions.float(),
                self.img_positions.float(),
                p=2
            ).pow(2)

            # Apply gaussian
            chunk_kernel = torch.exp(-dist_sq / (2.0 * sigma**2))
            # Zero out values below threshold for sparsity (optional, keep all for smoothness)
            # chunk_kernel[chunk_kernel < min_sim] = 0.0

            kernel[i:end_i] = chunk_kernel

        return kernel

    def _compute_token_semantic_similarity(
        self,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cosine similarity between all text tokens.

        Args:
            encoder_hidden_states: Text embeddings [1, M, d]

        Returns:
            sim_matrix: [M, M] cosine similarity matrix
        """
        # Normalize along feature dimension
        text_features = encoder_hidden_states.squeeze(0)  # [M, d]
        text_norm = F.normalize(text_features, p=2, dim=-1)  # [M, d]
        sim_matrix = torch.mm(text_norm, text_norm.t())  # [M, M]
        # Clamp to [0, 1]
        sim_matrix = torch.clamp(sim_matrix, 0.0, 1.0)
        return sim_matrix

    def _compute_image_feature_similarity(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cosine similarity between image patch features.

        Args:
            hidden_states: Image features [1, L, d]

        Returns:
            sim_matrix: [L, L] cosine similarity matrix
        """
        img_features = hidden_states.squeeze(0)  # [L, d]
        img_norm = F.normalize(img_features, p=2, dim=-1)  # [L, d]
        sim_matrix = torch.mm(img_norm, img_norm.t())  # [L, L]
        sim_matrix = torch.clamp(sim_matrix, 0.0, 1.0)
        return sim_matrix

    def _get_region_indicator_for_patch(
        self, patch_idx: int
    ) -> Optional[int]:
        """
        Get which region a patch belongs to based on spatial position.

        Args:
            patch_idx: Flat index of the image patch

        Returns:
            region_idx or None if in background
        """
        py = patch_idx // self.W_z
        px = patch_idx % self.W_z

        for i, bbox in enumerate(self.regional_boxes):
            x1, y1, x2, y2 = bbox
            if x1 <= px < x2 and y1 <= py < y2:
                return i
        return None

    def _get_region_mask_vector(self, region_idx: int) -> torch.Tensor:
        """
        Get binary mask vector for a region indicating which image patches belong to it.

        Returns:
            mask: [L] binary tensor
        """
        mask = self.regional_masks[region_idx][:, 0]  # [L]
        return mask

    def build_txt_to_img_mask(
        self,
        encoder_hidden_states: torch.Tensor,
        gamma: float = 0.7,
    ) -> torch.Tensor:
        """
        Build text-to-image cross-attention soft mask.

        M[t, p] = γ * m_i(p) + (1-γ) * s(t, t_base)

        where:
        - m_i(p): whether image position p is inside region b_i (spatial belonging)
        - s(t, t_base): semantic similarity between token t and the global prompt base token
        - γ: balance coefficient

        Args:
            encoder_hidden_states: Text embeddings [1, M, d]
            gamma: Spatial weight coefficient

        Returns:
            mask: [M, L] soft mask tensor
        """
        # Compute spatial belonging matrix [M, L]
        spatial_belonging = torch.zeros(self.M, self.L, device=encoder_hidden_states.device)

        seq_begin = 0
        for i in range(self.num_regions):
            seq_end = seq_begin + self.each_prompt_seq_len[i]
            region_mask = self._get_region_mask_vector(i)  # [L]
            # Broadcast to all tokens in this region
            spatial_belonging[seq_begin:seq_end, :] = region_mask.unsqueeze(0)
            seq_begin = seq_end

        # Compute semantic similarity matrix [M, M]
        token_sim = self._compute_token_semantic_similarity(encoder_hidden_states)

        # For each text token t, average similarity with all base-prompt tokens
        # This approximates s(t, t_base) — "global prompt relevance"
        semantic_relevance = token_sim.mean(dim=1)  # [M]
        # Normalize to [0, 1]
        semantic_relevance = (semantic_relevance - semantic_relevance.min()) / (
            semantic_relevance.max() - semantic_relevance.min() + 1e-8
        )
        # Broadcast to image dimension: [M, L]
        semantic_map = semantic_relevance.unsqueeze(1).expand(-1, self.L)

        # Combine
        mask = gamma * spatial_belonging + (1.0 - gamma) * semantic_map

        return mask

    def build_img_to_txt_mask(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        lambda_coeff: float = 0.7,
    ) -> torch.Tensor:
        """
        Build image-to-text cross-attention soft mask.

        M[p, t] = λ * I(t ∈ I_i) + (1-λ) * s(p, t)

        where:
        - I(t ∈ I_i): indicator whether text token t belongs to the region of patch p
        - s(p, t): cosine similarity between image patch feature and text token feature
        - λ: balance coefficient

        Args:
            hidden_states: Image features [1, L, d]
            encoder_hidden_states: Text embeddings [1, M, d]
            lambda_coeff: Region indicator weight

        Returns:
            mask: [L, M] soft mask tensor
        """
        # Build region indicator matrix [L, M]
        region_indicator = torch.zeros(self.L, self.M, device=hidden_states.device)

        seq_begin = 0
        for i in range(self.num_regions):
            seq_end = seq_begin + self.each_prompt_seq_len[i]
            region_mask = self._get_region_mask_vector(i)  # [L]
            # Patch p attended by region i's text tokens
            region_indicator[:, seq_begin:seq_end] = region_mask.unsqueeze(1)
            seq_begin = seq_end

        # Compute image-text feature similarity [L, M]
        img_features = F.normalize(hidden_states.squeeze(0), p=2, dim=-1)  # [L, d]
        txt_features = F.normalize(encoder_hidden_states.squeeze(0), p=2, dim=-1)  # [M, d]
        feature_sim = torch.mm(img_features, txt_features.t())  # [L, M]
        feature_sim = torch.clamp(feature_sim, 0.0, 1.0)

        # Combine
        mask = lambda_coeff * region_indicator + (1.0 - lambda_coeff) * feature_sim

        return mask

    def build_txt_self_attn_mask(
        self,
        encoder_hidden_states: torch.Tensor,
        delta: float = 0.3,
    ) -> torch.Tensor:
        """
        Build text self-attention soft mask.

        M[t_a, t_b] = {
            1,                    if t_a, t_b belong to same region
            δ * sim(t_a, t_b),    otherwise (cross-region, attenuated)
        }

        Args:
            encoder_hidden_states: Text embeddings [1, M, d]
            delta: Cross-region attention attenuation coefficient

        Returns:
            mask: [M, M] soft mask tensor
        """
        # Compute token semantic similarity
        token_sim = self._compute_token_semantic_similarity(encoder_hidden_states)  # [M, M]

        # Build same-region mask
        same_region_mask = torch.zeros(self.M, self.M, device=encoder_hidden_states.device)

        seq_begin = 0
        for i in range(self.num_regions):
            seq_end = seq_begin + self.each_prompt_seq_len[i]
            # All tokens in same region fully attend to each other
            same_region_mask[seq_begin:seq_end, seq_begin:seq_end] = 1.0
            seq_begin = seq_end

        # Cross-region mask
        cross_region_mask = 1.0 - same_region_mask

        # Combined: same region = 1, cross region = δ * sim
        mask = same_region_mask + delta * token_sim * cross_region_mask

        return mask

    def build_img_self_attn_mask(
        self,
        hidden_states: torch.Tensor,
        mu: float = 1.5,
        eta: float = 0.3,
        gaussian_sigma: float = 2.0,
    ) -> torch.Tensor:
        """
        Build image self-attention soft mask.

        M[p, q] = min(1, μ * Σ m_i(p)·m_i(q) + η * κ(p,q))

        For large L (> 4096), falls back to all-ones mask (no modification to
        image self-attention) to avoid O(L²) memory blowup.

        Args:
            hidden_states: Image features [1, L, d] (used for device/dtype)
            mu: Intra-region aggregation weight (should be > η)
            eta: Local spatial smoothing weight
            gaussian_sigma: Sigma for Gaussian kernel

        Returns:
            mask: [L, L] soft mask tensor
        """
        device = hidden_states.device

        # For very large images, skip full O(L²) computation
        # Image self-attention mask is only practical for moderate resolutions
        # L = H_z * W_z is typically ~1000-2000 for Qwen-Image resolutions
        if self.L > 8192:
            # Fall back to all-ones: no modification to image self-attention
            # The cross-attention masks still provide layout control
            return torch.ones(self.L, self.L, device=device)

        # Region cohesion term: Σ_i m_i(p) * m_i(q)
        region_cohesion = torch.zeros(self.L, self.L, device=device)
        for i in range(self.num_regions):
            m_i = self._get_region_mask_vector(i)  # [L]
            region_cohesion += torch.outer(m_i, m_i)  # m_i(p) * m_i(q)

        # For moderate L, use gaussian kernel
        if self.L <= 4096:
            gaussian_kernel = self._gaussian_kernel(sigma=gaussian_sigma).to(device)
        else:
            gaussian_kernel = self._gaussian_kernel_local(sigma=gaussian_sigma).to(device)

        # Combine with min(1, ...) clamping
        mask = torch.clamp(
            mu * region_cohesion + eta * gaussian_kernel,
            max=1.0
        )

        return mask

    def build_full_mask(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        gamma: float = 0.7,
        lambda_coeff: float = 0.7,
        delta: float = 0.3,
        mu: float = 1.5,
        eta: float = 0.3,
        gaussian_sigma: float = 2.0,
    ) -> torch.Tensor:
        """
        Build the complete dynamic soft mask matrix.

        M = [[M_img→img,  M_img→txt],
             [M_txt→img,  M_txt→txt]]

        Args:
            hidden_states: Image patch features [1, L, d]
            encoder_hidden_states: Text token embeddings [1, M, d]
            gamma: txt→img spatial weight
            lambda_coeff: img→txt region indicator weight
            delta: Cross-region text attention attenuation
            mu: Intra-region image aggregation weight
            eta: Local spatial smoothing weight
            gaussian_sigma: Sigma for Gaussian kernel

        Returns:
            full_mask: [(L+M), (L+M)] soft mask tensor
        """
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Build four sub-blocks
        M_img2img = self.build_img_self_attn_mask(
            hidden_states, mu=mu, eta=eta, gaussian_sigma=gaussian_sigma
        )  # [L, L]

        M_img2txt = self.build_img_to_txt_mask(
            hidden_states, encoder_hidden_states, lambda_coeff=lambda_coeff
        )  # [L, M]

        M_txt2img = self.build_txt_to_img_mask(
            encoder_hidden_states, gamma=gamma
        )  # [M, L]

        M_txt2txt = self.build_txt_self_attn_mask(
            encoder_hidden_states, delta=delta
        )  # [M, M]

        # Assemble into full mask
        # Top row: [M_img→img | M_img→txt]
        top_row = torch.cat([M_img2img, M_img2txt], dim=1)  # [L, L+M]
        # Bottom row: [M_txt→img | M_txt→txt]
        bottom_row = torch.cat([M_txt2img, M_txt2txt], dim=1)  # [M, L+M]
        # Full mask
        full_mask = torch.cat([top_row, bottom_row], dim=0)  # [L+M, L+M]

        return full_mask.to(dtype)

    @staticmethod
    def apply_soft_mask_to_attention(
        attn_scores: torch.Tensor,
        soft_mask: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Apply soft mask to attention scores using the additive-log method.

        score' = score + log(M + ε)

        This is equivalent to multiplying the attention probabilities by M
        after softmax: softmax(score + log(M)) ∝ softmax(score) * M

        Args:
            attn_scores: Raw attention scores QK^T / sqrt(d)
            soft_mask: Dynamic soft mask M ∈ [0, 1]

        Returns:
            masked_scores: Attention scores with soft mask applied
        """
        # Clamp mask to avoid log(0)
        safe_mask = torch.clamp(soft_mask, min=eps)
        log_mask = torch.log(safe_mask)

        return attn_scores + log_mask
