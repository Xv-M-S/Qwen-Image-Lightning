from diffusers import QwenImagePipeline
import torch
from typing import Union, List, Optional, Dict, Any, Callable, Tuple
from diffusers.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipelineOutput
import numpy as np
from diffusers.pipelines.qwenimage.pipeline_qwenimage import calculate_shift, retrieve_timesteps, XLA_AVAILABLE
# import torch_xla.core.xla_model as xm
from diffusers.models.attention_processor import Attention
from diffusers.models.attention_dispatch import dispatch_attention_fn
import torch.nn.functional as F
from pipeline.attentionControl import AttentionStore
import re
from pipeline.gaussion_smoothing import GaussianSmoothing
from config.boxLossConfig import boxConfig
from util.tool import cost_time
# from torchviz import make_dot
from util.visual import visualize_feature_activation, visualize_feature_channel, visualize_latent_map
from pipeline.lossDesign import compute_diff_loss
from pipeline.rnbLoss import compute_rnb_loss
from util.latent_search import histogram_matching, process_matrix_with_noise
from pipeline.LossUtil import LossUtil
from pipeline.optimizeLoss import compute_opt_loss
from pipeline.noise_filtering import compute_nqe_eaa_scores, select_best_noise
from pipeline.dynamic_mask import DynamicMaskBuilder
import random
import re
from difflib import SequenceMatcher
import nltk
from nltk.stem import PorterStemmer  # 用于提取词根（需安装nltk：pip install nltk）
nltk.data.path.append('/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/nltk_data') 


def apply_rotary_emb_qwen(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, S, H, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio, OmniGen, CogView4 and Cosmos
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, S, H, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(1)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)


class RegionalQwenImageAttnProcessor:
    """
    Attention processor for Qwen double-stream architecture, matching DoubleStreamLayerMegatron logic. This processor
    implements joint attention computation where text and image streams are processed together.

    Supports both hard (bool) regional attention masks and dynamic soft masks (when boxConfig.use_dynamic_mask=True).
    """

    _attention_backend = None

    def __init__(self, attnstore= None):
        self.regional_mask = None
        self.attnstore = attnstore
        # Dynamic soft mask cache
        self._dynamic_mask_builder = None
        self._cached_soft_mask = None
        self._cached_encoder_hidden_states_key = None
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "QwenDoubleStreamAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )
        


    def scaled_dot_product_attention_with_probs(self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False):
        """
        手动实现 scaled dot-product attention，并返回 attention probs
        query, key, value: [B, H, S, D]
        参考官方伪代码: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
        """
        L, S = query.size(-2), key.size(-2)
        scale = 1.0 / query.size(-1)**0.5

        # QK^T
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale  # [B, H, L, S]

        # 应用 causal mask
        # is_causal 参数在 scaled_dot_product_attention 函数中的作用是：
        # 启用因果注意力（Causal Attention）或称自回归掩码（Autoregressive Masking），
        # 它确保在解码过程中，每个位置只能关注到它之前（包括自身）的位置，而不能“看到”未来的信息。
        if is_causal:
            causal_mask = torch.triu(torch.ones(L, S, dtype=torch.bool, device=attn_scores.device), diagonal=1)
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

        # 应用用户提供的 mask
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask  # 注意：attn_mask 应为 float，-inf 表示遮挡

        # Softmax -> attention probs
        attn_probs = F.softmax(attn_scores, dim=-1)  # [B, H, L, S]

        # Dropout
        attn_probs = F.dropout(attn_probs, p=dropout_p, training=True)

        # @V
        attn_output = torch.matmul(attn_probs, value)  # [B, H, L, D]

        return attn_output, attn_probs  # ✅ 返回 probs
        
    def RegionalQwenAttnProcessor2_0_call(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,  # Image stream
        encoder_hidden_states: torch.FloatTensor = None,  # Text stream
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        index_block: str = None
    ) -> torch.FloatTensor:

        if encoder_hidden_states is None:
            raise ValueError("QwenDoubleStreamAttnProcessor2_0 requires encoder_hidden_states (text stream)")

        seq_txt = encoder_hidden_states.shape[1]

        # Compute QKV for image stream
        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)

        # Compute QKV for text stream
        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        # Reshape for multi-head attention
        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))

        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        # Apply QK normalization
        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        # Apply RoPE
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

        # Concatenate for joint attention: [text, image]
        joint_query = torch.cat([txt_query, img_query], dim=1).permute(0,2,1,3)  # dim=2: seq_len
        joint_key = torch.cat([txt_key, img_key], dim=1).permute(0,2,1,3)
        joint_value = torch.cat([txt_value, img_value], dim=1).permute(0,2,1,3)



        joint_hidden_states, attn_probs = self.scaled_dot_product_attention_with_probs(
            query=joint_query,
            key=joint_key,
            value=joint_value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False
        )

        # Store attention for box loss
        if self.attnstore is not None and boxConfig.switch_box_loss and index_block in boxConfig.train_layer:
            # Extract image-to-text cross attention: img_query vs txt_key
            # joint_query: [txt_query, img_query] -> img_query starts at seq_txt
            # joint_key:   [txt_key,   img_key]   -> txt_key ends at seq_txt
            img_txt_attn = attn_probs[:, :, seq_txt:, :seq_txt]  # [B, H, S_img, S_txt]
            img_txt_attn = img_txt_attn.mean(dim=1)  # Average over heads -> [B, S_img, S_txt]
            
            txt_img_attn = attn_probs[:, :, :seq_txt, seq_txt:]
            txt_img_attn = txt_img_attn.mean(dim=1)

            # self.attnstore(img_txt_attn, "img-to-txt")  # 放在一起会导致迭代,是一种代码错误
            # self.attnstore(txt_img_attn, "txt-to-img")
            self.attnstore(img_txt_attn, "img-to-txt", txt_img_attn, "txt-to-img")
            if img_txt_attn.shape[2] == boxConfig.text_len and boxConfig.visual_attention_map:
                # no viusal of negative prompt
                visualize_feature_activation(img_txt_attn.clone().detach(), index_block)
                visualize_feature_channel(img_txt_attn.clone().detach(), index_block)
                visualize_feature_channel(txt_img_attn.clone().detach().transpose(1, 2), index_block, "txt-to-img")

        # Reshape back
        joint_hidden_states = joint_hidden_states.transpose(1, 2).flatten(2, 3)  # [B, S_joint, H*D]
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        # Split
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]
        img_attn_output = joint_hidden_states[:, seq_txt:, :]

        # Output projections
        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)

        txt_attn_output = attn.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output
    def RegionalQwenAttnProcessor2_0_call_old(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,  # Image stream
        encoder_hidden_states: torch.FloatTensor = None,  # Text stream
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        index_block: str = None
    ) -> torch.FloatTensor:
        if encoder_hidden_states is None:
            raise ValueError("QwenDoubleStreamAttnProcessor2_0 requires encoder_hidden_states (text stream)")


        seq_txt = encoder_hidden_states.shape[1]

        # Compute QKV for image stream (sample projections)
        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)

        # Compute QKV for text stream (context projections)
        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        # Reshape for multi-head attention
        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))

        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        # Apply QK normalization
        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        # Apply RoPE
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

        # Concatenate for joint attention
        # Order: [text, image]
        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        # Compute joint attention
        joint_hidden_states = dispatch_attention_fn(
            joint_query,
            joint_key,
            joint_value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
        )

        # 使用不加速的方式重新计算一遍 attention socre
        if self.attnstore is not None and boxConfig.switch_box_loss and index_block in boxConfig.train_layer:
            score_query = hidden_states # img 
            score_key = encoder_hidden_states # txt
            attention_probs = attn.get_attention_scores(score_query, score_key, attention_mask=None)
            is_cross = encoder_hidden_states is not None
            self.attnstore(attention_probs, is_cross)


        # Reshape back
        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        # Split attention outputs back
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]  # Text part
        img_attn_output = joint_hidden_states[:, seq_txt:, :]  # Image part

        # Apply output projections
        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)  # dropout

        txt_attn_output = attn.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output
          

    def _build_dynamic_soft_mask(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Optional[torch.Tensor]:
        """
        Build dynamic soft mask from current hidden_states and encoder_hidden_states.

        Uses the DynamicMaskBuilder that was cached during pipeline setup.
        Converts the soft mask to log-space for additive application in attention.
        """
        if not boxConfig.use_dynamic_mask:
            return None
        if self._dynamic_mask_builder is None:
            return None

        builder = self._dynamic_mask_builder

        # Build full soft mask from current features
        soft_mask = builder.build_full_mask(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            gamma=boxConfig.dynamic_mask_gamma,
            lambda_coeff=boxConfig.dynamic_mask_lambda,
            delta=boxConfig.dynamic_mask_delta,
            mu=boxConfig.dynamic_mask_mu,
            eta=boxConfig.dynamic_mask_eta,
            gaussian_sigma=boxConfig.dynamic_mask_gaussian_sigma,
        )

        # Convert to log-space for additive mask: attn_scores + log(M + ε)
        log_mask = DynamicMaskBuilder.apply_soft_mask_to_attention(
            attn_scores=torch.zeros_like(soft_mask),
            soft_mask=soft_mask,
        ) - 0.0  # Remove identity for clarity: log_mask = log(M + ε)

        # Actually let's compute it directly:
        eps = 1e-8
        safe_mask = torch.clamp(soft_mask, min=eps)
        log_mask = torch.log(safe_mask)

        return log_mask

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,  # Image stream
        encoder_hidden_states: torch.FloatTensor = None,  # Text stream
        encoder_hidden_states_mask: torch.FloatTensor = None,
        encoder_hidden_states_base: torch.FloatTensor = None, # added
        encoder_hidden_states_base_mask: torch.FloatTensor = None, # added
        base_ratio: float = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        image_rotary_emb_base: Optional[torch.Tensor] = None, # added
        joint_attention_kwargs: Optional[Dict[str, Any]] = None, # added
    ) -> torch.FloatTensor:
        index_block = joint_attention_kwargs["index_block"] if joint_attention_kwargs is not None else None
        if base_ratio is not None:
            hidden_states_base, encoder_hidden_states_base = self.RegionalQwenAttnProcessor2_0_call(
                attn = attn,
                hidden_states = hidden_states,
                encoder_hidden_states = encoder_hidden_states_base,
                encoder_hidden_states_mask = encoder_hidden_states_base_mask,
                attention_mask = attention_mask,
                image_rotary_emb=image_rotary_emb_base,
                index_block = index_block
            )

        # Build attention mask (hard bool or dynamic soft)
        if boxConfig.use_dynamic_mask and base_ratio is not None:
            # Dynamic soft masking: build mask from current features
            regional_mask = self._build_dynamic_soft_mask(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                joint_attention_kwargs=joint_attention_kwargs,
            )
            if regional_mask is not None:
                regional_mask = regional_mask.to(hidden_states.device).to(hidden_states.dtype)
        elif base_ratio is not None and joint_attention_kwargs is not None and 'regional_attention_mask' in joint_attention_kwargs:
            # Hard bool mask (original behavior)
            self.regional_mask = joint_attention_kwargs['regional_attention_mask']
            regional_mask = self.regional_mask
        else:
            regional_mask = None

        hidden_states, encoder_hidden_states = self.RegionalQwenAttnProcessor2_0_call(
            attn = attn,
            hidden_states = hidden_states,
            encoder_hidden_states = encoder_hidden_states,
            encoder_hidden_states_mask = encoder_hidden_states_mask,
            attention_mask = regional_mask,
            image_rotary_emb=image_rotary_emb,
            index_block = index_block
        )


        if base_ratio is not None:
            if 'whole_regional_mask' in joint_attention_kwargs and joint_attention_kwargs["enable_whole_regional_mask"]:
                joint_attention_kwargs["whole_regional_mask"] = joint_attention_kwargs["whole_regional_mask"]  # .to(hidden_states.device).to(hidden_states.dtype)
                # mask merge method 1: non_regional_area 100 per use base_prompt
                hidden_states = hidden_states * joint_attention_kwargs["whole_regional_mask"] *(1 - base_ratio) + hidden_states_base * joint_attention_kwargs["whole_regional_mask"] * base_ratio
                hidden_states = hidden_states + hidden_states_base * (1 - joint_attention_kwargs["whole_regional_mask"])

                # apply regional control to hidden_states_base
                # hidden_states will be used for generate hidden_states_base and hidden_states for next step

                # mask merge method 2: regional_area use 100 per regional_control and add base_ratio non_regional_control
                # hidden_states = hidden_states * joint_attention_kwargs["whole_regional_mask"] * (1 - base_ratio) + hidden_states_base * base_ratio
                # hidden_states = hidden_states / (joint_attention_kwargs["whole_regional_mask"] +  base_ratio)  # avoid division by zero
            else:
                # merge hidden_states and hidden_states_base
                hidden_states = hidden_states*(1-base_ratio) + hidden_states_base*base_ratio
            return hidden_states, encoder_hidden_states, encoder_hidden_states_base
        else: # both regional and base input are base prompts, skip the merge
            return hidden_states, encoder_hidden_states, encoder_hidden_states



        

class RegionalQwenImagePipeline(QwenImagePipeline):

    def get_token_index(self, prompt, quote_prompt, region_prompts):
        DEBUG = False
        # 对基础提示进行 tokenization
        # 步骤 1: 提取所有双引号内的内容（保留原样）
        # if region_prompts==None:
        if quote_prompt:
            if DEBUG:print("🔍 提取的引号内容:")
            quoted_texts = re.findall(r'"(.*?)"', prompt)
            for i, text in enumerate(quoted_texts):
                if DEBUG:print(f"  [{i}] {repr(text)}")
        else:
            quoted_texts = region_prompts
        
        # 步骤 2: Tokenize 整个 prompt
        tokens = self.tokenizer.tokenize(prompt)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        if DEBUG:print(f"\n📊 总 token 数: {len(tokens)}")
        if DEBUG:print("🔤 前 20 个 tokens:", tokens[:100])

        # 步骤 3: 对每个引号内容，查找其在 token stream 中的位置
        if DEBUG:print("\n📌 引号内容在 tokenizer 中的位置:")
        quote_to_token_positions = {}

        for idx, quote in enumerate(quoted_texts):
            if DEBUG:print(f"\n--- 处理引号内容 [{idx}]: {repr(quote)} ---")

            # 将引号内容单独 tokenize
            quote_tokens = self.tokenizer.tokenize(quote)
            quote_token_ids = self.tokenizer.convert_tokens_to_ids(quote_tokens)
            quote_token_ids = quote_token_ids[1:] # 去掉开一个字符

            if DEBUG:print(f"  Tokenized 子串: {quote_tokens}")
            if DEBUG:print(f"  Token IDs: {quote_token_ids}")

            # 在完整 token stream 中查找匹配
            found = False
            positions = None
            for start_idx in range(len(token_ids) - len(quote_token_ids) + 1):
                end_idx = start_idx + len(quote_token_ids)
                if token_ids[start_idx:end_idx] == quote_token_ids:
                    if DEBUG:print(f"  ✅ 匹配位置: 起始索引 = {start_idx}, 结束索引 = {end_idx - 1} (token 范围: [{start_idx}:{end_idx}])")
                    # 可选：验证一下还原的文本
                    reconstructed = self.tokenizer.decode(token_ids[start_idx:end_idx])
                    if DEBUG:print(f"  🔁 重建文本: {repr(reconstructed)}")
                    found = True
                    positions = list(range(start_idx - 1, end_idx)) # 加一个字符
                    break

            if positions is not None:
                quote_to_token_positions[quote] = positions
                if DEBUG:print(f"  ✅ 匹配成功，token 位置: {positions}")
                if DEBUG:print(f"  🔄 从 tokens 重建: {repr(self.tokenizer.decode(token_ids[positions[0]:positions[-1]+1]))}")
            else:
                if DEBUG:print(f"  ❌ 未找到匹配")
                quote_to_token_positions[quote] = []
        
        # ======================
        # 6. 最终结果：每个引号内容对应的 token 位置集合
        # ======================
        if DEBUG:print("\n" + "="*60)
        if DEBUG:print("✅ 每个引号内容在 token 序列中的位置集合：")
        if DEBUG:print("="*60)
        for quote, positions in quote_to_token_positions.items():
            if DEBUG:print(f"""
        引号内容: "{quote}"
        位置集合: {positions}
        长度: {len(positions)} tokens
        """)
        
        # 从 Python 3.7 开始，dict（字典）保证保持插入顺序。
        return quote_to_token_positions

    
    def find_complete_word_matches(self, original_items, base_string):
        # 1. 提取基础字符串中的所有完整单词（以空格/标点分割，保留原始形态）
        # 正则匹配规则：仅包含字母的完整单词（忽略数字和特殊字符）
        base_words = re.findall(r"\b[a-zA-Z]+\b", base_string)
        # 转为小写用于匹配，同时保留原始单词用于结果输出
        base_words_lower = [word.lower() for word in base_words]
        base_word_pairs = list(zip(base_words, base_words_lower))  # [(原始单词, 小写单词)]

        # 2. 初始化词根提取器（处理词形变化：如 cups → cup，steaming → steam）
        stemmer = PorterStemmer()

        matches = {}
        for item in original_items:
            # 清理输入项：仅保留字母，转为小写
            item_clean = re.sub(r"[^a-zA-Z]", "", item).lower()
            if not item_clean:  # 过滤空字符串
                matches[item] = None
                continue

            best_match = None
            highest_score = 0.0
            item_stem = stemmer.stem(item_clean)  # 提取输入项的词根

            # 3. 遍历基础字符串中的所有完整单词，寻找最佳匹配
            for original_word, lower_word in base_word_pairs:
                # 计算原始字符串相似度
                similarity = SequenceMatcher(None, item_clean, lower_word).ratio()

                # 4. 词根匹配优化（处理词形变化）
                word_stem = stemmer.stem(lower_word)
                if word_stem == item_stem:
                    similarity += 0.3  # 词根相同则大幅提高权重

                # 5. 确保不匹配部分单词（仅接受完整单词匹配）
                # 例如：item是"cup"时，"cups"算匹配，但"cupboard"不算
                if (item_clean in lower_word and len(item_clean) < len(lower_word)) and word_stem != item_stem:
                    continue  # 排除部分包含但词根不同的情况（如"cup" vs "cupboard"）

                # 6. 更新最佳匹配
                if similarity > highest_score:
                    highest_score = similarity
                    best_match = original_word  # 保留原始单词的大小写和形态

            matches[item] = best_match

        return matches

    def get_token_index_v2(self, prompt, quote_prompt, region_prompts):
        DEBUG = True
        # if region_prompts==None:
        quoted_texts = region_prompts

        # 特殊处理，除去特殊字符下划线符
        quoted_texts = [text.replace("_", " ") for text in quoted_texts]
        # 针对counting的处理，如果第一个单词是‘a’ 或者‘an’，则去掉
        processed_texts = []
        for text in quoted_texts:
            if text.startswith(("a ")):
                text = text[2:]
            elif text.startswith(("an ")):
                text = text[3:]
            processed_texts.append(text)

        quoted_texts = processed_texts

        # 应对下面的例子
        # original_array = ['cup', 'steam', 'hot coffee', 'wooden table', 'side']
        # base_string = 'two cups filled with steaming hot coffee sit side-by-side on a wooden table. Ultra HD, 4K, cinematic composition.'
        # cup → cups
        # steam → steaming
        # hot coffee → hot coffee
        # wooden table → wooden table
        # side → side-by-side
        matches = self.find_complete_word_matches(quoted_texts, prompt)
        for index, value in enumerate(quoted_texts):
            if matches[value] is not None:
                if DEBUG:print(f"🔍 [{index}] '{value}' 匹配为完整单词 '{matches[value]}'")
                quoted_texts[index] = matches[value]
            else:
                if DEBUG:print(f"❌ [{index}] '{value}' 未找到完整单词匹配，保持不变")


            
        
        # 步骤 2: Tokenize 整个 prompt
        tokens = self.tokenizer.tokenize(prompt)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        if DEBUG:print(f"\n📊 总 token 数: {len(tokens)}")
        if DEBUG:print("🔤 前 20 个 tokens:", tokens[:100])

        # 步骤 3: 对每个引号内容，查找其在 token stream 中的位置
        if DEBUG:print("\n📌 引号内容在 tokenizer 中的位置:")
        quote_to_token_positions = {}

        for idx, quote in enumerate(quoted_texts):
            original_quote = quote
            if '"' not in quote:
                quote = "a " + quote # 加一个字符，因为第一个字符的token会不一样
            if DEBUG:print(f"\n--- 处理引号内容 [{idx}]: {repr(quote)} ---")

            # 将引号内容单独 tokenize
            quote_tokens = self.tokenizer.tokenize(quote)
            quote_token_ids = self.tokenizer.convert_tokens_to_ids(quote_tokens)

            # if '"' in quote:
            quote_token_ids = quote_token_ids[1:] # 去掉开一个字符


            if DEBUG:print(f"  Tokenized 子串: {quote_tokens}")
            if DEBUG:print(f"  Token IDs: {quote_token_ids}")

            # 在完整 token stream 中查找匹配
            found = False
            positions = []

            if original_quote in prompt:
                for start_idx in range(len(token_ids) - len(quote_token_ids) + 1):
                    end_idx = start_idx + len(quote_token_ids)
                    if token_ids[start_idx:end_idx] == quote_token_ids:
                        if DEBUG:print(f"  ✅ 匹配位置: 起始索引 = {start_idx}, 结束索引 = {end_idx - 1} (token 范围: [{start_idx}:{end_idx}])")
                        # 可选：验证一下还原的文本
                        reconstructed = self.tokenizer.decode(token_ids[start_idx:end_idx])
                        if DEBUG:print(f"  🔁 重建文本: {repr(reconstructed)}")
                        found = True
                        if '"' in quote:
                            # 有引号需要特殊处理
                            positions = list(range(start_idx - 1, end_idx)) # 加一个字符
                        else:
                            positions = list(range(start_idx, end_idx))
                        break
            else:
                # 属于计算个数【count】的类型，导致匹配失败,计算每个单词在 prompt 中的出现位置
                for index, ids in enumerate(quote_token_ids):
                    if index == 0:
                        continue # 去掉第一个
                    if ids not in token_ids: # 预防极端情况
                        continue
                    positions.append(token_ids.index(ids))

            if len(positions)!=0:
                quote_to_token_positions[quote] = positions
                if DEBUG:print(f"  ✅ 匹配成功，token 位置: {positions}")
                if DEBUG:print(f"  🔄 从 tokens 重建: {repr(self.tokenizer.decode(token_ids[positions[0]:positions[-1]+1]))}")
            else:
                if DEBUG:print(f"  ❌ 未找到匹配")
                quote_to_token_positions[quote] = []
        
        # ======================
        # 6. 最终结果：每个引号内容对应的 token 位置集合
        # ======================
        if DEBUG:print("\n" + "="*60)
        if DEBUG:print("✅ 每个引号内容在 token 序列中的位置集合：")
        if DEBUG:print("="*60)
        for quote, positions in quote_to_token_positions.items():
            if DEBUG:print(f"""
        引号内容: "{quote}"
        位置集合: {positions}
        长度: {len(positions)} tokens
        """)
        
        # 从 Python 3.7 开始，dict（字典）保证保持插入顺序。
        return quote_to_token_positions


    import re

    def get_token_index_v3(self, prompt, quote_prompt, region_prompts):
        """
        修复后的版本，专注于在 prompt 的 token 化表示中查找 region_prompts 中的短语。
        它不再依赖 find_complete_word_matches 进行短语匹配，而是直接处理 prompt_tokens。
        """
        DEBUG = True
        # 预处理region_prompts
        if region_prompts is None:
            return {}
        # 保留原始描述，不做 "_" 替换，因为我们的数据中没有 "_"
        quoted_texts = [text.strip() for text in region_prompts]

        # 步骤 1: Tokenize 整个 prompt
        prompt_tokens = self.tokenizer.tokenize(prompt)
        prompt_token_ids = self.tokenizer.convert_tokens_to_ids(prompt_tokens)

        if DEBUG:
            print(f"\n📊 Prompt total tokens: {len(prompt_tokens)}")
            print("🔤 First 20 prompt tokens:", prompt_tokens[:20])

        # 步骤 2: 初始化结果字典
        quote_to_token_positions = {}

        for idx, original_quote_text in enumerate(quoted_texts):
            original_quote_text_lower = original_quote_text.lower()
            
            if DEBUG:
                print(f"\n--- Processing Quote [{idx}]: '{original_quote_text}' ---")

            # --- 策略：直接在 prompt_tokens 中查找 original_quote_text ---
            # 1. Tokenize the target phrase
            phrase_tokens = self.tokenizer.tokenize(original_quote_text)
            phrase_token_ids = self.tokenizer.convert_tokens_to_ids(phrase_tokens)
            
            if DEBUG:
                print(f"  🔄 Tokenizing phrase: '{original_quote_text}' -> {phrase_tokens}")
                print(f"  🔄 Phrase token IDs: {phrase_token_ids}")

            # 2. 在 prompt_token_ids 中查找这个 phrase_token_ids 的序列
            # 使用辅助函数查找子列表
            pos_list = self._find_sublist_indices(phrase_token_ids, prompt_token_ids)
            
            if pos_list:
                # 找到了完整的短语匹配
                quote_to_token_positions[original_quote_text] = pos_list
                if DEBUG:
                    print(f"  ✅ Found exact phrase match. Positions: {pos_list}")
                    # 验证重建的文本
                    reconstructed_text = self.tokenizer.decode([prompt_token_ids[i] for i in pos_list])
                    print(f"  🔁 Reconstructed from tokens: '{reconstructed_text}'")
            else:
                # 3. 如果找不到完整短语，尝试宽松匹配（处理不定冠词等）
                # 首先检查原始短语是否在 prompt 中（用于调试）
                if original_quote_text_lower in prompt.lower():
                    # 确保找到的是完整单词（使用正则表达式）
                    # 注意：这里的匹配仍然是基于字符的，但我们接下来会尝试 token 化
                    import re
                    # 使用 \b 作为单词边界，但需要小心处理包含特殊字符的短语
                    escaped_phrase = re.escape(original_quote_text_lower)
                    # 如果短语包含引号等，\b 可能不适用，我们稍后再处理这种情况
                    
                    # 先尝试一种更直接的方法：允许短语前后是单词边界（或引号、空格等）
                    # pattern = r'\b' + escaped_phrase + r'\b'
                    # 更宽松的模式，匹配短语本身及其周围环境
                    # 使用 (?=\b|\s|") 来表示单词边界或空格或引号
                    # 使用 (?<=\b|\s|") 来表示前面也是单词边界或空格或引号
                    # pattern = rf"(?<=\b|\s|'){escaped_phrase}(?=\b|\s|')"
                    # 这种方法依然难以完美解决 tokenizer 的问题
                    
                    # 最稳健的方法：尝试移除不定冠词后再次查找
                    text_lower = original_quote_text_lower
                    modified_phrase = original_quote_text
                    starts_with_a = False
                    starts_with_an = False
                    if text_lower.startswith("a "):
                        modified_phrase = original_quote_text[2:].strip()
                        starts_with_a = True
                    elif text_lower.startswith("an "):
                        modified_phrase = original_quote_text[3:].strip()
                        starts_with_an = True
                    
                    if starts_with_a or starts_with_an:
                        if DEBUG:
                            print(f"  ℹ️  Original phrase '{original_quote_text}' starts with 'a'/'an'. Trying modified: '{modified_phrase}'")
                        
                        # Tokenize the modified phrase
                        modified_phrase_tokens = self.tokenizer.tokenize(modified_phrase)
                        modified_phrase_token_ids = self.tokenizer.convert_tokens_to_ids(modified_phrase_tokens)
                        
                        if DEBUG:
                            print(f"  🔄 Tokenizing modified phrase: '{modified_phrase}' -> {modified_phrase_tokens}")
                            print(f"  🔄 Modified phrase token IDs: {modified_phrase_token_ids}")

                        # 在 prompt_token_ids 中查找修改后的序列
                        modified_pos_list = self._find_sublist_indices(modified_phrase_token_ids, prompt_token_ids)
                        
                        if modified_pos_list:
                            quote_to_token_positions[original_quote_text] = modified_pos_list
                            if DEBUG:
                                print(f"  ✅ Found modified phrase match (removed 'a'/'an'). Positions: {modified_pos_list}")
                                reconstructed_text = self.tokenizer.decode([prompt_token_ids[i] for i in modified_pos_list])
                                print(f"  🔁 Reconstructed from tokens: '{reconstructed_text}'")
                            continue # 成功匹配，继续下一个
                
                # 4. 如果以上都不行，尝试在 prompt 中查找原始短语（非 token 化）
                # 并获取其字符位置，然后转换为 token 位置
                import re
                # pattern = r'\b' + re.escape(original_quote_text_lower) + r'\b'
                # 更适合处理可能包含引号的短语
                # pattern = r'(?<!\w)' + re.escape(original_quote_text_lower) + r'(?!\w)'
                # 直接搜索，因为我们的 prompt 中很多短语是用引号括起来的
                pattern = re.escape(original_quote_text_lower)
                match = re.search(pattern, prompt.lower())
                
                if match:
                    char_start = match.start()
                    char_end = match.end()
                    
                    if DEBUG:
                        print(f"  ℹ️  Found character-based match: '{prompt[char_start:char_end]}' at char pos {char_start}-{char_end}")
                    
                    # 将字符位置转换为 token 位置
                    # 这需要知道每个 token 在原字符串中的字符范围
                    # tokenizer 通常有 offsets_mapping 或类似功能，但这里我们用一种近似方法
                    # 计算 prompt[:char_start] 和 prompt[:char_end] 的 tokens
                    
                    # 获取从开头到匹配开始前的 tokens
                    prefix_tokens = self.tokenizer.tokenize(prompt[:char_start])
                    prefix_token_count = len(prefix_tokens)
                    
                    # 获取从开头到匹配结束的 tokens
                    full_prefix_tokens = self.tokenizer.tokenize(prompt[:char_end])
                    full_prefix_token_count = len(full_prefix_tokens)
                    
                    # 匹配部分的 token 范围就是 [prefix_token_count, full_prefix_token_count)
                    token_start = prefix_token_count
                    token_end = full_prefix_token_count
                    
                    if token_start < len(prompt_tokens) and token_end <= len(prompt_tokens):
                        token_range = list(range(token_start, token_end))
                        quote_to_token_positions[original_quote_text] = token_range
                        if DEBUG:
                            print(f"  ✅ Found character-based match, converted to token positions: {token_range}")
                            reconstructed_text = self.tokenizer.decode([prompt_token_ids[i] for i in token_range])
                            print(f"  🔁 Reconstructed from tokens: '{reconstructed_text}'")
                    else:
                        if DEBUG:
                            print(f"  ❌ Character-to-token conversion failed due to out-of-bounds indices.")
                else:
                    # 5. 如果 prompt 中也没有找到原始短语，则认为是“计数”场景
                    if DEBUG:
                        print(f"  ℹ️  '{original_quote_text}' not found in prompt, treating as 'counting' scenario.")

                    # 将短语tokenize
                    individual_tokens = self.tokenizer.tokenize(original_quote_text)
                    individual_token_ids = self.tokenizer.convert_tokens_to_ids(individual_tokens)
                    
                    if DEBUG:
                        print(f"  🔄 Tokenizing for counting: {individual_tokens}")
                        print(f"  🔄 Individual token IDs: {individual_token_ids}")

                    # 查找每个独立token在prompt中的所有位置
                    all_pos_lists = []
                    for token_id in individual_token_ids:
                        try:
                            indices = [i for i, x in enumerate(prompt_token_ids) if x == token_id]
                            if indices:
                                all_pos_lists.append(indices[0]) # 取第一个匹配位置
                            else:
                                if DEBUG:
                                    print(f"    ❌ Could not find token ID {token_id} ('{self.tokenizer.decode([token_id])}') in prompt.")
                        except Exception as e:
                            if DEBUG:
                                print(f"    ❌ Error finding token ID {token_id}: {e}")
                    
                    # 检查是否所有token都找到了
                    if len(all_pos_lists) == len(individual_token_ids):
                        quote_to_token_positions[original_quote_text] = all_pos_lists
                        if DEBUG:
                            print(f"  ✅ Counting scenario successful. Positions: {all_pos_lists}")
                    else:
                        # 计数场景失败，记录为空列表
                        quote_to_token_positions[original_quote_text] = []
                        if DEBUG:
                            print(f"  ❌ Counting scenario failed for '{original_quote_text}', no positions found.")


        # ======================
        # 6. 最终结果打印
        # ======================
        if DEBUG:
            print("\n" + "="*60)
            print("✅ Final token positions for each quote:")
            print("="*60)
            for quote, positions in quote_to_token_positions.items():
                print(f"""
        Quote Text: "{quote}"
        Token Positions: {positions}
        Length: {len(positions)} tokens
        """)
        
        return quote_to_token_positions


    def _find_sublist_indices(self, sublist, main_list):
        """
        Helper function to find the starting and ending indices of a sublist within a main list.
        Returns a list of indices corresponding to the occurrences of the sublist.
        Returns the first occurrence's full index range.
        """
        if not sublist:
            return []
        indices = []
        sublist_len = len(sublist)
        for i in range(len(main_list) - sublist_len + 1):
            if main_list[i:i + sublist_len] == sublist:
                indices.extend(range(i, i + sublist_len))
                break # Return only the first occurrence's range
        return indices
    
    def aggregate_attention(
                        self,
                        attention_store: AttentionStore,
                        is_cross: bool,
                        shape: Tuple[int, int],
                        select: int # select 为什么要设置为0，需要探究？
                    ) -> torch.Tensor:
        """ Aggregates the attention across the different layers and heads at the specified resolution. """
        out = []
        attention_maps = attention_store.get_store_attention()
        # print(attention_maps)

        for item in attention_maps[f"{'cross' if is_cross else 'self'}"]:
            cross_maps = item.reshape(1, -1, int(shape[0]/shape[2]), int(shape[1]/shape[2]), item.shape[-1])[select]
            out.append(cross_maps)
        out = torch.cat(out, dim=0)
        out = out.sum(0) / out.shape[0]
        return out
    

    def _compute_max_attention_per_index(self,
                                        attention_maps: torch.Tensor,
                                        indices_to_alter: List[int],
                                        smooth_attentions: bool = False,
                                        shape: Optional[Tuple[int, int, int]] = None,
                                        sigma: float = 0.5,
                                        kernel_size: int = 3,
                                        bbox: List[int] = None,
                                        ) -> List[torch.Tensor]:
        """ Computes the maximum attention value for each of the tokens we wish to alter. """
        # QwenVL 截断了 eot token,第一个便是有效字符 -> 旧实现见 _compute_max_attention_per_index_old
        attention_for_text = attention_maps
        # attention_for_text = attention_for_text * 100
        # attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)
        # attention_for_text = attention_for_text * 100
        # attention_sum = attention_for_text.sum(dim=-1)

        # Extract the maximum values
        max_indices_list_fg = []
        # for fit the old implementation, define but not used
        max_indices_list_bg = []
        dist_x = []
        dist_y = []

        height, width, scale_factor = shape

        cnt = 0
        """
            情况不一样 -> boxdiff是一个框只有一个token描述,而我们一个框有多个token描述,此处有待改进
        """
        for cnt, indices in enumerate(indices_to_alter):
            image_list = []

            for i in indices:
                image = attention_for_text[:, :, i]
                image_list.append(image)

            image_mean = torch.stack(image_list).mean(dim=0)
            # 基于极大极小值进行归一化
            image_mean = (image_mean - image_mean.min()) / (image_mean.max() - image_mean.min() + + 1e-8)

            box = [max(round(b/scale_factor), 0) for b in bbox[cnt]]
            x1, y1, x2, y2 = box

            # coordinates to masks
            obj_mask = torch.zeros_like(image_mean)
            ones_mask = torch.ones([y2 - y1, x2 - x1], dtype=obj_mask.dtype).to(obj_mask.device)
            obj_mask[y1:y2, x1:x2] = ones_mask
            bg_mask = 1 - obj_mask

            if smooth_attentions:
                smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).cuda()
                input = F.pad(image_mean.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
                image = smoothing(input).squeeze(0).squeeze(0)

            # Inner-Box constraint
            inner_pix_num = obj_mask.sum()
            inner_pix_avg = (image_mean * obj_mask).sum() /inner_pix_num


            # Outer-Box constraint
            outer_pix_num = bg_mask.sum()
            outer_pix_avg = (image_mean * bg_mask).sum() /outer_pix_num

            diff = inner_pix_avg - outer_pix_avg
            loss = - torch.log(torch.sigmoid(diff) + 1e-8)

            max_indices_list_fg.append(loss)

        return max_indices_list_fg, max_indices_list_bg, dist_x, dist_y

            
    def _compute_max_attention_per_index_old(self,
                                        attention_maps: torch.Tensor,
                                        indices_to_alter: List[int],
                                        smooth_attentions: bool = False,
                                        shape: Optional[Tuple[int, int, int]] = None,
                                        sigma: float = 0.5,
                                        kernel_size: int = 3,
                                        bbox: List[int] = None,
                                        ) -> List[torch.Tensor]:
        """ Computes the maximum attention value for each of the tokens we wish to alter. """
        # last_idx = -1
        # if normalize_eot:
        #     prompt = self.prompt
        #     if isinstance(self.prompt, list):
        #         prompt = self.prompt[0]
        #     last_idx = len(self.tokenizer(prompt)['input_ids']) - 1
        # attention_for_text = attention_maps[:, :, 1:last_idx]
        # QwenVL 截断了 eot token,第一个便是有效字符
        attention_for_text = attention_maps
        attention_for_text = attention_for_text * 100
        attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

        # Shift indices since we removed the first token
        # indices_to_alter = [index - 1 for index in indices_to_alter]
        indices_to_alter = [[index - 1 for index in sublist] for sublist in indices_to_alter]

        # Extract the maximum values
        max_indices_list_fg = []
        max_indices_list_bg = []
        dist_x = []
        dist_y = []

        height, width, scale_factor = shape

        cnt = 0
        """
            情况不一样 -> boxdiff是一个框只有一个token描述,而我们一个框有多个token描述,此处有待改进
        """
        for cnt, indices in enumerate(indices_to_alter):

            for i in indices:
                image = attention_for_text[:, :, i]

                box = [max(round(b/scale_factor), 0) for b in bbox[cnt]]
                x1, y1, x2, y2 = box

                # coordinates to masks
                obj_mask = torch.zeros_like(image)
                ones_mask = torch.ones([y2 - y1, x2 - x1], dtype=obj_mask.dtype).to(obj_mask.device)
                obj_mask[y1:y2, x1:x2] = ones_mask
                bg_mask = 1 - obj_mask

                if smooth_attentions:
                    smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).cuda()
                    input = F.pad(image.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
                    image = smoothing(input).squeeze(0).squeeze(0)

                # Inner-Box constraint
                k = (obj_mask.sum() * boxConfig.P).long()
                max_indices_list_fg.append((image * obj_mask).reshape(-1).topk(k)[0].mean())

                # Outer-Box constraint
                k = (bg_mask.sum() * boxConfig.P).long()
                max_indices_list_bg.append((image * bg_mask).reshape(-1).topk(k)[0].mean())

                # Corner Constraint
                gt_proj_x = torch.max(obj_mask, dim=0)[0]
                gt_proj_y = torch.max(obj_mask, dim=1)[0]
                corner_mask_x = torch.zeros_like(gt_proj_x)
                corner_mask_y = torch.zeros_like(gt_proj_y)

                # create gt according to the number config.L
                N = gt_proj_x.shape[0]
                corner_mask_x[max(box[0] - boxConfig.L, 0): min(box[0] + boxConfig.L + 1, N)] = 1.
                corner_mask_x[max(box[2] - boxConfig.L, 0): min(box[2] + boxConfig.L + 1, N)] = 1.
                corner_mask_y[max(box[1] - boxConfig.L, 0): min(box[1] + boxConfig.L + 1, N)] = 1.
                corner_mask_y[max(box[3] - boxConfig.L, 0): min(box[3] + boxConfig.L + 1, N)] = 1.
                dist_x.append((F.l1_loss(image.max(dim=0)[0], gt_proj_x, reduction='none') * corner_mask_x).mean())
                dist_y.append((F.l1_loss(image.max(dim=1)[0], gt_proj_y, reduction='none') * corner_mask_y).mean())

        return max_indices_list_fg, max_indices_list_bg, dist_x, dist_y

    @cost_time
    def _aggregate_and_get_max_attention_per_token(self, 
                                                    attention_store: AttentionStore,
                                                    indices_to_alter:Dict[str, List[int]],
                                                    gaussian_smoothing_kwargs:Dict[str, Any],
                                                    shape: Tuple[int, int, int],
                                                    bbox:List[List[int]]
                                                        ):
        """ Aggregates the attention for each token and computes the max activation value for each token to alter. """
        attention_maps = self.aggregate_attention(
                attention_store=attention_store,
                shape=shape, 
                is_cross=True,
                select=0 # why 0
            )
        values_list = [indices_to_alter[key] for key in indices_to_alter]
        max_attention_per_index_fg, max_attention_per_index_bg, dist_x, dist_y = self._compute_max_attention_per_index(
            attention_maps=attention_maps,
            indices_to_alter=values_list,
            smooth_attentions=gaussian_smoothing_kwargs.get("smooth_attentions", False),
            sigma=gaussian_smoothing_kwargs.get("sigma", 0.5),
            kernel_size=gaussian_smoothing_kwargs.get("kernel_size", 5),
            shape=shape,
            bbox=bbox
        )
        return max_attention_per_index_fg, max_attention_per_index_bg, dist_x, dist_y
    

    @staticmethod
    @cost_time
    def _compute_loss_old(max_attention_per_index_fg: List[torch.Tensor], max_attention_per_index_bg: List[torch.Tensor],
                      dist_x: List[torch.Tensor], dist_y: List[torch.Tensor], return_losses: bool = False) -> torch.Tensor:
        """ Computes the attend-and-excite loss using the maximum attention value for each token. """
        losses_fg = [max(0, 1. - curr_max) for curr_max in max_attention_per_index_fg]
        losses_bg = [max(0, curr_max) for curr_max in max_attention_per_index_bg]
        loss = sum(losses_fg) + sum(losses_bg) + sum(dist_x) + sum(dist_y)
        print(f"losses_fg: {max(losses_fg)} loss_bg:{max(losses_bg)} , loss:{loss}")
        if return_losses:
            return max(losses_fg), losses_fg
        else:
            return max(losses_fg), loss
        
    def _compute_loss(self, max_attention_per_index_fg: List[torch.Tensor], max_attention_per_index_bg: List[torch.Tensor],
                      dist_x: List[torch.Tensor], dist_y: List[torch.Tensor], return_losses: bool = False) -> torch.Tensor:
        """ Computes the attend-and-excite loss using the maximum attention value for each token. """
        loss = sum(max_attention_per_index_fg)
        print(f"loss: {loss}")
        return loss, loss
    
    import torch

    @staticmethod
    def _adjust_and_clip_tensor(grad_cond, target_min=0.01, target_max=0.1, clip_range=(-1.0, 1.0)):
        """
        调整矩阵 grad_cond：
        1. 缩放使其平均绝对值落在 [target_min, target_max] 范围内
        2. 裁剪到 [clip_min, clip_max]

        Args:
            grad_cond: Tensor 或 numpy array，任意形状的梯度矩阵
            target_min: 目标最小平均绝对值
            target_max: 目标最大平均绝对值
            clip_range: 裁剪范围 (min, max)

        Returns:
            scaled_clipped: 调整并裁剪后的矩阵
        """
        # Step 1: 计算当前平均绝对值
        current_avg = grad_cond.abs().mean().item()

        if current_avg < 1e-12:  # 防止除以零
            scale_factor = 1.0
        else:
            # 在 [target_min, target_max] 中选择一个目标值（可取平均值或随机）
            target_avg = (target_min + target_max) / 2  # 取中值，如 0.055
            # 或者随机选择：target_avg = torch.rand(1).item() * (target_max - target_min) + target_min
            scale_factor = target_avg / current_avg

        # Step 2: 缩放
        scaled = grad_cond * scale_factor

        # Step 3: 裁剪
        clip_min, clip_max = clip_range
        scaled_clipped = scaled.clamp(clip_min, clip_max)

        return scaled_clipped, scale_factor
    @staticmethod
    @cost_time
    def _update_latent(latents: torch.Tensor, loss: torch.Tensor, loss_list: list[torch.Tensor] ,step_size: float) -> torch.Tensor:
        """ Update the latent according to the computed loss. """
        if boxConfig.use_global_box_loss:
            grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
            # 1. 全局归一化（最推荐）
            # grad_cond = grad_cond / (grad_cond.norm() + 1e-8)
            # 2. 最大最小值归一化
            # grad_cond = (grad_cond - grad_cond.min()) / (grad_cond.max() - grad_cond.min() + 1e-8)
            # 3. 均值归一化 -> 这种归一化能放大梯度，同时不改变梯度的分布（核心思想：不改变梯度的比例系数、分布的情况下，合理的放大梯度）
            grad_cond = (grad_cond - grad_cond.mean()) / (grad_cond.std() + 1e-8)
            latents = latents - step_size * grad_cond
            if boxConfig.latents_gaussian:
                with torch.no_grad():
                    # latents = (latents - latents.mean()) / (latents.std() + 1e-8)
                    latents_old = latents.clone().detach()
                    # print(latents_old.min().item(), latents_old.max().item())
                    # latents_new = histogram_matching(latents_old, device=latents.device, dtype = latents.dtype)
                    latents_new = process_matrix_with_noise(latents_old)
                    latents_new.clone().detach().requires_grad_(True)
                    # print(latents_new.min().item(), latents_new.max().item())
                    return latents_new
        elif boxConfig.scale_grad == "max":
            grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
            # 1. 获取极值
            max_val = grad_cond.max().item()  # 3.0
            min_val = grad_cond.min().item()  # -1.2
            # 2. 设定目标最大值（在 0.01 ~ 0.1 之间）
            target_max = random.uniform(0.01, 0.1)  # 如 0.07
            scale_factor = target_max / max_val
            # 3. 缩放
            grad_cond_scaled = grad_cond * scale_factor
            latents = latents - step_size * grad_cond_scaled
            # 4. 输出结果
            print(f"Original  - Max: {max_val:.4f}, Min: {min_val:.4f}")
            print(f"Target Max: {target_max:.4f}")
            print(f"Scale Factor: {scale_factor:.6f}")
            print(f"Scaled    - Max: {grad_cond_scaled.max().item():.4f}, Min: {grad_cond_scaled.min().item():.4f}")
        elif boxConfig.scale_grad == "mean":
            grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
            scaled_clipped, scale_factor = RegionalQwenImagePipeline._adjust_and_clip_tensor(grad_cond)
            print(f"Scale Factor: {scale_factor:.6f}")
            latents = latents - step_size * scaled_clipped
        else:
            # 1.针对每一个box的loss计算梯度
            grad_conds = []
            for loss_fg in loss_list:
                grad_cond = torch.autograd.grad(loss_fg.requires_grad_(True), [latents], retain_graph=True)[0]
                grad_conds.append(grad_cond)  
            # 2.求梯度平均值
            grad_cond = sum(grad_conds) / len(grad_conds)
            # 3. 均值归一化 -> 这种归一化能放大梯度，同时不改变梯度的分布（核心思想：不改变梯度的比例系数、分布的情况下，合理的放大梯度）
            grad_cond = (grad_cond - grad_cond.mean()) / (grad_cond.std() + 1e-8)
            latents = latents - step_size * grad_cond
            if boxConfig.latents_gaussian:
                with torch.no_grad():
                    # latents = (latents - latents.mean()) / (latents.std() + 1e-8)
                    latents_old = latents.clone().detach()
                    # print(latents_old.min().item(), latents_old.max().item())
                    # latents_new = histogram_matching(latents_old, device=latents.device, dtype = latents.dtype)
                    latents_new = process_matrix_with_noise(latents_old)
                    latents_new.clone().detach().requires_grad_(True)
                    # print(latents_new.min().item(), latents_new.max().item())
                    return latents_new
        return latents
        

    def _perform_iterative_refinement_step(self,
                                           latents: torch.Tensor,
                                           encoder_hidden_states_mask: torch.Tensor,
                                           encoder_hidden_states: torch.Tensor,
                                           img_shapes: Tuple,
                                           regional_txt_seq_lens: int,
                                           indices_to_alter: List[int],
                                           loss_fg: torch.Tensor,
                                           threshold: float,
                                           attention_store: AttentionStore,
                                           step_size: float,
                                           timestep: torch.Tensor,
                                           guidance: torch.Tensor,
                                           max_refinement_steps: int = 20,
                                           loss_util: LossUtil = None
                                           ):
        """
        Performs the iterative latent refinement introduced in the paper. Here, we continuously update the latent
        code according to our loss objective until the given threshold is reached for all tokens.
        """
        if max_refinement_steps==0:
            return latents
        
        iteration = 0
        # target_loss = max(0, 1. - threshold)
        target_loss = 0
        while loss_fg > target_loss:
            iteration += 1

            latents = latents.clone().detach().requires_grad_(True)
            self.set_train_transform_layer(boxConfig.train_layer)
            noise_pred_text = self.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states_mask=encoder_hidden_states_mask,
                        encoder_hidden_states=encoder_hidden_states,
                        img_shapes=img_shapes,
                        # txt_seq_lens=negative_txt_seq_lens,
                        regional_txt_seq_lens=regional_txt_seq_lens,
                        attention_kwargs=self.attention_kwargs,
                        return_dict=False,
                    )[0]
            self.transformer.zero_grad()

            # Get max activation value for each subject token
            # max_attention_per_index_fg, max_attention_per_index_bg, dist_x, dist_y = self._aggregate_and_get_max_attention_per_token(
            #     attention_store=attention_store,
            #     indices_to_alter=indices_to_alter,
            #     gaussian_smoothing_kwargs=self._gaussian_smoothing_kwargs,
            #     shape = (self._height ,self._width ,self.vae_scale_factor*2),
            #     bbox=self.attention_kwargs.get("regional_boxes")
            # )

            # loss_fg, losses_fg = self._compute_loss(max_attention_per_index_fg, max_attention_per_index_bg, dist_x, dist_y, return_losses=True)

            if boxConfig.lossType == "diff":
                loss_fg, loss_list = compute_diff_loss(
                    attention_store=attention_store,
                    indices_to_alter=indices_to_alter,
                    gaussian_smoothing_kwargs=self._gaussian_smoothing_kwargs,
                    shape = (self._height ,self._width ,self.vae_scale_factor*2),
                    bbox=self.attention_kwargs.get("regional_boxes"),
                    child_bbox = self.attention_kwargs.get("regional_child_boxes")
                )
            elif boxConfig.lossType == "rnb":
                loss_fg, loss_list = compute_rnb_loss(
                    attention_store=attention_store,
                    indices_to_alter=indices_to_alter,
                    gaussian_smoothing_kwargs=self._gaussian_smoothing_kwargs,
                    shape = (self._height ,self._width ,self.vae_scale_factor*2),
                    bbox=self.attention_kwargs.get("regional_boxes"),
                    child_bbox = self.attention_kwargs.get("regional_child_boxes"),
                    loss_util=loss_util
                )
            elif boxConfig.lossType == "opt":
                loss_fg, loss_list = compute_opt_loss(
                    attention_store=attention_store,
                    indices_to_alter=indices_to_alter,
                    gaussian_smoothing_kwargs=self._gaussian_smoothing_kwargs,
                    shape = (self._height ,self._width ,self.vae_scale_factor*2),
                    bbox=self.attention_kwargs.get("regional_boxes"),
                    child_bbox = self.attention_kwargs.get("regional_child_boxes")
                )

            if loss_fg != 0:  # 此处算出来的梯度特别小，导致loss_fg基本没更新
                latents = self._update_latent(latents, loss_fg, loss_list, step_size)

        
            if iteration >= max_refinement_steps:
                print(f'\t Exceeded max number of iterations ({max_refinement_steps})! ')
                break
        return latents
    
    @cost_time
    def set_train_transform_layer(self, train_transform_layer):
        for name, param in self.transformer.named_parameters():
            param.requires_grad = False
            if 'transformer_blocks' in name:
                layer_index = int(name.split(".")[1])
                if layer_index in train_transform_layer:
                    param.requires_grad = True

    def _setup_dynamic_masking(
        self,
        hidden_seq_len: int,
        regional_embeds: torch.Tensor,
        masks: List[torch.Tensor],
        each_prompt_seq_len: List[int],
        H_z: int,
        W_z: int,
    ):
        """
        Initialize DynamicMaskBuilder and store it in all attention processors.
        This allows each attention layer to build soft masks from current features.

        Args:
            hidden_seq_len: L — number of image patch tokens
            regional_embeds: Concatenated regional text embeddings [1, M, d]
            masks: List of per-region spatial masks [L, seq_len_i]
            each_prompt_seq_len: Text sequence length per region
            H_z: Latent height in patches
            W_z: Latent width in patches
        """
        if not boxConfig.use_dynamic_mask:
            return

        encoder_seq_len = regional_embeds.shape[1]

        # Convert regional boxes from pixel to latent space
        regional_boxes_latent = []
        if hasattr(self, '_attention_kwargs') and self._attention_kwargs is not None:
            regional_boxes_pixel = self._attention_kwargs.get("regional_boxes", [])
            scale_factor = self.vae_scale_factor * 2
            for bbox in regional_boxes_pixel:
                x1, y1, x2, y2 = bbox
                regional_boxes_latent.append([
                    max(round(x1 / scale_factor), 0),
                    max(round(y1 / scale_factor), 0),
                    min(round(x2 / scale_factor), W_z),
                    min(round(y2 / scale_factor), H_z),
                ])

        builder = DynamicMaskBuilder(
            hidden_seq_len=hidden_seq_len,
            encoder_seq_len=encoder_seq_len,
            regional_masks=masks,
            each_prompt_seq_len=each_prompt_seq_len,
            regional_boxes=regional_boxes_latent if regional_boxes_latent else [[0, 0, W_z, H_z]],
            H_z=H_z,
            W_z=W_z,
        )

        # Store builder in all attention processors
        for name, module in self.transformer.named_modules():
            if hasattr(module, 'processor'):
                processor = module.get_processor()
                if isinstance(processor, RegionalQwenImageAttnProcessor):
                    processor._dynamic_mask_builder = builder




    """
        @torch.inference_mode() 是 PyTorch 提供的一个 装饰器（decorator），
        用于将函数或方法标记为“推理模式”（inference mode），即仅用于模型前向传播（forward pass），
        不进行梯度计算，也不构建计算图。
        它是 torch.no_grad() 的更严格、更高效的版本，专为推理（inference）场景设计。
    """
    # @torch.inference_mode() 会抑制梯度计算，所以此处用@torch.no_grad()
    # @torch.inference_mode()
    @torch.no_grad()
    def __call__(
        self,
        base_prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        attention_store: AttentionStore = None,
        true_cfg_scale: float = 4.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        mask_inject_steps: int = 5,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 1.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        gaussian_smoothing_kwargs: Optional[Dict[str, Any]] = None, # added ,用于attention map 的高斯平滑
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is
                not greater than `1`).
            true_cfg_scale (`float`, *optional*, defaults to 1.0):
                When > 1.0 and a provided `negative_prompt`, enables true classifier-free guidance.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 3.5):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.qwenimage.QwenImagePipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.qwenimage.QwenImagePipelineOutput`] or `tuple`:
            [`~pipelines.qwenimage.QwenImagePipelineOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is a list with the generated images.
        """

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            base_prompt,
            height,
            width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False
        self._gaussian_smoothing_kwargs = gaussian_smoothing_kwargs if gaussian_smoothing_kwargs is not None else {}
        self._height = height
        self._width = width

        # get quote prompt token index
        if isinstance(base_prompt, str):
            quote_to_token_positions = self.get_token_index_v3(base_prompt, quote_prompt=True, region_prompts = attention_kwargs.get("regional_prompts", None))
            print("🔗 引号内容:", base_prompt)
            print("🔗 引号内容对应的 token 位置:", quote_to_token_positions)
        else:
            quote_to_token_positions = None

        # 2. Define call parameters
        if base_prompt is not None and isinstance(base_prompt, str):
            batch_size = 1
        elif base_prompt is not None and isinstance(base_prompt, list):
            batch_size = len(base_prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        prompt_embeds, prompt_embeds_mask = self.encode_prompt( # 在只输入文本的情况下，可以看成是使用Qwen-7B-Chat进行文本编码
            prompt=base_prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        boxConfig.text_len = prompt_embeds.shape[1]
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )

        # added: define base mask and inputs
        # base_mask = torch.ones((height, width), device=device, dtype=self.transformer.dtype) # base mask uses the whole image mask
        # base_inputs = [(base_mask, prompt_embeds)]

        # added: encode regional prompts,define regional inputs
        regional_inputs = []
        if 'regional_prompts' in attention_kwargs and 'regional_masks' in attention_kwargs:
            for regional_prompt, regional_mask in zip(attention_kwargs['regional_prompts'], attention_kwargs['regional_masks']):
                regional_prompt_embeds, regional_prompt_embeds_masks = self.encode_prompt(
                    prompt=regional_prompt,
                    prompt_embeds=None,
                    prompt_embeds_mask=None,
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=max_sequence_length
                )
                regional_inputs.append((regional_mask, regional_prompt_embeds, regional_prompt_embeds_masks))

        ## added: prepare masks for regional control
        conds = []
        cond_masks = []
        masks = []
        each_prompt_seq_len = [] 
        H, W = height//(self.vae_scale_factor)//2, width//(self.vae_scale_factor)//2
        hidden_seq_len = H * W

        # prepare base ration regional masks
        if attention_kwargs is not None and attention_kwargs["enable_whole_regional_mask"]:
            attention_kwargs["whole_regional_mask"] = torch.nn.functional.interpolate(attention_kwargs["whole_regional_mask"][None, None, :, :], (H, W), mode='nearest-exact').flatten().unsqueeze(1)
        
        for mask, cond, cond_mask in regional_inputs:
            if mask is not None: # resize regional masks to image size, the flatten is to match the seq len
                mask = torch.nn.functional.interpolate(mask[None, None, :, :], (H, W), mode='nearest-exact').flatten().unsqueeze(1).repeat(1, cond.size(1))
            else:
                mask = torch.ones((H*W, cond.size(1))).to(device=cond.device)
            masks.append(mask)
            conds.append(cond)
            cond_masks.append(cond_mask)
            each_prompt_seq_len.append(cond.shape[1])
        regional_embeds = torch.cat(conds, dim=1)
        regional_embeds_mask = torch.cat(cond_masks, dim=1)
        encoder_seq_len = regional_embeds.shape[1]

        # initialize attention mask
        regional_attention_mask = torch.zeros(
            (encoder_seq_len + hidden_seq_len, encoder_seq_len + hidden_seq_len),
            device=masks[0].device,
            dtype=torch.bool
        )
        num_of_regions = len(masks)

        # initialize self-attended mask
        self_attend_masks = torch.zeros((hidden_seq_len, hidden_seq_len), device=masks[0].device, dtype=torch.bool)

        # initialize union mask
        union_masks = torch.zeros((hidden_seq_len, hidden_seq_len), device=masks[0].device, dtype=torch.bool)

        # handle each mask
        seq_len_begin = 0
        seq_len_end = 0
        for i in range(num_of_regions):
            # caculate the begin and end of the current region
            seq_len_begin = seq_len_end
            seq_len_end = seq_len_begin + each_prompt_seq_len[i]

            # txt attends to itself
            # regional_attention_mask[i*each_prompt_seq_len:(i+1)*each_prompt_seq_len, i*each_prompt_seq_len:(i+1)*each_prompt_seq_len] = True
            regional_attention_mask[seq_len_begin:seq_len_end, seq_len_begin:seq_len_end] = True

            # txt attends to corresponding regional img
            # regional_attention_mask[i*each_prompt_seq_len:(i+1)*each_prompt_seq_len, encoder_seq_len:] = masks[i].transpose(-1, -2)
            regional_attention_mask[seq_len_begin:seq_len_end, encoder_seq_len:] = masks[i].transpose(-1, -2)

            # regional img attends to corresponding txt
            # regional_attention_mask[encoder_seq_len:, i*each_prompt_seq_len:(i+1)*each_prompt_seq_len] = masks[i]
            regional_attention_mask[encoder_seq_len:, seq_len_begin:seq_len_end] = masks[i]

            # regional img attends to corresponding regional img
            img_size_masks = masks[i][:, :1].repeat(1, hidden_seq_len)
            img_size_masks_transpose = img_size_masks.transpose(-1, -2)
            self_attend_masks = torch.logical_or(self_attend_masks, 
                                                    torch.logical_and(img_size_masks, img_size_masks_transpose))

            # update union
            union_masks = torch.logical_or(union_masks, 
                                            torch.logical_or(img_size_masks, img_size_masks_transpose))

        background_masks = torch.logical_not(union_masks)

        background_and_self_attend_masks = torch.logical_or(background_masks, self_attend_masks)

        regional_attention_mask[encoder_seq_len:, encoder_seq_len:] = background_and_self_attend_masks
        ## added : done prepare masks for regional control

        # Setup dynamic soft masking if enabled (for __call__)
        self._setup_dynamic_masking(
            hidden_seq_len=hidden_seq_len,
            regional_embeds=regional_embeds,
            masks=masks,
            each_prompt_seq_len=each_prompt_seq_len,
            H_z=H,
            W_z=W,
        )


        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        img_shapes = [(1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2)] * batch_size

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # scale_range = np.linspace(boxConfig.scale_range[0], boxConfig.scale_range[1], self._num_timesteps)
        scale_range = boxConfig.scale_range_value

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if self.attention_kwargs is None:
            self._attention_kwargs = {}

        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
        negative_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist() if negative_prompt_embeds_mask is not None else None
        )
        regional_txt_seq_lens = regional_embeds_mask.sum(dim=1).tolist() if regional_embeds_mask is not None else None

        # handle infer attention mask
        regional_attention_mask = regional_attention_mask.to(device)
        infer_attention_kwargs = {
            'double_inject_blocks_interval': attention_kwargs['double_inject_blocks_interval'] if 'double_inject_blocks_interval' in attention_kwargs else len(self.transformer.transformer_blocks),
            "whole_regional_mask": attention_kwargs["whole_regional_mask"].to(device).to(latents.dtype),  # 操作是否局限在mask内
            "enable_whole_regional_mask": attention_kwargs["enable_whole_regional_mask"]
        }

        # add some args for visualization
        boxConfig.text_index = quote_to_token_positions
        boxConfig.bbox = attention_kwargs.get("regional_boxes")

        # LossUtil 初始化
        loss_names = boxConfig.text_index.keys()
        loss_util = LossUtil(loss_names, total_weight=boxConfig.total_weight)


        # 6. Denoising loop
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                boxConfig.now_step = i

                self._current_timestep = t
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                # 基于局部梯度更新latents，使得初始latents的布局更符合区域提示的要求
                if i in boxConfig.max_iter_to_alter:
                    boxConfig.switch_box_loss = True
                    with torch.enable_grad():
                        # 在训练循环开始前启用
                        # torch.autograd.set_detect_anomaly(True)
                        latents = latents.clone().detach().requires_grad_(True)

                        # train all layers has no such big memory cost
                        self.set_train_transform_layer(boxConfig.train_layer)

                        # Forward pass of denoising with text conditioning
                        noise_pred_text = self.transformer(
                            hidden_states=latents,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            encoder_hidden_states_mask=prompt_embeds_mask,
                            encoder_hidden_states=prompt_embeds,
                            img_shapes=img_shapes,
                            # txt_seq_lens=negative_txt_seq_lens,
                            regional_txt_seq_lens=txt_seq_lens,
                            attention_kwargs=self.attention_kwargs,
                            return_dict=False,
                        )[0]

                        self.transformer.zero_grad()

                        # Perform gradient update # 此处是几个局部的差值算了一个总的loss,然后该loss应用于全局，这样是否合理？存在问题，需要改进
                        if i in boxConfig.max_iter_to_alter:
                            # Get max activation value for each subject token
                            # max_attention_per_index_fg, max_attention_per_index_bg, dist_x, dist_y = self._aggregate_and_get_max_attention_per_token(
                            #     attention_store=attention_store,
                            #     indices_to_alter=quote_to_token_positions,
                            #     gaussian_smoothing_kwargs=gaussian_smoothing_kwargs,
                            #     shape = (height,width,self.vae_scale_factor*2),
                            #     bbox=attention_kwargs.get("regional_boxes")
                            # )

                            # loss_fg, loss = self._compute_loss(max_attention_per_index_fg, max_attention_per_index_bg, dist_x, dist_y)
                            if boxConfig.lossType == "diff":
                                loss_fg, loss_list = compute_diff_loss(
                                    attention_store=attention_store,
                                    indices_to_alter=quote_to_token_positions,
                                    gaussian_smoothing_kwargs=gaussian_smoothing_kwargs,
                                    shape = (height,width,self.vae_scale_factor*2),
                                    bbox= attention_kwargs.get("regional_boxes"),
                                    child_bbox = attention_kwargs.get("regional_child_boxes") # 不存在则返回None
                                )
                            elif boxConfig.lossType == "rnb":
                                loss_fg, loss_list = compute_rnb_loss(
                                    attention_store=attention_store,
                                    indices_to_alter=quote_to_token_positions,
                                    gaussian_smoothing_kwargs=gaussian_smoothing_kwargs,
                                    shape = (height,width,self.vae_scale_factor*2),
                                    bbox=attention_kwargs.get("regional_boxes"),
                                    child_bbox = attention_kwargs.get("regional_child_boxes"), # 不存在则返回None
                                    loss_util=loss_util
                                )
                            elif boxConfig.lossType == "opt":
                                loss_fg, loss_list = compute_opt_loss(
                                    attention_store=attention_store,
                                    indices_to_alter=quote_to_token_positions,
                                    gaussian_smoothing_kwargs=gaussian_smoothing_kwargs,
                                    shape = (height,width,self.vae_scale_factor*2),
                                    bbox= attention_kwargs.get("regional_boxes"),
                                    child_bbox = attention_kwargs.get("regional_child_boxes") # 不存在则返回None
                                )
                            if loss_fg != 0:
                                latents = self._update_latent(latents=latents, loss=loss_fg, loss_list = loss_list, # 原实现此处用loss
                                                                step_size=boxConfig.scale_factor * scale_range[i])

                            # Refinement from attend-and-excite (not necessary)
                            if True:

                                # loss_fg, loss = self._compute_loss(max_attention_per_index_fg, max_attention_per_index_bg, dist_x, dist_y)

                                # if i in boxConfig.thresholds.keys() and loss_fg > 1. - boxConfig.thresholds.get(i) and boxConfig.refine:
                                if boxConfig.refine:
                                    del noise_pred_text
                                    torch.cuda.empty_cache()
                                    latents = self._perform_iterative_refinement_step(
                                        latents=latents,
                                        encoder_hidden_states_mask=prompt_embeds_mask,
                                        encoder_hidden_states=prompt_embeds,
                                        img_shapes=img_shapes,
                                        regional_txt_seq_lens=txt_seq_lens,
                                        indices_to_alter=quote_to_token_positions,
                                        loss_fg=loss_fg,
                                        threshold=boxConfig.thresholds.get(i),
                                        attention_store=attention_store,
                                        step_size= boxConfig.scale_factor * scale_range[i],
                                        timestep=timestep,
                                        guidance=guidance,  # use a larger guidance scale for refinement
                                        max_refinement_steps=boxConfig.max_refinement_steps.get(i,boxConfig.default_value),
                                        loss_util=loss_util,

                                    )
                        
                        
                    boxConfig.switch_box_loss = False

                       



                if self.interrupt:
                    continue

                if i < mask_inject_steps:
                    chosen_prompt_embeds = regional_embeds
                    chosen_prompt_embeds_mask = regional_embeds_mask
                    base_ratio = attention_kwargs['base_ratio']
                    infer_attention_kwargs["regional_attention_mask"] = regional_attention_mask
                else:
                    chosen_prompt_embeds = prompt_embeds
                    chosen_prompt_embeds_mask = prompt_embeds_mask
                    regional_txt_seq_lens = txt_seq_lens
                    base_ratio = None

                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states_mask=chosen_prompt_embeds_mask,
                        encoder_hidden_states=chosen_prompt_embeds, # regional_embeds or base prompt_embeds -> change
                        encoder_hidden_states_base_mask=prompt_embeds_mask, # base prompt mask -> add
                        encoder_hidden_states_base=prompt_embeds, # base prompt embeds -> add
                        base_ratio=base_ratio, # base ratio for regional control -> add
                        img_shapes=img_shapes,
                        txt_seq_lens=txt_seq_lens,
                        regional_txt_seq_lens=regional_txt_seq_lens,
                        attention_kwargs=infer_attention_kwargs,
                        return_dict=False,
                    )[0]

                if do_true_cfg:
                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred = self.transformer(
                            hidden_states=latents,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            encoder_hidden_states_mask=negative_prompt_embeds_mask,
                            encoder_hidden_states=negative_prompt_embeds,
                            img_shapes=img_shapes,
                            # txt_seq_lens=negative_txt_seq_lens,
                            regional_txt_seq_lens=negative_txt_seq_lens,
                            attention_kwargs=self.attention_kwargs,
                            return_dict=False,
                        )[0]
                    comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                    noise_pred = comb_pred * (cond_norm / noise_norm)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                    
                if boxConfig.visual_middle_res:
                    visualize_latent_map(self, latents.clone().detach(), height, width, i)

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return QwenImagePipelineOutput(images=image)
    


    @torch.no_grad()
    def latents_choose(
        self,
        base_prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        attention_store: AttentionStore = None,
        true_cfg_scale: float = 4.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        mask_inject_steps: int = 5,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 1.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        gaussian_smoothing_kwargs: Optional[Dict[str, Any]] = None, # added ,用于attention map 的高斯平滑
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            base_prompt,
            height,
            width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False
        self._gaussian_smoothing_kwargs = gaussian_smoothing_kwargs if gaussian_smoothing_kwargs is not None else {}
        self._height = height
        self._width = width

        # get quote prompt token index
        if isinstance(base_prompt, str) and '"' in base_prompt:
            quote_to_token_positions = self.get_token_index(base_prompt, quote_prompt=True, region_prompts = attention_kwargs.get("regional_prompts", None)[:-1])
            # print("🔗 引号内容对应的 token 位置:", quote_to_token_positions)
        else:
            quote_to_token_positions = None

        # 2. Define call parameters
        if base_prompt is not None and isinstance(base_prompt, str):
            batch_size = 1
        elif base_prompt is not None and isinstance(base_prompt, list):
            batch_size = len(base_prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        prompt_embeds, prompt_embeds_mask = self.encode_prompt( # 在只输入文本的情况下，可以看成是使用Qwen-7B-Chat进行文本编码
            prompt=base_prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        boxConfig.text_len = prompt_embeds.shape[1]
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )

        # added: define base mask and inputs
        # base_mask = torch.ones((height, width), device=device, dtype=self.transformer.dtype) # base mask uses the whole image mask
        # base_inputs = [(base_mask, prompt_embeds)]

        # added: encode regional prompts,define regional inputs
        regional_inputs = []
        if 'regional_prompts' in attention_kwargs and 'regional_masks' in attention_kwargs:
            for regional_prompt, regional_mask in zip(attention_kwargs['regional_prompts'], attention_kwargs['regional_masks']):
                regional_prompt_embeds, regional_prompt_embeds_masks = self.encode_prompt(
                    prompt=regional_prompt,
                    prompt_embeds=None,
                    prompt_embeds_mask=None,
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=max_sequence_length
                )
                regional_inputs.append((regional_mask, regional_prompt_embeds, regional_prompt_embeds_masks))

        ## added: prepare masks for regional control
        conds = []
        cond_masks = []
        masks = []
        each_prompt_seq_len = [] 
        H, W = height//(self.vae_scale_factor)//2, width//(self.vae_scale_factor)//2
        hidden_seq_len = H * W

        # prepare base ration regional masks
        if attention_kwargs is not None and attention_kwargs["enable_whole_regional_mask"]:
            attention_kwargs["whole_regional_mask"] = torch.nn.functional.interpolate(attention_kwargs["whole_regional_mask"][None, None, :, :], (H, W), mode='nearest-exact').flatten().unsqueeze(1)
        
        for mask, cond, cond_mask in regional_inputs:
            if mask is not None: # resize regional masks to image size, the flatten is to match the seq len
                mask = torch.nn.functional.interpolate(mask[None, None, :, :], (H, W), mode='nearest-exact').flatten().unsqueeze(1).repeat(1, cond.size(1))
            else:
                mask = torch.ones((H*W, cond.size(1))).to(device=cond.device)
            masks.append(mask)
            conds.append(cond)
            cond_masks.append(cond_mask)
            each_prompt_seq_len.append(cond.shape[1])
        regional_embeds = torch.cat(conds, dim=1)
        regional_embeds_mask = torch.cat(cond_masks, dim=1)
        encoder_seq_len = regional_embeds.shape[1]

        # initialize attention mask
        regional_attention_mask = torch.zeros(
            (encoder_seq_len + hidden_seq_len, encoder_seq_len + hidden_seq_len),
            device=masks[0].device,
            dtype=torch.bool
        )
        num_of_regions = len(masks)

        # initialize self-attended mask
        self_attend_masks = torch.zeros((hidden_seq_len, hidden_seq_len), device=masks[0].device, dtype=torch.bool)

        # initialize union mask
        union_masks = torch.zeros((hidden_seq_len, hidden_seq_len), device=masks[0].device, dtype=torch.bool)

        # handle each mask
        seq_len_begin = 0
        seq_len_end = 0
        for i in range(num_of_regions):
            # caculate the begin and end of the current region
            seq_len_begin = seq_len_end
            seq_len_end = seq_len_begin + each_prompt_seq_len[i]

            # txt attends to itself
            # regional_attention_mask[i*each_prompt_seq_len:(i+1)*each_prompt_seq_len, i*each_prompt_seq_len:(i+1)*each_prompt_seq_len] = True
            regional_attention_mask[seq_len_begin:seq_len_end, seq_len_begin:seq_len_end] = True

            # txt attends to corresponding regional img
            # regional_attention_mask[i*each_prompt_seq_len:(i+1)*each_prompt_seq_len, encoder_seq_len:] = masks[i].transpose(-1, -2)
            regional_attention_mask[seq_len_begin:seq_len_end, encoder_seq_len:] = masks[i].transpose(-1, -2)

            # regional img attends to corresponding txt
            # regional_attention_mask[encoder_seq_len:, i*each_prompt_seq_len:(i+1)*each_prompt_seq_len] = masks[i]
            regional_attention_mask[encoder_seq_len:, seq_len_begin:seq_len_end] = masks[i]

            # regional img attends to corresponding regional img
            img_size_masks = masks[i][:, :1].repeat(1, hidden_seq_len)
            img_size_masks_transpose = img_size_masks.transpose(-1, -2)
            self_attend_masks = torch.logical_or(self_attend_masks, 
                                                    torch.logical_and(img_size_masks, img_size_masks_transpose))

            # update union
            union_masks = torch.logical_or(union_masks, 
                                            torch.logical_or(img_size_masks, img_size_masks_transpose))

        background_masks = torch.logical_not(union_masks)

        background_and_self_attend_masks = torch.logical_or(background_masks, self_attend_masks)

        regional_attention_mask[encoder_seq_len:, encoder_seq_len:] = background_and_self_attend_masks
        ## added : done prepare masks for regional control


        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        img_shapes = [(1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2)] * batch_size

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # scale_range = np.linspace(boxConfig.scale_range[0], boxConfig.scale_range[1], self._num_timesteps)
        scale_range = boxConfig.scale_range_value

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if self.attention_kwargs is None:
            self._attention_kwargs = {}

        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
        negative_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist() if negative_prompt_embeds_mask is not None else None
        )
        regional_txt_seq_lens = regional_embeds_mask.sum(dim=1).tolist() if regional_embeds_mask is not None else None

        # handle infer attention mask
        regional_attention_mask = regional_attention_mask.to(device)
        infer_attention_kwargs = {
            'double_inject_blocks_interval': attention_kwargs['double_inject_blocks_interval'] if 'double_inject_blocks_interval' in attention_kwargs else len(self.transformer.transformer_blocks),
            "whole_regional_mask": attention_kwargs["whole_regional_mask"].to(device).to(latents.dtype),  # 操作是否局限在mask内
            "enable_whole_regional_mask": attention_kwargs["enable_whole_regional_mask"]
        }

        # add some args for visualization
        boxConfig.text_index = quote_to_token_positions
        boxConfig.bbox = attention_kwargs.get("regional_boxes")

        # LossUtil 初始化
        loss_names = boxConfig.text_index.keys()
        loss_util = LossUtil(loss_names, total_weight=boxConfig.total_weight)


        # 6. Denoising loop
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                boxConfig.now_step = i

                self._current_timestep = t
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                # 基于局部梯度更新latents，使得初始latents的布局更符合区域提示的要求
                if i in boxConfig.max_iter_to_alter:
                    boxConfig.switch_box_loss = True
                    with torch.enable_grad():
                        # 在训练循环开始前启用
                        # torch.autograd.set_detect_anomaly(True)
                        latents = latents.clone().detach().requires_grad_(True)

                        # train all layers has no such big memory cost
                        self.set_train_transform_layer(boxConfig.train_layer)

                        # Forward pass of denoising with text conditioning
                        noise_pred_text = self.transformer(
                            hidden_states=latents,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            encoder_hidden_states_mask=prompt_embeds_mask,
                            encoder_hidden_states=prompt_embeds,
                            img_shapes=img_shapes,
                            # txt_seq_lens=negative_txt_seq_lens,
                            regional_txt_seq_lens=txt_seq_lens,
                            attention_kwargs=self.attention_kwargs,
                            return_dict=False,
                        )[0]

                        self.transformer.zero_grad()

                        # Perform gradient update # 此处是几个局部的差值算了一个总的loss,然后该loss应用于全局，这样是否合理？存在问题，需要改进
                        if i in boxConfig.max_iter_to_alter:
                            # Get max activation value for each subject token
                            # max_attention_per_index_fg, max_attention_per_index_bg, dist_x, dist_y = self._aggregate_and_get_max_attention_per_token(
                            #     attention_store=attention_store,
                            #     indices_to_alter=quote_to_token_positions,
                            #     gaussian_smoothing_kwargs=gaussian_smoothing_kwargs,
                            #     shape = (height,width,self.vae_scale_factor*2),
                            #     bbox=attention_kwargs.get("regional_boxes")
                            # )

                            # loss_fg, loss = self._compute_loss(max_attention_per_index_fg, max_attention_per_index_bg, dist_x, dist_y)
                            if boxConfig.lossType == "diff":
                                loss_fg, loss_list = compute_diff_loss(
                                    attention_store=attention_store,
                                    indices_to_alter=quote_to_token_positions,
                                    gaussian_smoothing_kwargs=gaussian_smoothing_kwargs,
                                    shape = (height,width,self.vae_scale_factor*2),
                                    bbox= attention_kwargs.get("regional_boxes"),
                                    child_bbox = attention_kwargs.get("regional_child_boxes") # 不存在则返回None
                                )
                            elif boxConfig.lossType == "rnb":
                                loss_fg, loss_list = compute_rnb_loss(
                                    attention_store=attention_store,
                                    indices_to_alter=quote_to_token_positions,
                                    gaussian_smoothing_kwargs=gaussian_smoothing_kwargs,
                                    shape = (height,width,self.vae_scale_factor*2),
                                    bbox=attention_kwargs.get("regional_boxes"),
                                    child_bbox = attention_kwargs.get("regional_child_boxes"), # 不存在则返回None
                                    loss_util=loss_util
                                )
                            elif boxConfig.lossType == "opt":
                                loss_fg, loss_list = compute_opt_loss(
                                    attention_store=attention_store,
                                    indices_to_alter=quote_to_token_positions,
                                    gaussian_smoothing_kwargs=gaussian_smoothing_kwargs,
                                    shape = (height,width,self.vae_scale_factor*2),
                                    bbox= attention_kwargs.get("regional_boxes"),
                                    child_bbox = attention_kwargs.get("regional_child_boxes") # 不存在则返回None
                                )
                            return loss_fg
                        

    @torch.no_grad()
    def multi_step_loss(
        self,
        base_prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        attention_store: AttentionStore = None,
        true_cfg_scale: float = 4.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        mask_inject_steps: int = 5,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 1.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        gaussian_smoothing_kwargs: Optional[Dict[str, Any]] = None, # added ,用于attention map 的高斯平滑
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            base_prompt,
            height,
            width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False
        self._gaussian_smoothing_kwargs = gaussian_smoothing_kwargs if gaussian_smoothing_kwargs is not None else {}
        self._height = height
        self._width = width

        # get quote prompt token index
        # if isinstance(base_prompt, str) and '"' in base_prompt:
        #     quote_to_token_positions = self.get_token_index_v2(base_prompt, quote_prompt=True, region_prompts = attention_kwargs.get("regional_prompts", None)[:-1])
        #     # print("🔗 引号内容对应的 token 位置:", quote_to_token_positions)
        # else:
        #     quote_to_token_positions = None


        if isinstance(base_prompt, str):
            quote_to_token_positions = self.get_token_index_v3(base_prompt, quote_prompt=True, region_prompts = attention_kwargs.get("regional_prompts", None))
            print("🔗 引号内容对应的 token 位置:", quote_to_token_positions)
        else:
            quote_to_token_positions = None

        # 2. Define call parameters
        if base_prompt is not None and isinstance(base_prompt, str):
            batch_size = 1
        elif base_prompt is not None and isinstance(base_prompt, list):
            batch_size = len(base_prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        prompt_embeds, prompt_embeds_mask = self.encode_prompt( # 在只输入文本的情况下，可以看成是使用Qwen-7B-Chat进行文本编码
            prompt=base_prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        boxConfig.text_len = prompt_embeds.shape[1]
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )

        # added: define base mask and inputs
        # base_mask = torch.ones((height, width), device=device, dtype=self.transformer.dtype) # base mask uses the whole image mask
        # base_inputs = [(base_mask, prompt_embeds)]

        # added: encode regional prompts,define regional inputs
        regional_inputs = []
        if 'regional_prompts' in attention_kwargs and 'regional_masks' in attention_kwargs:
            for regional_prompt, regional_mask in zip(attention_kwargs['regional_prompts'], attention_kwargs['regional_masks']):
                regional_prompt_embeds, regional_prompt_embeds_masks = self.encode_prompt(
                    prompt=regional_prompt,
                    prompt_embeds=None,
                    prompt_embeds_mask=None,
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=max_sequence_length
                )
                regional_inputs.append((regional_mask, regional_prompt_embeds, regional_prompt_embeds_masks))

        ## added: prepare masks for regional control
        conds = []
        cond_masks = []
        masks = []
        each_prompt_seq_len = [] 
        H, W = height//(self.vae_scale_factor)//2, width//(self.vae_scale_factor)//2
        hidden_seq_len = H * W

        # prepare base ration regional masks
        if attention_kwargs is not None and attention_kwargs["enable_whole_regional_mask"]:
            attention_kwargs["whole_regional_mask"] = torch.nn.functional.interpolate(attention_kwargs["whole_regional_mask"][None, None, :, :], (H, W), mode='nearest-exact').flatten().unsqueeze(1)
        
        for mask, cond, cond_mask in regional_inputs:
            if mask is not None: # resize regional masks to image size, the flatten is to match the seq len
                mask = torch.nn.functional.interpolate(mask[None, None, :, :], (H, W), mode='nearest-exact').flatten().unsqueeze(1).repeat(1, cond.size(1))
            else:
                mask = torch.ones((H*W, cond.size(1))).to(device=cond.device)
            masks.append(mask)
            conds.append(cond)
            cond_masks.append(cond_mask)
            each_prompt_seq_len.append(cond.shape[1])
        regional_embeds = torch.cat(conds, dim=1)
        regional_embeds_mask = torch.cat(cond_masks, dim=1)
        encoder_seq_len = regional_embeds.shape[1]

        # initialize attention mask
        regional_attention_mask = torch.zeros(
            (encoder_seq_len + hidden_seq_len, encoder_seq_len + hidden_seq_len),
            device=masks[0].device,
            dtype=torch.bool
        )
        num_of_regions = len(masks)

        # initialize self-attended mask
        self_attend_masks = torch.zeros((hidden_seq_len, hidden_seq_len), device=masks[0].device, dtype=torch.bool)

        # initialize union mask
        union_masks = torch.zeros((hidden_seq_len, hidden_seq_len), device=masks[0].device, dtype=torch.bool)

        # handle each mask
        seq_len_begin = 0
        seq_len_end = 0
        for i in range(num_of_regions):
            # caculate the begin and end of the current region
            seq_len_begin = seq_len_end
            seq_len_end = seq_len_begin + each_prompt_seq_len[i]

            # txt attends to itself
            # regional_attention_mask[i*each_prompt_seq_len:(i+1)*each_prompt_seq_len, i*each_prompt_seq_len:(i+1)*each_prompt_seq_len] = True
            regional_attention_mask[seq_len_begin:seq_len_end, seq_len_begin:seq_len_end] = True

            # txt attends to corresponding regional img
            # regional_attention_mask[i*each_prompt_seq_len:(i+1)*each_prompt_seq_len, encoder_seq_len:] = masks[i].transpose(-1, -2)
            regional_attention_mask[seq_len_begin:seq_len_end, encoder_seq_len:] = masks[i].transpose(-1, -2)

            # regional img attends to corresponding txt
            # regional_attention_mask[encoder_seq_len:, i*each_prompt_seq_len:(i+1)*each_prompt_seq_len] = masks[i]
            regional_attention_mask[encoder_seq_len:, seq_len_begin:seq_len_end] = masks[i]

            # regional img attends to corresponding regional img
            img_size_masks = masks[i][:, :1].repeat(1, hidden_seq_len)
            img_size_masks_transpose = img_size_masks.transpose(-1, -2)
            self_attend_masks = torch.logical_or(self_attend_masks, 
                                                    torch.logical_and(img_size_masks, img_size_masks_transpose))

            # update union
            union_masks = torch.logical_or(union_masks, 
                                            torch.logical_or(img_size_masks, img_size_masks_transpose))

        background_masks = torch.logical_not(union_masks)

        background_and_self_attend_masks = torch.logical_or(background_masks, self_attend_masks)

        regional_attention_mask[encoder_seq_len:, encoder_seq_len:] = background_and_self_attend_masks
        ## added : done prepare masks for regional control


        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        img_shapes = [(1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2)] * batch_size

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # scale_range = np.linspace(boxConfig.scale_range[0], boxConfig.scale_range[1], self._num_timesteps)
        scale_range = boxConfig.scale_range_value

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if self.attention_kwargs is None:
            self._attention_kwargs = {}

        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
        negative_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist() if negative_prompt_embeds_mask is not None else None
        )
        regional_txt_seq_lens = regional_embeds_mask.sum(dim=1).tolist() if regional_embeds_mask is not None else None

        # handle infer attention mask
        regional_attention_mask = regional_attention_mask.to(device)
        infer_attention_kwargs = {
            'double_inject_blocks_interval': attention_kwargs['double_inject_blocks_interval'] if 'double_inject_blocks_interval' in attention_kwargs else len(self.transformer.transformer_blocks),
            "whole_regional_mask": attention_kwargs["whole_regional_mask"].to(device).to(latents.dtype),  # 操作是否局限在mask内
            "enable_whole_regional_mask": attention_kwargs["enable_whole_regional_mask"]
        }

        # add some args for visualization
        boxConfig.text_index = quote_to_token_positions
        print("🔗 引号内容对应的 token 位置:", quote_to_token_positions)
        boxConfig.bbox = attention_kwargs.get("regional_boxes")

        # LossUtil 初始化
        if boxConfig.text_index is not None:
            loss_names = boxConfig.text_index.keys()
            loss_util = LossUtil(loss_names, total_weight=boxConfig.total_weight)


        # 6. Denoising loop
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                boxConfig.now_step = i

                self._current_timestep = t
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                # 基于局部梯度更新latents，使得初始latents的布局更符合区域提示的要求
                if i == boxConfig.Cumulate_steps:
                    boxConfig.switch_box_loss = True
                    with torch.enable_grad():
                        # 在训练循环开始前启用
                        # torch.autograd.set_detect_anomaly(True)
                        latents = latents.clone().detach().requires_grad_(True)

                        # train all layers has no such big memory cost
                        self.set_train_transform_layer(boxConfig.train_layer)

                        # Forward pass of denoising with text conditioning
                        noise_pred_text = self.transformer(
                            hidden_states=latents,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            encoder_hidden_states_mask=prompt_embeds_mask,
                            encoder_hidden_states=prompt_embeds,
                            img_shapes=img_shapes,
                            # txt_seq_lens=negative_txt_seq_lens,
                            regional_txt_seq_lens=txt_seq_lens,
                            attention_kwargs=self.attention_kwargs,
                            return_dict=False,
                        )[0]

                        self.transformer.zero_grad()

                        # Perform gradient update # 此处是几个局部的差值算了一个总的loss,然后该loss应用于全局，这样是否合理？存在问题，需要改进
                    
                        if boxConfig.lossType == "diff":
                            loss_fg, loss_list = compute_diff_loss(
                                attention_store=attention_store,
                                indices_to_alter=quote_to_token_positions,
                                gaussian_smoothing_kwargs=gaussian_smoothing_kwargs,
                                shape = (height,width,self.vae_scale_factor*2),
                                bbox= attention_kwargs.get("regional_boxes"),
                                child_bbox = attention_kwargs.get("regional_child_boxes") # 不存在则返回None
                            )
                        elif boxConfig.lossType == "rnb":
                            loss_fg, loss_list = compute_rnb_loss(
                                attention_store=attention_store,
                                indices_to_alter=quote_to_token_positions,
                                gaussian_smoothing_kwargs=gaussian_smoothing_kwargs,
                                shape = (height,width,self.vae_scale_factor*2),
                                bbox=attention_kwargs.get("regional_boxes"),
                                child_bbox = attention_kwargs.get("regional_child_boxes"), # 不存在则返回None
                                loss_util=loss_util
                            )
                        elif boxConfig.lossType == "opt":
                            loss_fg, loss_list = compute_opt_loss(
                                attention_store=attention_store,
                                indices_to_alter=quote_to_token_positions,
                                gaussian_smoothing_kwargs=gaussian_smoothing_kwargs,
                                shape = (height,width,self.vae_scale_factor*2),
                                bbox= attention_kwargs.get("regional_boxes"),
                                child_bbox = attention_kwargs.get("regional_child_boxes") # 不存在则返回None
                            )
                        if loss_fg != 0:
                            latents = self._update_latent(latents=latents, loss=loss_fg, loss_list = loss_list, # 原实现此处用loss
                                                            step_size=boxConfig.scale_factor * scale_range[i])
                        return loss_fg

                    boxConfig.switch_box_loss = False

                       



                if self.interrupt:
                    continue

                if i < mask_inject_steps:
                    chosen_prompt_embeds = regional_embeds
                    chosen_prompt_embeds_mask = regional_embeds_mask
                    base_ratio = attention_kwargs['base_ratio']
                    infer_attention_kwargs["regional_attention_mask"] = regional_attention_mask
                else:
                    chosen_prompt_embeds = prompt_embeds
                    chosen_prompt_embeds_mask = prompt_embeds_mask
                    regional_txt_seq_lens = txt_seq_lens
                    base_ratio = None

                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states_mask=chosen_prompt_embeds_mask,
                        encoder_hidden_states=chosen_prompt_embeds, # regional_embeds or base prompt_embeds -> change
                        encoder_hidden_states_base_mask=prompt_embeds_mask, # base prompt mask -> add
                        encoder_hidden_states_base=prompt_embeds, # base prompt embeds -> add
                        base_ratio=base_ratio, # base ratio for regional control -> add
                        img_shapes=img_shapes,
                        txt_seq_lens=txt_seq_lens,
                        regional_txt_seq_lens=regional_txt_seq_lens,
                        attention_kwargs=infer_attention_kwargs,
                        return_dict=False,
                    )[0]

                if do_true_cfg:
                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred = self.transformer(
                            hidden_states=latents,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            encoder_hidden_states_mask=negative_prompt_embeds_mask,
                            encoder_hidden_states=negative_prompt_embeds,
                            img_shapes=img_shapes,
                            # txt_seq_lens=negative_txt_seq_lens,
                            regional_txt_seq_lens=negative_txt_seq_lens,
                            attention_kwargs=self.attention_kwargs,
                            return_dict=False,
                        )[0]
                    comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                    noise_pred = comb_pred * (cond_norm / noise_norm)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                    
                if boxConfig.visual_middle_res:
                    visualize_latent_map(self, latents.clone().detach(), height, width, i)

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None
        return loss_fg
    


    """
        @torch.inference_mode() 是 PyTorch 提供的一个 装饰器（decorator），
        用于将函数或方法标记为“推理模式”（inference mode），即仅用于模型前向传播（forward pass），
        不进行梯度计算，也不构建计算图。
        它是 torch.no_grad() 的更严格、更高效的版本，专为推理（inference）场景设计。
    """
    # @torch.inference_mode() 会抑制梯度计算，所以此处用@torch.no_grad()
    # @torch.inference_mode()
    @torch.no_grad()
    def mutil_step_call(
        self,
        base_prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        attention_store: AttentionStore = None,
        true_cfg_scale: float = 4.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        mask_inject_steps: int = 5,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 1.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        gaussian_smoothing_kwargs: Optional[Dict[str, Any]] = None, # added ,用于attention map 的高斯平滑
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is
                not greater than `1`).
            true_cfg_scale (`float`, *optional*, defaults to 1.0):
                When > 1.0 and a provided `negative_prompt`, enables true classifier-free guidance.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 3.5):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.qwenimage.QwenImagePipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.qwenimage.QwenImagePipelineOutput`] or `tuple`:
            [`~pipelines.qwenimage.QwenImagePipelineOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is a list with the generated images.
        """

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            base_prompt,
            height,
            width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False
        self._gaussian_smoothing_kwargs = gaussian_smoothing_kwargs if gaussian_smoothing_kwargs is not None else {}
        self._height = height
        self._width = width

        # get quote prompt token index
        # if isinstance(base_prompt, str) and '"' in base_prompt:
        if isinstance(base_prompt, str):
            quote_to_token_positions = self.get_token_index_v2(base_prompt, quote_prompt=False, region_prompts = attention_kwargs.get("regional_prompts", None)[:-1])
            print("🔗 引号内容对应的 token 位置:", quote_to_token_positions)
        else:
            quote_to_token_positions = None

        # 2. Define call parameters
        if base_prompt is not None and isinstance(base_prompt, str):
            batch_size = 1
        elif base_prompt is not None and isinstance(base_prompt, list):
            batch_size = len(base_prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        prompt_embeds, prompt_embeds_mask = self.encode_prompt( # 在只输入文本的情况下，可以看成是使用Qwen-7B-Chat进行文本编码
            prompt=base_prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        boxConfig.text_len = prompt_embeds.shape[1]
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )

        # added: define base mask and inputs
        # base_mask = torch.ones((height, width), device=device, dtype=self.transformer.dtype) # base mask uses the whole image mask
        # base_inputs = [(base_mask, prompt_embeds)]

        # added: encode regional prompts,define regional inputs
        regional_inputs = []
        if 'regional_prompts' in attention_kwargs and 'regional_masks' in attention_kwargs:
            for regional_prompt, regional_mask in zip(attention_kwargs['regional_prompts'], attention_kwargs['regional_masks']):
                regional_prompt_embeds, regional_prompt_embeds_masks = self.encode_prompt(
                    prompt=regional_prompt,
                    prompt_embeds=None,
                    prompt_embeds_mask=None,
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=max_sequence_length
                )
                regional_inputs.append((regional_mask, regional_prompt_embeds, regional_prompt_embeds_masks))

        ## added: prepare masks for regional control
        conds = []
        cond_masks = []
        masks = []
        each_prompt_seq_len = [] 
        H, W = height//(self.vae_scale_factor)//2, width//(self.vae_scale_factor)//2
        hidden_seq_len = H * W

        # prepare base ration regional masks
        if attention_kwargs is not None and attention_kwargs["enable_whole_regional_mask"]:
            attention_kwargs["whole_regional_mask"] = torch.nn.functional.interpolate(attention_kwargs["whole_regional_mask"][None, None, :, :], (H, W), mode='nearest-exact').flatten().unsqueeze(1)
        
        for mask, cond, cond_mask in regional_inputs:
            if mask is not None: # resize regional masks to image size, the flatten is to match the seq len
                mask = torch.nn.functional.interpolate(mask[None, None, :, :], (H, W), mode='nearest-exact').flatten().unsqueeze(1).repeat(1, cond.size(1))
            else:
                mask = torch.ones((H*W, cond.size(1))).to(device=cond.device)
            masks.append(mask)
            conds.append(cond)
            cond_masks.append(cond_mask)
            each_prompt_seq_len.append(cond.shape[1])
        regional_embeds = torch.cat(conds, dim=1)
        regional_embeds_mask = torch.cat(cond_masks, dim=1)
        encoder_seq_len = regional_embeds.shape[1]

        # initialize attention mask
        regional_attention_mask = torch.zeros(
            (encoder_seq_len + hidden_seq_len, encoder_seq_len + hidden_seq_len),
            device=masks[0].device,
            dtype=torch.bool
        )
        num_of_regions = len(masks)

        # initialize self-attended mask
        self_attend_masks = torch.zeros((hidden_seq_len, hidden_seq_len), device=masks[0].device, dtype=torch.bool)

        # initialize union mask
        union_masks = torch.zeros((hidden_seq_len, hidden_seq_len), device=masks[0].device, dtype=torch.bool)

        # handle each mask
        seq_len_begin = 0
        seq_len_end = 0
        for i in range(num_of_regions):
            # caculate the begin and end of the current region
            seq_len_begin = seq_len_end
            seq_len_end = seq_len_begin + each_prompt_seq_len[i]

            # txt attends to itself
            # regional_attention_mask[i*each_prompt_seq_len:(i+1)*each_prompt_seq_len, i*each_prompt_seq_len:(i+1)*each_prompt_seq_len] = True
            regional_attention_mask[seq_len_begin:seq_len_end, seq_len_begin:seq_len_end] = True

            # txt attends to corresponding regional img
            # regional_attention_mask[i*each_prompt_seq_len:(i+1)*each_prompt_seq_len, encoder_seq_len:] = masks[i].transpose(-1, -2)
            regional_attention_mask[seq_len_begin:seq_len_end, encoder_seq_len:] = masks[i].transpose(-1, -2)

            # regional img attends to corresponding txt
            # regional_attention_mask[encoder_seq_len:, i*each_prompt_seq_len:(i+1)*each_prompt_seq_len] = masks[i]
            regional_attention_mask[encoder_seq_len:, seq_len_begin:seq_len_end] = masks[i]

            # regional img attends to corresponding regional img
            img_size_masks = masks[i][:, :1].repeat(1, hidden_seq_len)
            img_size_masks_transpose = img_size_masks.transpose(-1, -2)
            self_attend_masks = torch.logical_or(self_attend_masks, 
                                                    torch.logical_and(img_size_masks, img_size_masks_transpose))

            # update union
            union_masks = torch.logical_or(union_masks, 
                                            torch.logical_or(img_size_masks, img_size_masks_transpose))

        background_masks = torch.logical_not(union_masks)

        background_and_self_attend_masks = torch.logical_or(background_masks, self_attend_masks)

        regional_attention_mask[encoder_seq_len:, encoder_seq_len:] = background_and_self_attend_masks
        ## added : done prepare masks for regional control


        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        img_shapes = [(1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2)] * batch_size

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # scale_range = np.linspace(boxConfig.scale_range[0], boxConfig.scale_range[1], self._num_timesteps)
        scale_range = boxConfig.scale_range_value

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if self.attention_kwargs is None:
            self._attention_kwargs = {}

        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
        negative_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist() if negative_prompt_embeds_mask is not None else None
        )
        regional_txt_seq_lens = regional_embeds_mask.sum(dim=1).tolist() if regional_embeds_mask is not None else None

        # handle infer attention mask
        regional_attention_mask = regional_attention_mask.to(device)
        infer_attention_kwargs = {
            'double_inject_blocks_interval': attention_kwargs['double_inject_blocks_interval'] if 'double_inject_blocks_interval' in attention_kwargs else len(self.transformer.transformer_blocks),
            "whole_regional_mask": attention_kwargs["whole_regional_mask"].to(device).to(latents.dtype),  # 操作是否局限在mask内
            "enable_whole_regional_mask": attention_kwargs["enable_whole_regional_mask"]
        }

        # add some args for visualization
        boxConfig.text_index = quote_to_token_positions
        print(quote_to_token_positions)
        boxConfig.bbox = attention_kwargs.get("regional_boxes")

        # LossUtil 初始化
        if boxConfig.text_index is not None:
            loss_names = boxConfig.text_index.keys()
        loss_util = LossUtil(loss_names, total_weight=boxConfig.total_weight)


        # 6. Denoising loop
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                boxConfig.now_step = i

                self._current_timestep = t
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                # 基于局部梯度更新latents，使得初始latents的布局更符合区域提示的要求
                if i in boxConfig.max_iter_to_alter:
                    boxConfig.switch_box_loss = True
                    with torch.enable_grad():
                        # 在训练循环开始前启用
                        for _ in range(boxConfig.max_refinement_steps[i]):
                            # torch.autograd.set_detect_anomaly(True)
                            latents = latents.clone().detach().requires_grad_(True)
                            for _ in range(boxConfig.Cumulate_steps):
                                latents_copy = latents * 1.0   # 防止latents被赋值消失

                                # train all layers has no such big memory cost
                                self.set_train_transform_layer(boxConfig.train_layer)


                                if i < mask_inject_steps:
                                    chosen_prompt_embeds = regional_embeds
                                    chosen_prompt_embeds_mask = regional_embeds_mask
                                    base_ratio = attention_kwargs['base_ratio']
                                    infer_attention_kwargs["regional_attention_mask"] = regional_attention_mask
                                else:
                                    chosen_prompt_embeds = prompt_embeds
                                    chosen_prompt_embeds_mask = prompt_embeds_mask
                                    regional_txt_seq_lens = txt_seq_lens
                                    base_ratio = None

                                with self.transformer.cache_context("cond"):
                                    noise_pred = self.transformer(
                                        hidden_states=latents_copy,
                                        timestep=timestep / 1000,
                                        guidance=guidance,
                                        encoder_hidden_states_mask=chosen_prompt_embeds_mask,
                                        encoder_hidden_states=chosen_prompt_embeds, # regional_embeds or base prompt_embeds -> change
                                        encoder_hidden_states_base_mask=prompt_embeds_mask, # base prompt mask -> add
                                        encoder_hidden_states_base=prompt_embeds, # base prompt embeds -> add
                                        base_ratio=base_ratio, # base ratio for regional control -> add
                                        img_shapes=img_shapes,
                                        txt_seq_lens=txt_seq_lens,
                                        regional_txt_seq_lens=regional_txt_seq_lens,
                                        attention_kwargs=infer_attention_kwargs,
                                        return_dict=False,
                                    )[0]

                                if do_true_cfg:
                                    with self.transformer.cache_context("uncond"):
                                        neg_noise_pred = self.transformer(
                                            hidden_states=latents_copy,
                                            timestep=timestep / 1000,
                                            guidance=guidance,
                                            encoder_hidden_states_mask=negative_prompt_embeds_mask,
                                            encoder_hidden_states=negative_prompt_embeds,
                                            img_shapes=img_shapes,
                                            # txt_seq_lens=negative_txt_seq_lens,
                                            regional_txt_seq_lens=negative_txt_seq_lens,
                                            attention_kwargs=self.attention_kwargs,
                                            return_dict=False,
                                        )[0]
                                    comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                                    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                                    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                                    noise_pred = comb_pred * (cond_norm / noise_norm)

                                # compute the previous noisy sample x_t -> x_t-1
                                latents_dtype = latents_copy.dtype
                                latents_copy = self.scheduler.step(noise_pred, t, latents_copy, return_dict=False)[0]

                                if latents_copy.dtype != latents_dtype:
                                    if torch.backends.mps.is_available():
                                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                                        latents_copy = latents_copy.to(latents_dtype)


                                self.transformer.zero_grad()

                            # Perform gradient update # 此处是几个局部的差值算了一个总的loss,然后该loss应用于全局，这样是否合理？存在问题，需要改进
                            if i in boxConfig.max_iter_to_alter:
                                if boxConfig.lossType == "diff":
                                    loss_fg, loss_list = compute_diff_loss(
                                        attention_store=attention_store,
                                        indices_to_alter=quote_to_token_positions,
                                        gaussian_smoothing_kwargs=gaussian_smoothing_kwargs,
                                        shape = (height,width,self.vae_scale_factor*2),
                                        bbox= attention_kwargs.get("regional_boxes"),
                                        child_bbox = attention_kwargs.get("regional_child_boxes") # 不存在则返回None
                                    )
                                elif boxConfig.lossType == "rnb":
                                    loss_fg, loss_list = compute_rnb_loss(
                                        attention_store=attention_store,
                                        indices_to_alter=quote_to_token_positions,
                                        gaussian_smoothing_kwargs=gaussian_smoothing_kwargs,
                                        shape = (height,width,self.vae_scale_factor*2),
                                        bbox=attention_kwargs.get("regional_boxes"),
                                        child_bbox = attention_kwargs.get("regional_child_boxes"), # 不存在则返回None
                                        loss_util=loss_util
                                    )
                                elif boxConfig.lossType == "opt":
                                    loss_fg, loss_list = compute_opt_loss(
                                        attention_store=attention_store,
                                        indices_to_alter=quote_to_token_positions,
                                        gaussian_smoothing_kwargs=gaussian_smoothing_kwargs,
                                        shape = (height,width,self.vae_scale_factor*2),
                                        bbox= attention_kwargs.get("regional_boxes"),
                                        child_bbox = attention_kwargs.get("regional_child_boxes") # 不存在则返回None
                                    )
                                if loss_fg != 0:
                                    latents = self._update_latent(latents=latents, loss=loss_fg, loss_list = loss_list, # 原实现此处用loss
                                                                    step_size=boxConfig.scale_factor * scale_range[i])
                                    
                            self.scheduler.set_step_index(i) # 重置step_index,保证每次迭代时，scheduler都从当前timestep开始
                        
                        
                    boxConfig.switch_box_loss = False

                       



                if self.interrupt:
                    continue

                if i < mask_inject_steps:
                    chosen_prompt_embeds = regional_embeds
                    chosen_prompt_embeds_mask = regional_embeds_mask
                    base_ratio = attention_kwargs['base_ratio']
                    infer_attention_kwargs["regional_attention_mask"] = regional_attention_mask
                else:
                    chosen_prompt_embeds = prompt_embeds
                    chosen_prompt_embeds_mask = prompt_embeds_mask
                    regional_txt_seq_lens = txt_seq_lens
                    base_ratio = None

                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states_mask=chosen_prompt_embeds_mask,
                        encoder_hidden_states=chosen_prompt_embeds, # regional_embeds or base prompt_embeds -> change
                        encoder_hidden_states_base_mask=prompt_embeds_mask, # base prompt mask -> add
                        encoder_hidden_states_base=prompt_embeds, # base prompt embeds -> add
                        base_ratio=base_ratio, # base ratio for regional control -> add
                        img_shapes=img_shapes,
                        txt_seq_lens=txt_seq_lens,
                        regional_txt_seq_lens=regional_txt_seq_lens,
                        attention_kwargs=infer_attention_kwargs,
                        return_dict=False,
                    )[0]

                if do_true_cfg:
                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred = self.transformer(
                            hidden_states=latents,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            encoder_hidden_states_mask=negative_prompt_embeds_mask,
                            encoder_hidden_states=negative_prompt_embeds,
                            img_shapes=img_shapes,
                            # txt_seq_lens=negative_txt_seq_lens,
                            regional_txt_seq_lens=negative_txt_seq_lens,
                            attention_kwargs=self.attention_kwargs,
                            return_dict=False,
                        )[0]
                    comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                    noise_pred = comb_pred * (cond_norm / noise_norm)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                    
                if boxConfig.visual_middle_res:
                    visualize_latent_map(self, latents.clone().detach(), height, width, i)

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return QwenImagePipelineOutput(images=image)