# 论文方法实现变更记录

基于论文《扩散模型结构与注意力机制分析》在 Qwen-Image-Lightning 代码库中实现了三项免训练布局控制方法。

---

## 变更总结

### 新建文件

#### 1. `pipeline/noise_filtering.py` (338 行) — NQE-EAA 噪声筛选

实现了论文 Algorithm 1（NQE-EAA：基于早期注意力对齐的噪声质量评估）：

| 函数 | 说明 |
|------|------|
| `extract_attention_from_block()` | 从 DiT 块中提取 txt→img 和 img→txt 注意力图，聚合为空间图 |
| `compute_single_noise_score()` | 对单个候选噪声评分：全局均值阈值、双向注意力加权融合（α）、面积加权 |
| `compute_nqe_eaa_scores()` | 批量评估 K 个候选噪声的 NQE-EAA 分数 |
| `select_best_noise()` | 从 K 个候选中选出最高分噪声 z* = argmax s_k |

核心公式：
- s_i = α · a_i^(txt→img) + (1-α) · a_i^(img→txt)
- s_k = Σ w_i · s_i,  w_i = Area(b_i) / (H_z · W_z)
- z* = argmax_{z_k} s_k

#### 2. `pipeline/dynamic_mask.py` (480 行) — 动态软掩码策略

实现了 `DynamicMaskBuilder` 类，构建四类语义感知的软掩码：

| 方法 | 对应公式 | 说明 |
|------|---------|------|
| `build_txt_to_img_mask()` | M[t,p] = γ·m_i(p) + (1-γ)·s(t, t_base) | 文本→图像交叉注意力 |
| `build_img_to_txt_mask()` | M[p,t] = λ·I(t∈I_i) + (1-λ)·s(p, t) | 图像→文本交叉注意力 |
| `build_txt_self_attn_mask()` | 同区域=1, 跨区域=δ·sim(t_a, t_b) | 文本自注意力（语义感知衰减） |
| `build_img_self_attn_mask()` | min(1, μ·Σm_i(p)m_i(q) + η·κ(p,q)) | 图像自注意力（区域聚合+局部平滑） |
| `build_full_mask()` | 组装 (L+M)×(L+M) 完整掩码 | 四子块拼接 |
| `apply_soft_mask_to_attention()` | score' = score + log(M + ε) | 对数空间加法应用（等价于 softmax 后乘性） |

#### 3. `paper_demo.py` (409 行) — 综合演示入口

展示三步走全流程的完整脚本：
1. NQE-EAA 噪声筛选 → 最优初始噪声
2. LALO 布局感知损失 → 推理阶段潜变量优化
3. 动态软掩码 → 跨区域干扰抑制

---

### 修改文件

#### 4. `config/boxLossConfig.py` — 新增 10 个配置项

```python
# NQE-EAA: 基于早期注意力对齐的噪声质量评估
use_nqe_eaa: bool = False
nqe_eaa_candidates: int = 10        # K: 候选噪声数量
nqe_eaa_alpha: float = 0.5          # α: 双向注意力平衡系数

# Dynamic Masking: 动态软掩码策略
use_dynamic_mask: bool = False
dynamic_mask_gamma: float = 0.7     # txt→img 空间权重
dynamic_mask_lambda: float = 0.7    # img→txt 区域指示权重
dynamic_mask_delta: float = 0.3     # 跨区域文本注意力衰减
dynamic_mask_mu: float = 1.5        # 图像区域内聚合权重
dynamic_mask_eta: float = 0.3       # 图像局部平滑权重
dynamic_mask_gaussian_sigma: float = 2.0
```

#### 5. `pipeline/pipeline_qwenimage_regional.py` — 核心管道修改

| 修改位置 | 说明 |
|---------|------|
| 导入区 (L23-24) | 新增 `noise_filtering` 和 `dynamic_mask` 导入 |
| `RegionalQwenImageAttnProcessor.__init__` | 新增 `_dynamic_mask_builder` 和 `_cached_soft_mask` 属性 |
| `RegionalQwenImageAttnProcessor._build_dynamic_soft_mask` | **新增方法**：从当前特征实时构建动态软掩码 |
| `RegionalQwenImageAttnProcessor.__call__` | 修改：`boxConfig.use_dynamic_mask=True` 时调用软掩码替代硬掩码 |
| `RegionalQwenImagePipeline._setup_dynamic_masking` | **新增方法**：初始化 DynamicMaskBuilder 并注入所有 attention processor |
| `RegionalQwenImagePipeline.__call__` (L1688-1696) | 在掩码构建完成后调用 `_setup_dynamic_masking` |

---

## 架构总览

```
论文方法实现
├── 1. NQE-EAA (潜在噪声筛选)
│   └── pipeline/noise_filtering.py
│       ├── 遍历 K 个候选噪声
│       ├── 单次前向传播 → 提取全部 N 个 DiT 块注意力
│       ├── 聚合为空间注意力图 → 全局均值阈值 → 双向融合
│       └── select_best_noise() → 最优 z*
│
├── 2. LALO (布局感知注意力损失)  [已存在，基于 lossDesign.py]
│   └── pipeline/lossDesign.py (compute_diff_loss)
│       └── L_i = -log(σ(μ_i^in - μ_i^out))  → 梯度更新 z_t
│
└── 3. Dynamic Masking (动态软掩码)
    └── pipeline/dynamic_mask.py
        └── DynamicMaskBuilder
            ├── M_txt→img: 空间归属 + 语义关联
            ├── M_img→txt: 区域指示 + 特征相似
            ├── M_txt→txt: 语义感知跨区域衰减
            └── M_img→img: 区域聚合 + 高斯平滑
```

## 使用方式

```bash
# 运行论文方法演示
python paper_demo.py

# 或通过配置开关选择性启用
# 修改 paper_demo.py 中的 configure_for_paper_demo() 函数
```

### 配置开关

| 开关 | 默认值 | 说明 |
|------|--------|------|
| `boxConfig.use_nqe_eaa` | False | 启用 NQE-EAA 噪声筛选 |
| `boxConfig.use_dynamic_mask` | False | 启用动态软掩码（替代硬掩码） |
| `boxConfig.lossType` | "rnb" | 设为 "diff" = 论文 LALO 损失 |

## 向后兼容

所有新功能默认关闭，不影响现有代码和生成流程。
