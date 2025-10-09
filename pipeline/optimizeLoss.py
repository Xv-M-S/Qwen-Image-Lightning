from typing import Dict, List, Tuple, Any, Optional
from pipeline.attentionControl import AttentionStore
from pipeline.gaussion_smoothing import GaussianSmoothing
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.boxLossConfig import boxConfig as config
import math

def compute_attention_center_loss(attention_map, box, top_k=10, distance_type='l2'):
    """
    计算注意力中心损失：
    1. 提取 attention_map 中前 top_k 大的值的位置
    2. 计算这些位置到 box 中心的距离，取平均作为 loss

    Args:
        attention_map: (H, W) 或 (1, H, W) 的注意力图
        box: [x1, y1, x2, y2] 归一化或像素坐标
        top_k: 取前 top_k 个最大响应点
        distance_type: 'l1' 或 'l2'

    Returns:
        loss: 标量 Tensor
    """
    # 确保 attention_map 是 (H, W)
    if attention_map.dim() == 3:
        attention_map = attention_map.squeeze(0)  # (H, W)
    
    H, W = attention_map.shape

    # Step 1: 获取前 top_k 大的值的索引
    values, indices = torch.topk(attention_map.view(-1), k=int(top_k), largest=True, sorted=False)
    # indices: [0, H*W) 的一维索引

    # 转换为二维坐标 (y, x)
    y_coords = indices // W  # 行号
    x_coords = indices % W   # 列号

    # Step 2: 计算 box 中心点
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # 注意：box 可能是归一化坐标 [0,1] 或像素坐标 [0, W], [0, H]
    # 如果 attention_map 是归一化空间，需将中心点也归一化到 [0,1] 或 [0,W], [0,H]

    # 将中心点转换为像素坐标（如果 box 是归一化的）
    # 假设 box 是归一化坐标 [0,1]
    center_x_pixel = center_x * (W - 1)
    center_y_pixel = center_y * (H - 1)

    # Step 3: 计算距离
    if distance_type == 'l2':
        distances = torch.sqrt(
            (x_coords.float() - center_x_pixel) ** 2 +
            (y_coords.float() - center_y_pixel) ** 2
        )
    elif distance_type == 'l1':
        distances = torch.abs(x_coords.float() - center_x_pixel) + \
                    torch.abs(y_coords.float() - center_y_pixel)
    else:
        raise ValueError("distance_type must be 'l1' or 'l2'")

    # Step 4: 取平均距离作为 loss
    loss = distances.mean()

    return loss

def _aggregate_attention(
                    attention_store: AttentionStore,
                    is_cross: bool,
                    shape: Tuple[int, int],
                    select: int = 0,
                ) -> torch.Tensor:
    """ Aggregates the attention across the different layers and heads at the specified resolution. """
    out = []
    attention_maps = attention_store.get_store_attention()

    if "img-to-txt" in config.feature_map:
        for item in attention_maps[f"{'img-to-txt'}"]:
            cross_maps = item.reshape(1, -1, int(shape[0]/shape[2]), int(shape[1]/shape[2]), item.shape[-1])[select]
            out.append(cross_maps)
    if "txt-to-img" in config.feature_map:
        for item in attention_maps[f"{'txt-to-img'}"]:
            item.transpose_(1, 2)
            cross_maps = item.reshape(1, -1, int(shape[0]/shape[2]), int(shape[1]/shape[2]), item.shape[-1])[select]
            out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out

def _compute_max_attention_per_character(
                                    attention_maps: torch.Tensor,
                                    indices_to_alter: List[int],
                                    smooth_attentions: bool = False,
                                    shape: Optional[Tuple[int, int, int]] = None,
                                    sigma: float = 0.5,
                                    kernel_size: int = 3,
                                    bbox: List[int] = None,
                                    child_bbox:List[List[List[int]]] = None,
                                    ) -> List[torch.Tensor]:
    """ Computes the maximum attention value for each of the tokens we wish to alter. """
    # QwenVL 截断了 eot token,第一个便是有效字符 -> 旧实现见 _compute_max_attention_per_index_old
    attention_for_text = attention_maps
    # attention_for_text = attention_for_text * 100
    # attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

    # Extract the maximum values
    diff_fg_bg = []


    _, _, scale_factor = shape

    cnt = 0
    """
        情况不一样 -> boxdiff是一个框只有一个token描述,而我们一个框有多个token描述,此处有待改进
    """
    for cnt, indices in enumerate(indices_to_alter):
        character_boxes = child_bbox[cnt]

        length = len(indices)
        # 去掉引号 1,length-1
        for index in range(1,length-1):
            i = indices[index]
            box_index = index - 1
            image_mean = attention_for_text[:, :, i]
            
            # 基于极大极小值进行归一化
            image_mean = (image_mean - image_mean.min()) / (image_mean.max() - image_mean.min() + 1e-8)

            box = [max(round(b/scale_factor), 0) for b in character_boxes[box_index]]
            x1, y1, x2, y2 = box

            # coordinates to masks
            obj_mask = torch.zeros_like(image_mean)
            ones_mask = torch.ones([y2 - y1, x2 - x1], dtype=obj_mask.dtype).to(obj_mask.device)
            obj_mask[y1:y2, x1:x2] = ones_mask
            bg_mask = 1 - obj_mask

            if smooth_attentions:
                smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).cuda()
                input = F.pad(image_mean.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
                image_mean = smoothing(input).squeeze(0).squeeze(0)

            # Inner-Box constraint
            inner_pix_num = obj_mask.sum()
            inner_pix_avg = (image_mean * obj_mask).sum() /inner_pix_num

            if inner_pix_num == 0:
                # 可能由于box国小导致inner_pix_num为0
                continue
            


            # Outer-Box constraint
            outer_pix_num = bg_mask.sum()
            outer_pix_avg = (image_mean * bg_mask).sum() /outer_pix_num

            diff = inner_pix_avg - outer_pix_avg
            loss = - torch.log(torch.sigmoid(diff) + 1e-8)

            diff_fg_bg.append(loss)

    return diff_fg_bg

def _compute_max_attention_per_index(
                                    attention_maps: torch.Tensor,
                                    indices_to_alter: List[int],
                                    smooth_attentions: bool = False,
                                    shape: Optional[Tuple[int, int, int]] = None,
                                    sigma: float = 0.5,
                                    kernel_size: int = 3,
                                    bbox: List[int] = None,
                                    ) -> List[torch.Tensor]:
    """ Computes the maximum attention value for each of the tokens we wish to alter. """
    """
        新的Loss计算方式:
            给定一个attention map
            1.求前 x 大的值
            2.计算这些值与原本位置box的中心点的距离作为loss指标
    """
    # QwenVL 截断了 eot token,第一个便是有效字符 -> 旧实现见 _compute_max_attention_per_index_old
    attention_for_text = attention_maps
    # attention_for_text = attention_for_text * 100
    # attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

    # Extract the maximum values
    diff_fg_bg = []


    _, _, scale_factor = shape

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
        image_mean = (image_mean - image_mean.min()) / (image_mean.max() - image_mean.min() + 1e-8)

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
            image_mean = smoothing(input).squeeze(0).squeeze(0)

        # Inner-Box constraint
        inner_pix_num = obj_mask.sum()
        inner_pix_avg = (image_mean * obj_mask).sum() /inner_pix_num


        # Outer-Box constraint
        outer_pix_num = bg_mask.sum()
        outer_pix_avg = (image_mean * bg_mask).sum() /outer_pix_num

        diff = inner_pix_avg - outer_pix_avg
        loss = - torch.log(torch.sigmoid(diff) + 1e-8)

        distance_loss = compute_attention_center_loss(image_mean, box, top_k=math.fabs(x1-x2) * math.fabs(y1-y2), distance_type='l2')
        print(distance_loss)
        loss = loss + distance_loss

        diff_fg_bg.append(loss)

    return diff_fg_bg
def _aggregate_and_get_max_attention_per_token(
                                                attention_store: AttentionStore,
                                                indices_to_alter:Dict[str, List[int]],
                                                gaussian_smoothing_kwargs:Dict[str, Any],
                                                shape: Tuple[int, int, int],
                                                bbox:List[List[int]],
                                                child_bbox:List[List[List[int]]]
                                                ):
    """ Aggregates the attention for each token and computes the max activation value for each token to alter. """
    attention_maps = _aggregate_attention(
            attention_store=attention_store,
            shape=shape, 
            is_cross=True,
            select=0 # why 0
        )
    values_list = [indices_to_alter[key] for key in indices_to_alter]
    if child_bbox is None or config.use_character_box_loss is False:
        diff_fg_bg = _compute_max_attention_per_index(
            attention_maps=attention_maps,
            indices_to_alter=values_list,
            smooth_attentions=gaussian_smoothing_kwargs.get("smooth_attentions", False),
            sigma=gaussian_smoothing_kwargs.get("sigma", 0.5),
            kernel_size=gaussian_smoothing_kwargs.get("kernel_size", 5),
            shape=shape,
            bbox=bbox
        )
    else:
        diff_fg_bg = _compute_max_attention_per_character(
            attention_maps=attention_maps,
            indices_to_alter=values_list,
            smooth_attentions=gaussian_smoothing_kwargs.get("smooth_attentions", False),
            sigma=gaussian_smoothing_kwargs.get("sigma", 0.5),
            kernel_size=gaussian_smoothing_kwargs.get("kernel_size", 5),
            shape=shape,
            bbox=bbox,
            child_bbox=child_bbox
        )

    return  diff_fg_bg


def compute_opt_loss(
    attention_store: AttentionStore,
    indices_to_alter:Dict[str, List[int]],
    gaussian_smoothing_kwargs:Dict[str, Any],
    shape: Tuple[int, int, int],
    bbox:List[List[int]],
    child_bbox:List[List[List[int]]]
):
    diff_fg_bg = _aggregate_and_get_max_attention_per_token(
        attention_store=attention_store,
        indices_to_alter=indices_to_alter,
        gaussian_smoothing_kwargs=gaussian_smoothing_kwargs,
        shape=shape,
        bbox=bbox,
        child_bbox=child_bbox
    )

    print(f"diff_fg_bg:{sum(diff_fg_bg)}")

    return sum(diff_fg_bg), diff_fg_bg

