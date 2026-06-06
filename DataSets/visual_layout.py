import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.colors as mcolors

# ---------------------- 通用配置 ----------------------
DEFAULT_CANVAS_SIZE = (800, 800)
# 边框配色库（按元素名称分配，仅用于边框）
BORDER_COLOR_POOL = [
    "#4CAF50", "#FF9800", "#2196F3", "#8BC34A", "#E91E63",
    "#9C27B0", "#607D8B", "#FFEB3B", "#00BCD4", "#FF5722", "#795548"
]
# 需要隐藏标签的元素名称（精确匹配）
HIDE_LABELS = ["Background"]


# ---------------------- 字体配置 (修改此处) ----------------------
def configure_chinese_font():
    """配置 matplotlib 支持中文字体"""
    # 优先使用 Noto Sans CJK JP (你的系统中有这个字体)
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans', 'sans-serif']
    # 解决负号 '-' 显示为方块的问题
    plt.rcParams['axes.unicode_minus'] = False

# 调用一次配置函数
configure_chinese_font()

# # ---------------------- 核心通用函数 ----------------------
# def load_layout_data(input_data: str | Dict) -> List[Dict]:
#     """加载布局数据：支持JSON字符串/文件/字典"""
#     if isinstance(input_data, str):
#         if input_data.endswith(".json"):
#             with open(input_data, "r", encoding="utf-8") as f:
#                 data = json.load(f)
#         else:
#             data = json.loads(input_data)
#     elif isinstance(input_data, dict):
#         data = input_data
#     else:
#         raise ValueError("输入数据必须是JSON字符串、文件路径或字典")
#     # return data["layout"]
#     return data


# ---------------------- 核心通用函数 ----------------------
def load_layout_data(input_data: str | Dict):
    """ 只修复逻辑，不改变你原来的结构 """
    if isinstance(input_data, str):
        if input_data.endswith(".json"):
            with open(input_data, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.loads(input_data)
    elif isinstance(input_data, dict):
        data = input_data
    else:
        # 关键修复：直接返回你传进来的 tuple/list
        return input_data

    return data.get("layout", data)

def box_to_coords(box: List[float], canvas_w: int, canvas_h: int) -> Tuple[float, float, float, float]:
    """坐标转换：[cx, cy, w, h] → 像素坐标"""
    cx, cy, w, h = box
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))
    
    w_px = w * canvas_w
    h_px = h * canvas_h
    x = (cx * canvas_w) - (w_px / 2)
    y = canvas_h - ((cy * canvas_h) + (h_px / 2))
    return x, y, w_px, h_px

def get_border_color(name: str, color_map: Dict) -> str:
    """为元素分配边框颜色（重复元素复用）"""
    if name not in color_map:
        color_idx = len(color_map) % len(BORDER_COLOR_POOL)
        color_map[name] = BORDER_COLOR_POOL[color_idx]
    return color_map[name]

# ---------------------- 主可视化函数（最终版） ----------------------
def visualize_layout(
    input_data: str | Dict,
    save_path: str = "layout_final.png",
    canvas_size: Tuple[int, int] = DEFAULT_CANVAS_SIZE,
    show_plot: bool = True
) -> None:
    layout = load_layout_data(input_data)
    canvas_w, canvas_h = canvas_size
    
    # 初始化画布
    fig, ax = plt.subplots(1, 1, figsize=(canvas_w/100, canvas_h/100), dpi=100)
    ax.set_xlim(0, canvas_w)
    ax.set_ylim(0, canvas_h)
    ax.set_aspect('equal')
    ax.axis('off')
    # 设置画布背景为白色
    ax.set_facecolor('white')
    
    # 遍历绘制所有元素
    color_map = {}
    for elem in layout:
        name = elem.get("name", f"Element_{layout.index(elem)}")
        box = elem.get("box", [0.5, 0.5, 0.1, 0.1])
        
        if (abs(box[0] - 0.5) < 1e-6 and abs(box[1] - 0.5) < 1e-6 and 
            abs(box[2] - 1) < 1e-6 and abs(box[3] - 1) < 1e-6):
            print(box)
            continue

        # 坐标转换
        x, y, w, h = box_to_coords(box, canvas_w, canvas_h)
        # 获取边框颜色（Background用浅灰色边框）
        if name == "Background":
            border_color = "#dddddd"
        else:
            border_color = get_border_color(name, color_map)
        
        # ========== 核心1：仅绘制边框（无填充） ==========
        shape = patches.Rectangle(
            (x, y), w, h,
            facecolor="none",  # 无内部填充
            edgecolor=border_color,
            linewidth=2,       # 边框宽度
            alpha=1.0
        )
        ax.add_patch(shape)
        
        # ========== 核心2：控制标签显示（隐藏Background，文字放框上方） ==========
        if name not in HIDE_LABELS:
            # 文字位置：框体上边缘外侧（居中）
            text_x = x + w/2
            text_y = y - 5  # 上移5像素，避免重叠
            # 自动调整字体大小
            font_size = max(8, min(12, (w + h) / 20))
            ax.text(
                text_x, text_y,
                name,
                ha='center', va='bottom',  # 文字底部对齐框上沿
                fontsize=font_size, 
                weight='bold', 
                color=border_color,
                # 文字加白色描边，提升可读性
                bbox=dict(boxstyle="round,pad=0.1", facecolor='white', edgecolor='none', alpha=0.8)
            )
    
    # 保存+显示
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    if show_plot:
        plt.show()
    plt.close()

# ---------------------- 用法示例 ----------------------
if __name__ == "__main__":
    # 你的布局数据
    your_layout = {'name': 'Walnut Cake', 'box': [0.5, 0.5, 0.2, 0.3]}, {'name': 'White Plate', 'box': [0.5, 0.6, 0.3, 0.2]}, {'name': 'Cutlery Set', 'box': [0.7, 0.5, 0.1, 0.4]}, {'name': 'Coffee Cup', 'box': [0.3, 0.5, 0.1, 0.3]}, {'name': 'Green Plant', 'box': [0.4, 0.8, 0.1, 0.2]}
    # 生成最终可视化图
    visualize_layout(your_layout, save_path="layout_final.png")