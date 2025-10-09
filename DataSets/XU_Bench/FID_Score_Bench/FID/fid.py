import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
import os

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """计算两个高斯分布之间的Frechet距离"""
    # 检查输入维度是否匹配
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    # 检查均值向量维度是否相同
    assert mu1.shape == mu2.shape, 'Mean vectors have different lengths'
    # 检查协方差矩阵维度是否相同
    assert sigma1.shape == sigma2.shape, 'Covariances have different dimensions'
    
    # 计算均值差的平方和
    diff = mu1 - mu2
    
    # 计算矩阵乘积的平方根
    # 参考: https://github.com/bioinf-jku/TTUR/blob/master/fid.py
    covmean, _ = torch.linalg.sqrtm(torch.tensor(np.dot(sigma1, sigma2)), 
                                   hermitian=True)
    # 如果实部矩阵是奇异的，添加小的单位矩阵
    if not torch.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = torch.eye(sigma1.shape[0]) * eps
        covmean = torch.linalg.sqrtm((torch.tensor(sigma1) + offset) @ 
                                    (torch.tensor(sigma2) + offset))
    
    # 确保实部计算
    if torch.is_complex(covmean):
        if not torch.allclose(torch.imag(covmean), torch.zeros_like(covmean)):
            m = torch.max(torch.abs(torch.imag(covmean)))
            raise ValueError(f'Imaginary component {m}')
        covmean = torch.real(covmean)
    
    # 计算最终的FID值
    tr_covmean = torch.trace(covmean)
    fid = (diff @ diff) + torch.trace(torch.tensor(sigma1)) + torch.trace(torch.tensor(sigma2)) - 2 * tr_covmean
    
    return fid.item()

class InceptionV3(nn.Module):
    """Inception-V3模型，用于提取图像特征"""
    def __init__(self, output_blocks=[3], resize_input=True, normalize_input=True):
        super().__init__()
        # 加载预训练的Inception-v3模型
        inception = models.inception_v3(pretrained=True)
        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)
        
        # 确保模型处于评估模式
        inception.eval()
        
        # 移除不需要的层
        for param in inception.parameters():
            param.requires_grad = False
        
        self.blocks = nn.ModuleList()
        
        # 第一个块：输入到MaxPool2d
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))
        
        # 第二个块：到AvgPool2d
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.AvgPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))
        
        # 第三个块：到Mixed_5c
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d
            ]
            self.blocks.append(nn.Sequential(*block2))
        
        # 第四个块：到Mixed_6e
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e
            ]
            self.blocks.append(nn.Sequential(*block3))
        
        # 第五个块：到Mixed_7c（不包括辅助分类器）
        if self.last_needed_block >= 4:
            block4 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c
            ]
            self.blocks.append(nn.Sequential(*block4))
        
        # 自适应平均池化层
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
    
    def forward(self, x):
        """前向传播，返回指定块的特征"""
        # 输入预处理
        if self.resize_input:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        if self.normalize_input:
            x = 2 * x - 1  # 将输入从[0,1]归一化到[-1,1]
        
        # 存储中间输出
        outputs = []
        
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outputs.append(x)
            
            if idx == self.last_needed_block:
                break
        
        # 应用自适应池化并展平
        if self.last_needed_block >= 3:
            x = self.adaptive_pool(x)
            x = torch.flatten(x, 1)
            outputs.append(x)
        
        return outputs

def get_activations(images, model, batch_size=32, dims=2048, device='cpu'):
    """计算图像的激活值（特征）"""
    model.eval()
    
    if batch_size > len(images):
        print(f'警告: 批次大小({batch_size})大于图像数量({len(images)})。将批次大小设为{len(images)}')
        batch_size = len(images)
    
    pred_arr = np.empty((len(images), dims))
    
    for i in range(0, len(images), batch_size):
        batch = torch.stack([img.to(device) for img in images[i:i+batch_size]])
        
        # 获取模型输出
        with torch.no_grad():
            pred = model(batch)[0]
        
        # 如果是特征图，进行全局平均池化
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
        
        # 展平特征
        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        pred_arr[i:i+batch_size] = pred
    
    return pred_arr

def calculate_activation_statistics(images, model, batch_size=32, dims=2048, device='cpu'):
    """计算激活值的均值和协方差矩阵"""
    act = get_activations(images, model, batch_size, dims, device)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def load_images_from_folder(folder_path, transform=None):
    """从文件夹加载图像"""
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            img_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(img_path).convert('RGB')
                if transform:
                    img = transform(img)
                images.append(img)
            except Exception as e:
                print(f"无法加载图像 {img_path}: {e}")
    return images

def calculate_fid_given_paths(paths, batch_size=50, device='cpu', dims=2048):
    """计算两个图像文件夹之间的FID分数"""
    # 检查路径是否存在
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"路径不存在: {path}")
    
    # 定义图像预处理
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])
    
    # 加载图像
    print("加载图像...")
    images1 = load_images_from_folder(paths[0], transform)
    images2 = load_images_from_folder(paths[1], transform)
    
    if len(images1) == 0 or len(images2) == 0:
        raise ValueError("至少有一个文件夹不包含有效的图像")
    
    # 初始化Inception模型
    print("初始化Inception模型...")
    model = InceptionV3(output_blocks=[3], normalize_input=True).to(device)
    
    # 计算统计量
    print("计算统计量...")
    m1, s1 = calculate_activation_statistics(images1, model, batch_size, dims, device)
    m2, s2 = calculate_activation_statistics(images2, model, batch_size, dims, device)
    
    # 计算FID
    print("计算FID分数...")
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    
    return fid_value

# 使用示例
if __name__ == "__main__":
    # 设置参数
    paths = ['/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/FID_Score_Bench/real_images', '/home/sxm/flux-workspace/Qwen-Image-Lightning/expData/FID_GI_NAME']
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dims = 2048  # Inception-v3最后一层特征维度
    
    # 计算FID
    fid_score = calculate_fid_given_paths(paths, batch_size, device, dims)
    print(f"FID分数: {fid_score:.4f}")