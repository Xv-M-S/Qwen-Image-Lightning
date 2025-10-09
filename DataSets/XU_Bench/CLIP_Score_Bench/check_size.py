import os
from PIL import Image

def check_image_size(folder_path):
    # 获取文件夹下所有文件
    for filename in os.listdir(folder_path):
        # 构建完整路径
        file_path = os.path.join(folder_path, filename)
        
        # 检查是否为图片
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            try:
                with Image.open(file_path) as img:
                    width, height = img.size
                    # print(f"图片 {filename} 的大小为 {width}x{height}")
                    # 检查图片大小是否为512x512
                    if width != 512 or height != 512:
                        print(f"图片 {filename} 的大小不是 512x512，其实际大小为 {width}x{height}")
            except IOError:
                print(f"无法打开图片文件: {file_path}")

if __name__ == "__main__":
    folder = "/home/sxm/flux-workspace/Qwen-Image-Lightning/expData/sizeResGIName"
    folder = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/CLIP_Score_Bench/format_data/size_dataset/image"
    check_image_size(folder)