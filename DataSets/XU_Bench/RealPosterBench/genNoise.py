from PIL import Image
import random

# 设置图像尺寸为512x512
width, height = 512, 512

# 创建RGB模式的图像（3个颜色通道）
image = Image.new('RGB', (width, height))

# 获取像素操作对象
pixels = image.load()

# 为每个像素生成随机的RGB值
for i in range(width):
    for j in range(height):
        # 红、绿、蓝三个通道分别生成0-255的随机值
        red = random.randint(0, 255)
        green = random.randint(0, 255)
        blue = random.randint(0, 255)
        # 给当前像素赋值
        pixels[i, j] = (red, green, blue)

# 保存生成的彩色噪声图像
image.save('random_color_noise_512x512.png')
print("512x512随机彩色噪声图像已保存为 'random_color_noise_512x512.png'")

# 可选：显示图像
# image.show()