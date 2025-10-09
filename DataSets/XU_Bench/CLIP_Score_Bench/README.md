# intro
this is a repo shows how to caculate the CLIP score

ref repo : https://github.com/Taited/clip-score
zhihu : https://zhuanlan.zhihu.com/p/645816974
# env download

``` bash
# install pytorch
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113 # Choose a version that suits your GPU
# install CLIP
pip install openai-clip
# Install clip-score from PyPI:
pip install clip-score
# sxm has change the clip-score caculate method
# [Optional] If you want to modify this project on your own, you can install it with:
git clone https://github.com/Taited/clip-score && cd clip-score
pip install -e .
# 拉取clip-score
git clone https://github.com/Taited/clip-score

unset PYTORCH_CUDA_ALLOC_CONF
```

# data format
对数据集的格式有要求：
Below is an example of the expected directory structure:

├── path/to/image
│   ├── cat.png
│   ├── dog.png
│   └── bird.jpg
└── path/to/text
    ├── cat.txt
    ├── dog.txt
    └── bird.txt

## task: 格式化输入数据
从文件夹/home/sxm/flux-workspace/Qwen-Image-Lightning/expData/colorResGIName中读取图片数据，重命名
其文件名为“_”分割的前两个字符，例如colors_1_generated_image.png希望重命名为colors_1.png
同时读取json文件/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/HRS/key_value_mapping_id_prompt.json，根据colors_1的名称获取对应的prompt，将prompt写入到colors_1.txt文件中，单行

处理后数据集的格式如下：
Below is an example of the expected directory structure:

├── path/to/image
│   ├── cat.png
│   ├── dog.png
│   └── bird.jpg
└── path/to/text
    ├── cat.txt
    ├── dog.txt
    └── bird.txt

## run the script to format the input data
``` bash
python dataGen.py
```

# use clip-score to caculate the CLIP score

``` bash
# 使用 clip-score仓库下的clip_score
python -m clip_score /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/CLIP_Score_Bench/format_data/color_dataset/image /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/CLIP_Score_Bench/format_data/color_dataset/text
```