# intro
this is a repo shows how to use coco data and api to caclulate the FID

# download coco data
下载test数据集
``` bash
modelscope download \
  --dataset OmniData/COCO_2014 \
  --include "raw/val2014.zip" "raw/annotations_trainval2014.zip" \
  --local_dir ./coco_dataset
```
下载train数据集
``` bash
modelscope download \
  --dataset OmniData/COCO_2014 \
  --include "raw/train2014.zip" \
  --local_dir ./coco_dataset
```
# coco api download
``` bash
git clone git@github.com:cocodataset/cocoapi.git
```

## task: generate caption and layout information【输入豆包】
使用 pycocotools.coco 模块，读取instances_val2014.json和captions_val2014.json文件，针对每个图片，生成对应的描述和布局信息。输出例子如下：
{'prompt': 'a real scene of desert road with a sign written on it "a bicycle and toaster make a strange combination."', '0': {'description': 'desert road', 'mask': [0, 0, 512, 384]}, '1': {'description': 'sign', 'mask': [180, 384, 332, 460]}, '2': {'description': '"a bicycle and toaster make a strange combination."', 'mask': [190, 410, 322, 450]}, '3': {'description': 'bicycle', 'mask': [50, 300, 150, 384]}, '4': {'description': 'toaster', 'mask': [362, 320, 462, 384]}}

run the script to generate the prompt and layout information
``` bash
python coco_prompt_layout.py
```

# use qwen-image-improved to generate image

batch run:
``` bash
CUDA_VISIBLE_DEVICES=2 nohup python batchRun.py \
  --data_dir /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/FID_Score_Bench/coco_prompt_layout.p \
  --save_path /home/sxm/flux-workspace/Qwen-Image-Lightning/expData/FID_GI \
  --global_id_map /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/FID_Score_Bench/global_id_map.json \
  > fid.log 2>&1 &
```


# 获取原始图片
给定一个json文件，其原始格式如下：
{
  "a man with a red helmet on a small moped on a dirt road.": "391895",
  "a woman wearing a net on her head cutting a cake.": "522418",
  "a child holding a flowered umbrella and petting a yak.": "184613",
  "a young boy standing in front of a computer keyboard.": "318219",
从中读取所有的id，基于该id与原始路径/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/FID_Score_Bench/cocoapi/coco_dataset/raw/val2014拼接成图片路径，图片名称格式为COCO_val2014_000000391895.jpg,
然后将所有的相关图片复制到一个新的文件夹，名字叫做real_images