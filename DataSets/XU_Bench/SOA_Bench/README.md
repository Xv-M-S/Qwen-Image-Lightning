# intro
this the repo that try to change the SOA caculation method to fit QwenImage-Adaptor project.

# data prepare
## task : generate prompt and layout information
``` task
在文件夹 /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/semantic-object-accuracy-for-generative-text-to-image-synthesis/SOA/captions 中，存在着很多的.pkl文件，
文件目录如下：
$ tree
.
├── label_00_person.pkl
├── label_01_bicycle.pkl
├── label_02_car.pkl
├── label_03_motorcycle.pkl
├── label_04_plane.pkl
├── label_05_bus.pkl
├── label_06_train.pkl
在每个pkl文件中，其结构如下：
{'idx': [40414, 3], 'image_id': 44045, 'id': 392108, 'caption': 'A bicycle leaned up on a stop sign.'}
{'idx': [40414, 4], 'image_id': 44045, 'id': 393533, 'caption': 'A bicycle is leaning against a stop sign.'}
{'idx': [40462, 1], 'image_id': 57323, 'id': 476595, 'caption': 'A group of bikers riding bikes down a street.'}
提取其中的image_id 和 caption, 做以下两件事：
1. 根据image_id从/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/FID_Score_Bench/cocoapi/coco_dataset/raw/annotations/captions_train2014.json 和 /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/FID_Score_Bench/cocoapi/coco_dataset/raw/annotations/captions_val2014.json中获取caption，比对获取的caption和pkl文件中的caption是否一致，不一致则打印出警告，但是继续执行
2. 根据image_id从/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/FID_Score_Bench/cocoapi/coco_dataset/raw/annotations/instances_train2014.json 和 /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/FID_Score_Bench/cocoapi/coco_dataset/raw/annotations/instances_val2014.json中找到对应的layout信息
3. 将caption和layout信息保存成pickle文件，格式为：
{'prompt': 'a real scene of desert road with a sign written on it "a bicycle and toaster make a strange combination."', '0': {'description': 'desert road', 'mask': [0, 0, 512, 384]}, '1': {'description': 'sign', 'mask': [180, 384, 332, 460]}, '2': {'description': '"a bicycle and toaster make a strange combination."', 'mask': [190, 410, 322, 450]}, '3': {'description': 'bicycle', 'mask': [50, 300, 150, 384]}, '4': {'description': 'toaster', 'mask': [362, 320, 462, 384]}}
最后将这些处理的文件保存在一个新的文件夹下，目录结构与文件名与源文件夹类似。box的格式希望是归一化的[x, y, width, height]格式。
```


## run the script

``` bash
python collect_layout.py
```

## task : pick small batch of datasets

``` txt
1. 在文件夹/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/processed_captions_with_layout下，
其目录结构如下：
(qwenImageLightEnv) sxm@lkshpc:~/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/processed_captions_with_layout$ tree
.
├── label_00_person.pkl
├── label_01_bicycle.pkl
├── label_02_car.pkl
├── label_03_motorcycle.pkl
├── label_04_plane.pkl
├── label_05_bus.pkl
├── label_06_train.pkl
├── label_07_truck.pkl
├── label_08_boat.pkl
对于其中每个文件的内容格式如下：
{'prompt': 'a statue of two people surfing in a fountain', '0': {'description': 'person', 'mask': [0.419234375, 0.6524375, 0.024265625, 0.058312499999999996]}, '1': {'description': 'surfboard', 'mask': [0.42471875, 0.6876458333333333, 0.193828125, 0.1393125]}, '2': {'description': 'person', 'mask': [0.43987499999999996, 0.6478958333333333, 0.033078125, 0.06010416666666667]}, '3': {'description': 'person', 'mask': [0.11834375, 0.6568958333333333, 0.0418125, 0.057708333333333334]}, '4': {'description': 'person', 'mask': [0.924453125, 0.6563333333333333, 0.02484375, 0.0464375]}, '5': {'description': 'surfboard', 'mask': [0.12440625000000001, 0.6831458333333333, 0.11703125, 0.0785]}}

2. 现在需要从每个文件中随机挑出50条数据，并将挑出的文件存储在新的文件夹下，同时mask需要进行格式转换，转换为分辨率为512X512的格式且坐标分别为左上角坐标和右下角坐标的[X1, Y1, X2, Y2]格式。
```

``` bash
python pick_data.py
```

## task : generate global_id_map
``` txt
在文件夹 /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/semantic-object-accuracy-for-generative-text-to-image-synthesis/SOA/captions 中，存在着很多的.pkl文件，
文件目录如下：
$ tree
.
├── label_00_person.pkl
├── label_01_bicycle.pkl
├── label_02_car.pkl
├── label_03_motorcycle.pkl
├── label_04_plane.pkl
├── label_05_bus.pkl
├── label_06_train.pkl
在每个pkl文件中，其结构如下：
{'idx': [40414, 3], 'image_id': 44045, 'id': 392108, 'caption': 'A bicycle leaned up on a stop sign.'}
{'idx': [40414, 4], 'image_id': 44045, 'id': 393533, 'caption': 'A bicycle is leaning against a stop sign.'}
{'idx': [40462, 1], 'image_id': 57323, 'id': 476595, 'caption': 'A group of bikers riding bikes down a street.'}
读取文件夹中的所有.pkl文件，并生成一个全局的id映射表，格式为：
分别为{image_id:caption}和{caption:image_id},
{image_id:caption}和{caption:image_id}每种映射单独生成一个json文件保存
```


``` bash
python generate_id_map.py
```

## task : generate box labels

``` txt
给定一个文件夹/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/sampled_captions_512res
.
├── label_00_person.pkl
├── label_01_bicycle.pkl
├── label_02_car.pkl
├── label_03_motorcycle.pkl
├── label_04_plane.pkl
├── label_05_bus.pkl
├── label_06_train.pkl
├── label_07_truck.pkl
├── label_08_boat.pkl

其中每个文件的格式如下：
{'prompt': 'A person is surfing on the waves of an empty ocean.', '0': {'description': 'person', 'mask': [199, 234, 220, 273], 'category_id': 1}, '1': {'description': 'surfboard', 'mask': [193, 272, 225, 283], 'category_id': 42}, '2': {'description': 'boat', 'mask': [398, 228, 424, 239], 'category_id': 9}}
{'prompt': 'A young child claps his hands at the "sculpture" he made.', '0': {'description': 'refrigerator', 'mask': [372, 0, 509, 313], 'category_id': 82}, '1': {'description': 'person', 'mask': [116, 110, 320, 445], 'category_id': 1}, '2': {'description': 'wine glass', 'mask': [46, 62, 76, 109], 'category_id': 46}, '3': {'description': 'bowl', 'mask': [294, 354, 467, 461], 'category_id': 51}, '4': {'description': 'cup', 'mask': [315, 9, 407, 207], 'category_id': 47}, '5': {'description': 'cup', 'mask': [0, 67, 22, 106], 'category_id': 47}, '6': {'description': 'bowl', 'mask': [481, 265, 512, 312], 'category_id': 51}, '7': {'description': 'dining table', 'mask': [77, 369, 322, 505], 'category_id': 67}, '8': {'description': 'dining table', 'mask': [77, 294, 512, 512], 'category_id': 67}, '9': {'description': 'chair', 'mask': [29, 312, 163, 512], 'category_id': 62}}
读取其中的每一行数据做如下处理：
1. 首先从/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/mappings/caption_to_image_id.json读取caption到id的映射表
2. 根据prompt获取映射的id加上".png"的后缀作为图片名称,box的格式需要从: [x1, y1, x2, y2] 转换成归一化的yolo格式[x, y, width, height],其中w和h都为512
3. 转换成如下格式{"{id} + '.png'": [[], [category_id, category_id], [[0.1, 0.1, 0.3, 0.5], [0.6, 0.2, 0.2, 0.4]]]}
将处理后的数据存储在一个新的文件夹下，文件名称命名类似于之前文件夹，但是后缀加上ground_truth
```

``` bash
python /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/semantic-object-accuracy-for-generative-text-to-image-synthesis/getGroundTruth.py
```
# use /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/sampled_captions_512res to generate images

TODO : this step need to write a script to run batchRun.py for each floder.
现在有一个文件夹/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/sampled_captions_512res，里面全是.pkl文件，现在需要基于这些pkl文件，分批运行 batchRun.py 脚本生成图片，
脚本 batchRun.py 的作用是生成图片，具体使用方法如下：
python batchRun.py --data_dir /path/to/.pkl --save_path /path/to/save_image,
请写一个python脚本分批运行batchRun.py生成对应的图片。