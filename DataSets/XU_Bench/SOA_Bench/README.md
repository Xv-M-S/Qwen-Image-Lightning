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

``` bash
nohup python mini_batch_gen_image.py --save_root /home/sxm/flux-workspace/Qwen-Image-Lightning/expData_v2/raw-qwen-image/soaResGI > gen_image_log.txt 2>&1 &
```

# 将生成的图片移动到一个新的文件夹

TASK: 一个文件夹的结构如下图所示:
(tifaEnv) sxm@lkshpc:~/flux-workspace/Qwen-Image-Lightning/expData_v2/raw-qwen-image/soaResGI$ tree -L 1
.
├── label_00_person
├── label_01_bicycle
├── label_02_car
├── label_03_motorcycle
├── label_04_plane
├── label_05_bus
├── label_06_train
├── label_07_truck
├── label_08_boat
├── label_09_trafficlight
├── label_10_hydrant
├── label_11_stopsign
├── label_12_parkingmeter
├── label_13_bench
├── label_14_bird
├── label_15_cat
├── label_16_dog
├── label_17_horse
├── label_18_sheep
├── label_19_cow
├── label_20_elephant
├── label_21_bear
├── label_22_zebra
├── label_23_giraffe
├── label_24_backpack
├── label_25_umbrella
├── label_26_handbag
写一个python脚本,新建一个新的文件夹,然后将生成的图片(特指文件名称中带有"generated"的图片")移动到这个文件夹下

``` bash
python /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/mvFile.py --source_dir /path/to/source_dir --new_folder /path/to/target_dir

python /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/mvFile.py --source_dir /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/ground_truth_results --new_folder /path/to/target_dir
```

# 将groundTruth移动到新的文件夹下
现在有一些groundtruth数据，格式如下：
(tifaEnv) sxm@lkshpc:~/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/ground_truth_results$ tree 
.
├── label_00_person_ground_truth.pkl
├── label_01_bicycle_ground_truth.pkl
├── label_02_car_ground_truth.pkl
├── label_03_motorcycle_ground_truth.pkl
├── label_04_plane_ground_truth.pkl
├── label_05_bus_ground_truth.pkl
├── label_06_train_ground_truth.pkl
├── label_07_truck_ground_truth.pkl
├── label_08_boat_ground_truth.pkl
├── label_09_trafficlight_ground_truth.pkl
现在需要将这些groundtruth数据复制一份到对应的文件夹下,文件夹的结构如下,需要将label_00_person_ground_truth.pkl复制一份到label_00_person文件夹下,依次类推:
.
├── label_00_person
├── label_01_bicycle
├── label_02_car
├── label_03_motorcycle
├── label_04_plane
├── label_05_bus
├── label_06_train
├── label_07_truck
├── label_08_boat
├── label_09_trafficlight
├── label_10_hydrant
├── label_11_stopsign
``` bash
python /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/mvBoxFile.py --ground_truth_dir /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/ground_truth_results --target_root_dir /path/to/target_dir
```
# 运行评估代码

``` bash
python calculate_soa.py --images /home/sxm/flux-workspace/Qwen-Image-Lightning/expData_v2/raw-qwen-image/soaResGINAME --output /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/semantic-object-accuracy-for-generative-text-to-image-synthesis/SOA/eval_res --gpu 0 --iou


python calculate_soa.py --images /home/sxm/flux-workspace/Qwen-Image-Lightning/expData/soaResGINAME --output /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/semantic-object-accuracy-for-generative-text-to-image-synthesis/SOA/eval_res_contorl --gpu 0 --iou
```

## 根据 ground truth 可视化生成的图片

``` bash
python /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/visualize_detections.py
```