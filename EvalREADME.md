# counting eval
1. generate the counting result:
``` bash
export CUDA_VISIBLE_DEVICES=3
python batchRun.py --data_dir /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/LLM_Bench/counting.p --save_path /home/sxm/flux-workspace/Qwen-Image-Lightning/expData_v2/raw-qwen-image/countingResGI
```
2. mv the generated images to the counting_results folder
num : 509
``` bash
mkdir countingResGINAME
cp -r  /home/sxm/flux-workspace/Qwen-Image-Lightning/expData_v2/raw-qwen-image/countingResGI/counting_*_generated_image*  /home/sxm/flux-workspace/Qwen-Image-Lightning/expData_v2/raw-qwen-image/countingResGINAME
```

3. Run the inference code to generate the bounding boxes and save them as follows:
``` bash
cd /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/detection/UniDet-master
conda activate xu_bench
unset PYTORCH_CUDA_ALLOC_CONF

python demo.py --config-file /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/detection/UniDet-master/configs/Partitioned_COI_RS101_2x.yaml \
--input /home/sxm/flux-workspace/Qwen-Image-Lightning/expData_v2/raw-qwen-image/countingResGINAME/* --pkl_pth /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/detection/UniDet-master/pklDir/countingResName.pkl \
--output /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/detection/UniDet-master/DetectRes/countingResName --opts MODEL.WEIGHTS /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/detection/UniDet-master/weight/Partitioned_COI_RS101_2x.pth
```

4. calculate the counting accuracy:

``` bash
cd /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/counting
python /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/counting/calc_counting_acc.py "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/detection/UniDet-master/pklDir/countingResName.pkl" /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/HRS/counting_prompts.csv 1
```

# size eval
1. generate the size result:
``` bash
export CUDA_VISIBLE_DEVICES=3
python batchRun.py --data_dir /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/LLM_Bench/size.p --save_path /home/sxm/flux-workspace/Qwen-Image-Lightning/expData/sizeResGI
nohup python batchRun.py --data_dir /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/LLM_Bench/size.p --save_path /home/sxm/flux-workspace/Qwen-Image-Lightning/expData_v2/raw-qwen-image/sizeResGI > size_run.log 2>&1 &
```

2. mv the generated images to the size_results folder
num : 375
``` bash
cp -r  /home/sxm/flux-workspace/Qwen-Image-Lightning/expData/sizeResGI/size_*_generated_image*  /home/sxm/flux-workspace/Qwen-Image-Lightning/expData/sizeResGIName
```

3. Run the inference code to generate the bounding boxes and save them as follows:
``` bash
cd /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/detection/UniDet-master
conda activate xu_bench
unset PYTORCH_CUDA_ALLOC_CONF

python demo.py --config-file /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/detection/UniDet-master/configs/Partitioned_COI_RS101_2x.yaml \
--input /home/sxm/flux-workspace/Qwen-Image-Lightning/expData/sizeResGIName/* --pkl_pth /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/detection/UniDet-master/pklDir/sizeResName.pkl \
--output /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/detection/UniDet-master/DetectRes/sizeResName --opts MODEL.WEIGHTS /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/detection/UniDet-master/weight/Partitioned_COI_RS101_2x.pth
```

4. calculate the size accuracy:

``` bash
cd /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/compositions

python calc_size_comp_acc.py "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/detection/UniDet-master/pklDir/sizeResName.pkl" /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/HRS/size_compositions_prompts.csv 1
```

# spatial eval

1. generate the spatial result:

``` bash
export CUDA_VISIBLE_DEVICES=1
nohup python batchRun.py --data_dir /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/LLM_Bench/spatial.p --save_path /home/sxm/flux-workspace/Qwen-Image-Lightning/expData_v2/raw-qwen-image/spatialResGI > spatial_run.log 2>&1 &
```

2. mv the generated images to the spatial_results folder
num : 399
``` bash
cp -r  /home/sxm/flux-workspace/Qwen-Image-Lightning/expData/spatialResGI/spatial_*_generated_image*  /home/sxm/flux-workspace/Qwen-Image-Lightning/expData/spatialResGIName
```

3. Run the inference code to generate the bounding boxes and save them as follows:
``` bash
cd /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/detection/UniDet-master
conda activate xu_bench
unset PYTORCH_CUDA_ALLOC_CONF

python demo.py --config-file /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/detection/UniDet-master/configs/Partitioned_COI_RS101_2x.yaml \
--input /home/sxm/flux-workspace/Qwen-Image-Lightning/expData/spatialResGIName/* --pkl_pth /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/detection/UniDet-master/pklDir/spatialResName.pkl \
--output /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/detection/UniDet-master/DetectRes/spatialResName --opts MODEL.WEIGHTS /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/detection/UniDet-master/weight/Partitioned_COI_RS101_2x.pth
```

4. calculate the spatial accuracy:

``` bash
cd /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/compositions
python calc_spatial_relation_acc.py "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/detection/UniDet-master/pklDir/spatialRes.pkl" /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/HRS/spatial_compositions_prompts.csv 1
```

# color eval [need change the seg-model]
1. generate the color result:
``` bash
export CUDA_VISIBLE_DEVICES=2
python batchRun.py --data_dir /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/LLM_Bench/color.p --save_path /home/sxm/flux-workspace/Qwen-Image-Lightning/expData_v2/raw-qwen-image/colorResGI
```

2. mv the generated images to the color_results folder
num : 484
``` bash
cp -r  /home/sxm/flux-workspace/Qwen-Image-Lightning/expData/colorResGI/colors_*_generated_image*  /home/sxm/flux-workspace/Qwen-Image-Lightning/expData/colorResGIName
```

3. Run the inference code to generate the bounding boxes and save them as follows:
``` bash 
cd /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/colors/MaskDINO

python /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/colors/MaskDINO/demo/demo.py --config-file /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/colors/MaskDINO/configs/coco/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048_no_maskEnhance.yaml --input '/home/sxm/flux-workspace/Qwen-Image-Lightning/expData/colorResGIName/*' --output /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/colors/MaskDINO/outputData/colorRes --opts MODEL.WEIGHTS /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/colors/MaskDINO/model/maskdino_swinl_50ep_300q_hid2048_3sd1_panoptic_58.3pq.pth
```
4. calculate the color accuracy:

``` bash
cd /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/colors

python /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/colors/hue_based_color_classifier.py /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/colors/MaskDINO/outputData/colorRes /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/HRS/colors_composition_prompts.csv /home/sxm/flux-workspace/Qwen-Image-Lightning/expData/colorResGIName
```

# CLIP eval
1. based on the image generated by upper steps.

``` bash
conda activate tifaEnv

cd /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/CLIP_Score_Bench
```

2. generate the images and texts

``` bash
python /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/CLIP_Score_Bench/dataGen.py \
  --raw-image-dir "/home/sxm/flux-workspace/Qwen-Image-Lightning/expData/colorResGIName" \
  --output-root "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/CLIP_Score_Bench/format_data/color_dataset"
```

3. generate the CLIP score:

``` bash
python -m clip_score /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/CLIP_Score_Bench/format_data/color_dataset/image /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/CLIP_Score_Bench/format_data/color_dataset/text
```



# FID eval
ref : https://juejin.cn/post/7148966703454486535
1. generate the FID result:

``` bash
CUDA_VISIBLE_DEVICES=2 nohup python batchRun.py \
  --data_dir /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/FID_Score_Bench/coco_prompt_layout.p \
  --save_path /home/sxm/flux-workspace/Qwen-Image-Lightning/expData_v2/raw-qwen-image/FID_GI \
  --global_id_map /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/FID_Score_Bench/global_id_map.json \
  > fid.log 2>&1 &
```

2. cp the generated images to the FID_results folder
``` bash
cp -r /home/sxm/flux-workspace/Qwen-Image-Lightning/expData/FID_GI/*generated* FID_GI_NAME/
```

3. calculate the FID score:

real_image : /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/FID_Score_Bench/real_images
generated_image : /home/sxm/flux-workspace/Qwen-Image-Lightning/expData/FID_GI_NAME
``` bash
cd /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/FID_Score_Bench
python fid.py
```


``` bash
cd /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/FID_Score_Bench/FID
python fid_score.py --gpu 1 --batch-size 24 --path1 /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/FID_Score_Bench/real_images --path2 /home/sxm/flux-workspace/Qwen-Image-Lightning/expData/FID_GI_NAME
```


# tifa eval

1. use prompt to generate the data
``` bash
export CUDA_VISIBLE_DEVICES=3
nohup python -u batchRun.py --data_dir /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/Tifa_Bench/tifa_v1.0/qwen_coco_combined.p --save_path /home/sxm/flux-workspace/Qwen-Image-Lightning/expData_v2/raw-qwen-image/tifaResGI --global_id_map /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/Tifa_Bench/tifa_v1.0/caption_to_id.json > tifa_run.log 2>&1 &
```

2. cp the generated images to the specified folder

``` bash
cp -r /home/sxm/flux-workspace/Qwen-Image-Lightning/expData/tifaResGI/*generated*  /home/sxm/flux-workspace/Qwen-Image-Lightning/expData/tifaResGINAME
```

3. generate a json file that describes the image path and the prompt.

给定一个文件夹/home/sxm/flux-workspace/Qwen-Image-Lightning/expData/tifaResGINAME，
读取文件夹中的所有图片，形成{图片id：图片路径的json}文件，格式如下：
{
    "coco_301091": "coco_301091.jpg",
    "drawbench_52": "drawbench_52.jpg"
}
其中图片id为图片名称中以“_”分割的前两个字符用“_”拼接起来，例如coco_293_generated_image.png
的图片id为coco_293

``` bash
python /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/Tifa_Bench/generate_image_json.py
```

4. calculate the tifa score:

``` bash
cd /home/sxm/flux-workspace/Qwen-Image-Lightning/tifa
python tifaSampleTest.py 
```

# SOA eval

``` bash
python batchRun.py --data_dir /path/to/.pkl --save_path /path/to/save_image
```

mini_batch_run:

``` bash
nohup python mini_batch_gen_image.py --save_root /home/sxm/flux-workspace/Qwen-Image-Lightning/expData_v2/raw-qwen-image/soaResGI > gen_image_log.txt 2>&1 &
```


## quick eval script

``` bash  
python AutoEval.py --image_save_path /home/sxm/flux-workspace/Qwen-Image-Lightning/expData

export CUDA_VISIBLE_DEVICES=3
unset PYTORCH_CUDA_ALLOC_CONF
nohup python AutoEval.py \
  --image_save_path /home/sxm/flux-workspace/Qwen-Image-Lightning/expData_v2/both_iTt_and_tTi_features_20_to_49_layer \
  > run.log 2>&1 &
```

## Soa Visualization
task description:
现在有一个文件夹，其内部文件架构如下：
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
在每个子文件夹中，都是图片，然后有一个pkl文件描述了每一个图片中物体的位置和名称，其内容结构如下：
{'112905.png': [[], [1, 41], [[0.456055, 0.498047, 0.560547, 0.949219], [0.139648, 0.743164, 0.244141, 0.451172]]]}, {'47285.png': [[], [44, 44, 59, 27, 1, 1, 1, 1, 1, 1], [[0.214844, 0.504883, 0.007812, 0.021484], [0.233398, 0.506836, 0.013672, 0.025391], [0.392578, 0.494141, 0.035156, 0.019531], [0.141602, 0.387695, 0.044922, 0.083984], [0.807617, 0.350586, 0.076172, 0.287109], [0.276367, 0.257812, 0.017578, 0.042969], [0.311523, 0.328125, 0.029297, 0.128906], [0.253906, 0.317383, 0.058594, 0.177734], [0.349609, 0.400391, 0.132812, 0.339844], [0.585938, 0.366211, 0.042969, 0.240234]]]}, {'391794.png': [[], [1, 77, 76, 47, 72, 74, 74, 77], [[0.654297, 0.499023, 0.691406, 0.998047], [0.202148, 0.398438, 0.126953, 0.089844], [0.210938, 0.72168, 0.316406, 0.556641], [0.360352, 0.086914, 0.080078, 0.169922], [0.024414, 0.165039, 0.048828, 0.330078], [0.439453, 0.503906, 0.082031, 0.082031], [0.330078, 0.446289, 0.09375, 0.087891], [0.345703, 0.198242, 0.105469, 0.060547]]]}, {'356293.png': [[], [1, 1, 34], [[0.462891, 0.619141, 0.085938, 0.226562], [0.24707, 0.623047, 0.220703, 0.621094], [0.456055, 0.143555, 0.060547, 0.076172]]]},
现在需要读取将这个文件可视化在每一个图片上，并存储在一个新文件夹中，类别的映射关系是类似coco的映射关系
例如person映射为0，bycle映射为1，car映射为2，motorcycle映射为3，plane映射为4，bus映射为5，train映射为6，truck映射为7，boat映射为8，trafficlight映射为9，hydrant映射为10，stopsign映射为11，parkingmeter映射为12，bench映射为13，bird映射为14，cat映射为15，dog映射为16，horse映射为17，sheep映射为18，cow映射为19，elephant映射为20，bear映射为21

``` bash

python /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/visualize_detections.py
```

problem: 发现一个问题，就是图片质量不高，如果是针对类目的一个评估
那么对于生成任务来说，一张图片里面特定类目的数量应该是比较少的才好；
因为coco数据集是用于评估目标检测的，可能会出现非常多的物体的场景，
对于生成任务来说，这样的评估场景并不是最理想的。
所以重构了数据集：限制了每张图片里面关键元素的数量，同时在生成过程中
清除了无关物体的一个布局box，使得文胜图模型在生成图片的时候更加聚焦于
关键元素的生成。