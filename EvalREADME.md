# counting eval
1. generate the counting result:
``` bash
export CUDA_VISIBLE_DEVICES=3
python batchRun.py --data_dir /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/LLM_Bench/counting.p --save_path /home/sxm/flux-workspace/Qwen-Image-Lightning/expData/countingResGI
```
2. mv the generated images to the counting_results folder
num : 509
``` bash
cp -r  /home/sxm/flux-workspace/Qwen-Image-Lightning/expData/countingResGI/counting_*_generated_image*  /home/sxm/flux-workspace/Qwen-Image-Lightning/expData/countingResGIName
```

3. Run the inference code to generate the bounding boxes and save them as follows:
``` bash
cd /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/detection/UniDet-master
conda activate xu_bench
unset PYTORCH_CUDA_ALLOC_CONF

python demo.py --config-file /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/detection/UniDet-master/configs/Partitioned_COI_RS101_2x.yaml \
--input /home/sxm/flux-workspace/Qwen-Image-Lightning/expData/countingResGIName/* --pkl_pth /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/detection/UniDet-master/pklDir/countingResName.pkl \
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
nohup python batchRun.py --data_dir /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/LLM_Bench/size.p --save_path /home/sxm/flux-workspace/Qwen-Image-Lightning/expData/sizeResGI > size_run.log 2>&1 &
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
nohup python batchRun.py --data_dir /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/LLM_Bench/spatial.p --save_path /home/sxm/flux-workspace/Qwen-Image-Lightning/expData/spatialResGI > spatial_run.log 2>&1 &
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
python batchRun.py --data_dir /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/LLM_Bench/color.p --save_path /home/sxm/flux-workspace/Qwen-Image-Lightning/expData/colorResGI
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
  --save_path /home/sxm/flux-workspace/Qwen-Image-Lightning/expData/FID_GI \
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
python fid_score.py --gpu 1 --batch-size 24 --path1 /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/FID_Score_Bench/real_images --path2 /home/sxm/flux-workspace/Qwen-Image-Lightning/expData/FID_GI_NAME
```


# tifa eval

1. use prompt to generate the data
``` bash
export CUDA_VISIBLE_DEVICES=3
nohup python -u batchRun.py --data_dir /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/Tifa_Bench/tifa_v1.0/qwen_coco_combined.p --save_path /home/sxm/flux-workspace/Qwen-Image-Lightning/expData/tifaResGI --global_id_map /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/Tifa_Bench/tifa_v1.0/caption_to_id.json > tifa_run.log 2>&1 &
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
nohup python mini_batch_gen_image.py > gen_image_log.txt 2>&1 &
```