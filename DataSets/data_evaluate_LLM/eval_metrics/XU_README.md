# env download
``` bash
conda create -n xu_bench python=3.10 -y

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

# issue sovle : https://github.com/facebookresearch/detectron2/issues/5453
git clone git@github.com:facebookresearch/detectron2.git
pip install setuptools==75.8.0
unset PYTORCH_CUDA_ALLOC_CONF
python -m pip install -e detectron2
pip install numpy==1.26.4
# pip install opencv-python==4.12.0.88 # 支持numpy2.0
 pip install opencv-python==4.7.0.72
pip install pandas
```

ps:需要cuda11.3的环境
参考教程：
``` bash
wget https://developer.download.nvidia.com/compute/cuda/11.3.1/local_installers/cuda_11.3.1_465.19.01_linux.run
sudo sh cuda_11.3.1_465.19.01_linux.run
```

weights and configs download:
weights : https://drive.google.com/file/d/110JSpmfNU__7T3IMSJwv0QSfLLo_AqtZ/edit
configurations : https://github.com/xingyizhou/UniDet/blob/master/configs/Partitioned_COI_RS101_2x.yaml

# run

## 构建全局唯一的id，方便评估

``` bash
cd /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/HRS
python global_id.py
```

## Inference code

``` bash
cd data_evaluate_LLM/eval_metrics/detection/UniDet-master
python demo.py --config-file /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/detection/UniDet-master/configs/Partitioned_COI_RS101_2x.yaml \
--input /home/sxm/flux-workspace/Qwen-Image-Lightning/expData/countingResGIName/* --pkl_pth /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/detection/UniDet-master/pklDir/countingResName.pkl \
--output /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/detection/UniDet-master/DetectRes/countingResName --opts MODEL.WEIGHTS /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/detection/UniDet-master/weight/Partitioned_COI_RS101_2x.pth
```

the pkl save format:
``` bash
101: {              # 外层 key：可能是视频帧编号或图像批次索引
    101: {           # 内层 key：可能是图像 ID 或子索引
        0: [         # 第 0 个检测目标组（可能有多个重复框）
            array(['5.8123145', '310.17346', '329.87836', '438.57675', 'sink'], dtype='<U32'),
            array(['5.8123145', '310.17346', '329.87836', '438.57675', 'sink'], dtype='<U32')
        ],
        1: [         # 第 1 个检测目标
            array(['223.40253', '267.19888', '468.3134', '351.00113', 'sink'], dtype='<U32')
        ],
        2: [         # 第 2 个检测目标
            array(['1.1367968', '254.55527', '494.67654', '441.85886', 'sink'], dtype='<U32')
        ]
    }
}
```

## Counting accuracy
Run the calc_counting_acc.py to calculate the counting accuracy, as follows:

``` bash
cd data_evaluate_LLM/eval_metrics/counting
python calc_counting_acc.py /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/detection/UniDet-master/pklDir/mixRes.pkl [GT-csv] [Number of Iteration]
```

``` bash
cd data_evaluate_LLM/eval_metrics/counting
python /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/counting/calc_counting_acc.py "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/detection/UniDet-master/pklDir/countingResName.pkl" /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/HRS/counting_prompts.csv 1
```

## Spatial Composition accuracy

it can be called by the following command:
``` bash
python calc_spatial_relation_acc.py "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/detection/UniDet-master/pklDir/spatialRes.pkl" /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/HRS/spatial_compositions_prompts.csv 1
```

## Size accuracy

it can be called by the following command:
``` bash
python calc_size_comp_acc.py "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/detection/UniDet-master/pklDir/sizeRes.pkl" /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/HRS/size_compositions_prompts.csv 1
```

## Color accuracy
it can be called by the following command:

### env download
reference : https://github.com/IDEA-Research/MaskDINO/blob/main/INSTALL.md
``` bash
pip install timm
pip install scipy
```

CUDA kernel for MSDeformAttn
``` bash
cd maskdino/modeling/pixel_decoder/ops
sh make.sh
```

### run

run to generate the mask:[MaskDINO对于分割超大物体的能力很差，导致最终结果很差，需要更换分割模型]

``` bash  
cd data_evaluate_LLM/eval_metrics/colors/MaskDINO/demo
python /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/colors/MaskDINO/demo/demo.py --config-file /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/colors/MaskDINO/configs/coco/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048_no_maskEnhance.yaml --input '/home/sxm/flux-workspace/Qwen-Image-Lightning/expData/colorResGIName/*' --output /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/colors/MaskDINO/outputData/colorRes --opts MODEL.WEIGHTS /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/colors/MaskDINO/model/maskdino_swinl_50ep_300q_hid2048_3sd1_panoptic_58.3pq.pth
```

run to detect the color:
``` bash
cd data_evaluate_LLM/eval_metrics/colors/
python /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/colors/hue_based_color_classifier.py /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/colors/MaskDINO/outputData/colorRes /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/HRS/colors_composition_prompts.csv /home/sxm/flux-workspace/Qwen-Image-Lightning/expData/colorResGIName
```

