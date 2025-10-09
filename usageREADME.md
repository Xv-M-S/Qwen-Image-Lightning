# env download

``` bash
conda create -n qwenImageLightEnv python=3.10
conda activate qwenImageEnvLight
# 安装最新的 diffusers
# pip install git+https://github.com/huggingface/diffusers
git clone git@github.com:huggingface/diffusers.git
cd diffusers
pip install -e .

# 安装的transformers>=4.51.3 
pip install transformers==4.55.0

# 安装其他安装包
pip install -r requirements.txt

pip install matplotlib
pip install torchviz
pip install scikit-learn
pip install pyinstrument

pip install accelerate
pip install nltk
# 手动下载punkt包 https://blog.csdn.net/qq_29831163/article/details/105145341?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-1.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-1.control
```

# download model

``` bash
pip install peft
pip install "huggingface_hub[cli]"
huggingface-cli download lightx2v/Qwen-Image-Lightning --local-dir ./Qwen-Image-Lightning
# for v2 model
# there sometimes error for network problem, just try again
huggingface-cli download lightx2v/Qwen-Image-Lightning --local-dir ./Qwen-Image-Lightning_v2
```

# run and test

## Run base Model

#50 steps, cfg 4.0
``` bash
python generate_with_diffusers.py \
--prompt_list_file examples/prompt_list.txt \
--out_dir test_base_results \
--base_seed 42 --steps 50 --cfg 4.0
```

## Run 4-step Model
4 steps, cfg 1.0

``` bash
python generate_with_diffusers.py \
--prompt_list_file examples/prompt_list.txt \
--out_dir test_lora_4_step_results \
--lora_path Qwen-Image-Lightning/Qwen-Image-Lightning-4steps-V1.0.safetensors \
--base_seed 42 --steps 4 --cfg 1.0
```

## Run 8-step Model
8 steps, cfg 1.0

``` bash
python generate_with_diffusers.py \
--prompt_list_file examples/prompt_list.txt \
--out_dir test_lora_8_step_results \
--lora_path Qwen-Image-Lightning/Qwen-Image-Lightning-8steps-V1.0.safetensors \
--base_seed 42 --steps 8 --cfg 1.0
```

## Run box loss generation
``` bash
python boxLossControlStart.py
```

# batch run
run the spatial result:
``` bash
python batchRun.py --data_dir /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/LLM_Bench/spatial.p --save_path /home/sxm/flux-workspace/Qwen-Image-Lightning/expData/spatialResGI
export CUDA_VISIBLE_DEVICES=1
nohup python batchRun.py --data_dir /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/LLM_Bench/spatial.p --save_path /home/sxm/flux-workspace/Qwen-Image-Lightning/expData/spatialResGI > spatial_run.log 2>&1 &
```

run the size result:
``` bash
export CUDA_VISIBLE_DEVICES=1
python batchRun.py --data_dir /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/LLM_Bench/size.p --save_path /home/sxm/flux-workspace/Qwen-Image-Lightning/expData/sizeResGI
nohup python batchRun.py --data_dir /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/LLM_Bench/size.p --save_path /home/sxm/flux-workspace/Qwen-Image-Lightning/expData/sizeResGI > size_run.log 2>&1 &
```

run the color result:
``` bash
export CUDA_VISIBLE_DEVICES=2
python batchRun.py --data_dir /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/LLM_Bench/color.p --save_path /home/sxm/flux-workspace/Qwen-Image-Lightning/expData/colorResGI
```

run the counting result:
``` bash
export CUDA_VISIBLE_DEVICES=3
python batchRun.py --data_dir /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/LLM_Bench/counting.p --save_path /home/sxm/flux-workspace/Qwen-Image-Lightning/expData/countingResGI
```

# eval : FID score
reference: https://xuexutao.github.io/2025/07/01/2025/2507/FID/ [Inception-v3的预训练模型已经集成到torch当中]

# eval : CLIP score
reference: https://github.com/Taited/clip-score
zhihu : https://zhuanlan.zhihu.com/p/645816974

# eval : SOA score
reference: https://github.com/tohinz/semantic-object-accuracy-for-generative-text-to-image-synthesis