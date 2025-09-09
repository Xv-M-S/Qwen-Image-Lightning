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
```

# download model

``` bash
pip install peft
pip install "huggingface_hub[cli]"
huggingface-cli download lightx2v/Qwen-Image-Lightning --local-dir ./Qwen-Image-Lightning
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
