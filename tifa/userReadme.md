# env download
ref repo : https://github.com/Yushi-Hu/tifa
``` bash
conda create -n tifaEnv python=3.10 -y
conda activate tifaEnv
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install pip==24.0

git clone git@github.com:facebookresearch/fairseq.git
cd fairseq
git checkout v0.12.2 
pip install --editable ./

cd ..
pip install -r requirements.txt
pip install -e .
pip install soxr
pip install transformers==4.31.0
pip install "modelscope==1.28.0"

unset PYTORCH_CUDA_ALLOC_CONF

Changed metrics import location on speech_dlm_criterion:https://github.com/facebookresearch/fairseq/pull/5365
fix:https://github.com/facebookresearch/fairseq/pull/5365/commits/8daf5fbd4d6af7c550d19135cabb23ed068b03b6
```