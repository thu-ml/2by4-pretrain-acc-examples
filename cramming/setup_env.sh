conda create -n pytorch python=3.11 -y && conda activate pytorch

pip3 install torch torchvision torchaudio

pip install packaging

pip install -r requirements.txt

MAX_JOBS=4 pip install flash-attn==1.0.8 --no-build-isolation

wandb login your_id
