pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 torchaudio==2.3.1+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html
pip install -r requirements.txt
pip install protobuf fonttools imgaug opencv-python pyyaml regex scipy
mkdir benchmark/pretrained_model_weight/
cd benchmark/pretrained_model_weight/
wget https://github.com/dandelin/ViLT/releases/download/200k/vilt_200k_mlm_itm.ckpt
git clone https://huggingface.co/bert-base-uncased
cd ../..
mkdir fedtask