git clone https://github.com/Hieu3333/D-ExpertTransformer.git
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/miniconda3/bin/activate
conda init
source ~/.bashrc
cd D-ExpertTransformer
conda env create -f env.yaml
conda activate .env
sudo apt-get install default-jre
sudo apt-get install default-jdk
cd D-ExpertTransformer
sudo apt update && sudo apt install unzip -y
pip install timm
pip install numpy==1.21.2
pip install transformers
pip install tokenizers
pip install datasets
pip install pycocoevalcap
sudo apt update
sudo apt install git-lfs
git lfs install
mkdir data


scp -P 41073 "/mnt/c/Users/hieu3/Downloads/DeepEyeNet.zip" root@50.217.254.167:/workspace/D-ExpertTransformer
scp -P 41073 "/mnt/c/D-ExpertTransformer/data/cleaned_DeepEyeNet_train.json" root@50.217.254.167:/workspace/D-ExpertTransformer/data
scp -P 41073 "/mnt/c/D-ExpertTransformer/data/cleaned_DeepEyeNet_test.json" root@50.217.254.167:/workspace/D-ExpertTransformer/data
scp -P 41073 "/mnt/c/D-ExpertTransformer/data/cleaned_DeepEyeNet_val.json" root@50.217.254.167:/workspace/D-ExpertTransformer/data
scp -P 41073 "/mnt/c/D-ExpertTransformer/data/vocab.json" root@50.217.254.167:/workspace/D-ExpertTransformer/data

scp -P 45783 "/mnt/c/Users/hieu3/Downloads/iu_xray.zip" root@185.65.93.114:/workspace/D-ExpertTransformer

unzip DeepEyeNet.zip -d data
unzip iu_xray.zip -d data





scp -P 41073 root@50.217.254.167:/workspace/D-ExpertTransformer/results/effnet-diff/efficientnet_deepeyenet.pth "/mnt/c/D-ExpertTransformer/results/diffDA"
scp -P 41073 -r root@50.217.254.167:/workspace/D-ExpertTransformer/logs/effnet-diff/ "/mnt/c/D-ExpertTransformer/logs/diffDA"

scp -P 41073 root@50.217.254.167:/workspace/D-ExpertTransformer/results/resnet-diff-roco/roco.pth "/mnt/c/D-ExpertTransformer/results/roco"
scp -P 41073 -r root@50.217.254.167:/workspace/D-ExpertTransformer/roco/logs/resnet-diff-roco/ "/mnt/c/D-ExpertTransformer/roco/logs"

scp -P 41073 root@50.217.254.167:/workspace/D-ExpertTransformer/results/resnet-iu/iu_xray.pth "/mnt/c/D-ExpertTransformer/results/iu_xray"
scp -P 41073 -r root@50.217.254.167:/workspace/D-ExpertTransformer/roco/logs/resnet_iu/ "/mnt/c/D-ExpertTransformer/iu_xray/logs"