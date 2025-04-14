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
sudo apt update
sudo apt install git-lfs
git lfs install
mkdir data


scp -P 54264 "/mnt/c/Users/hieu3/Downloads/DeepEyeNet.zip" root@143.55.45.86:/workspace/D-ExpertTransformer
scp -P 54264 "/mnt/c/D-ExpertTransformer/data/cleaned_DeepEyeNet_test.json" root@143.55.45.86:/workspace/D-ExpertTransformer/data
scp -P 54264 "/mnt/c/D-ExpertTransformer/data/cleaned_DeepEyeNet_train.json" root@143.55.45.86:/workspace/D-ExpertTransformer/data
scp -P 54264 "/mnt/c/D-ExpertTransformer/data/cleaned_DeepEyeNet_val.json" root@143.55.45.86:/workspace/D-ExpertTransformer/data
scp -P 54264 "/mnt/c/D-ExpertTransformer/data/vocab.json" root@143.55.45.86:/workspace/D-ExpertTransformer/data

scp -P 54264 "/mnt/c/Users/hieu3/Downloads/iu_xray.zip" root@143.55.45.86:/workspace/D-ExpertTransformer

unzip DeepEyeNet.zip -d data
unzip iu_xray.zip -d data

mkdir -p /workspace/D-ExpertTransformer/pycocoevalcap/meteor/data/
cd /workspace/D-ExpertTransformer/pycocoevalcap/meteor/data/
wget https://raw.githubusercontent.com/cmu-mtlab/meteor/master/data/paraphrase-en.gz



scp -P 24156 root@185.150.27.254:/workspace/D-ExpertTransformer/results/resnet-diff-iu-combine/iu_xray.pth "/mnt/c/D-ExpertTransformer/results/iu_xray"

scp -P 24156 -r root@185.150.27.254:/workspace/D-ExpertTransformer/roco/logs/resnet-diff-roco/ "/mnt/c/D-ExpertTransformer/roco/logs"