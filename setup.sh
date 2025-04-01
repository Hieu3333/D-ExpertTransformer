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
mkdir data


scp -P 23643 "/mnt/c/Users/hieu3/Downloads/DeepEyeNet.zip" root@192.165.134.27:/workspace/D-ExpertTransformer
scp -P 23643 "/mnt/c/D-ExpertTransformer/data/cleaned_DeepEyeNet_test.json" root@192.165.134.27:/workspace/D-ExpertTransformer/data
scp -P 23643 "/mnt/c/D-ExpertTransformer/data/cleaned_DeepEyeNet_train.json" root@192.165.134.27:/workspace/D-ExpertTransformer/data
scp -P 23643 "/mnt/c/D-ExpertTransformer/data/cleaned_DeepEyeNet_val.json" root@192.165.134.27:/workspace/D-ExpertTransformer/data

unzip DeepEyeNet.zip -d data

mkdir -p /workspace/D-ExpertTransformer/pycocoevalcap/meteor/data/
cd /workspace/D-ExpertTransformer/pycocoevalcap/meteor/data/
wget https://raw.githubusercontent.com/cmu-mtlab/meteor/master/data/paraphrase-en.gz