git clone https://github.com/Hieu3333/D-ExpertTransformer.git
cd D-ExpertTransformer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/miniconda3/bin/activate
conda init
source ~/.bashrc
conda env create -f env.yaml
mkdir data


scp -P 53588 "/mnt/c/Users/hieu3/Downloads/DeepEyeNet.zip" root@143.55.45.86:/workspace/D-ExpertTransformer

sudo apt update && sudo apt install unzip -y
unzip DeepEyeNet.zip -d data
