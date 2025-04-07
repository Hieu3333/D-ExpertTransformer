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


scp -P 23166 "/mnt/c/Users/hieu3/Downloads/DeepEyeNet.zip" root@192.165.134.27:/workspace/D-ExpertTransformer
scp -P 23166 "/mnt/c/D-ExpertTransformer/data/cleaned_DeepEyeNet_test.json" root@192.165.134.27:/workspace/D-ExpertTransformer/data
scp -P 23166 "/mnt/c/D-ExpertTransformer/data/cleaned_DeepEyeNet_train.json" root@192.165.134.27:/workspace/D-ExpertTransformer/data
scp -P 23166 "/mnt/c/D-ExpertTransformer/data/cleaned_DeepEyeNet_val.json" root@192.165.134.27:/workspace/D-ExpertTransformer/data
scp -P 23166 "/mnt/c/D-ExpertTransformer/data/vocab.json" root@192.165.134.27:/workspace/D-ExpertTransformer/data
scp -P 23166 "/mnt/c/Users/hieu3/Downloads/evaluation/pycocoevalcap/spice/cache/data.mdb" root@192.165.134.27:/workspace/D-ExpertTransformer/pycocoevalcap/spice/cache
scp -P 23166 "/mnt/c/Users/hieu3/Downloads/evaluation/pycocoevalcap/spice/lib/stanford-corenlp-3.6.0-models.jar" root@192.165.134.27:/workspace/D-ExpertTransformer/pycocoevalcap/spice/lib

scp -P 23228 "/mnt/c/Users/hieu3/Downloads/iu_xray.zip" root@192.165.134.27:/workspace/D-ExpertTransformer

unzip DeepEyeNet.zip -d data
unzip iu_xray.zip -d data

mkdir -p /workspace/D-ExpertTransformer/pycocoevalcap/meteor/data/
cd /workspace/D-ExpertTransformer/pycocoevalcap/meteor/data/
wget https://raw.githubusercontent.com/cmu-mtlab/meteor/master/data/paraphrase-en.gz


scp -P 23228 root@192.165.134.27:/workspace/D-ExpertTransformer/results/resnet-deepeyenet-diff-gca/val_result_epoch_30.json "/mnt/c/D-ExpertTransformer/"
scp -P 23228 root@192.165.134.27:/workspace/D-ExpertTransformer/results/resnet-deepeyenet-diff-gca/test_result_epoch_30.json "/mnt/c/D-ExpertTransformer/"