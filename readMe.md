## Setup
Install all required packages using conda:

conda env create -f env.yaml

## Training
To train the model, first you need to download the DeepEyeNet dataset from this link: https://github.com/Jhhuangkay/DeepOpht-Medical-Report-Generation-for-Retinal-Images-via-Deep-Models-and-Visual-Explanation

After downloading the dataset, put it into the "data" folder.

Then, use this command to train the model on GPU environment:

bash scripts/train_effnet_eye.sh 

## Streamlit 
To run the platform, use the command:
streamlit run app.py


