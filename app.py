import cv2
import torchvision.transforms.functional as TF

import os
import sys
import matplotlib.pyplot as plt
import numpy as np

import streamlit as st
from PIL import Image
import torch
from model.model import ExpertTransformer
from modules.tokenizer import Tokenizer
from Args import Args
from modules.utils import get_inference_transform


st.markdown(
    """
    <style>
        
        /* Make button red */
        button {
            background-color: white !important;
            color: blue !important;
            border: 2px solid blue !important;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-weight: bold;
            cursor: pointer;
        }

        /* Make caption output appear in a green box */
        .green-box {
            background-color: #e6ffe6;
            border-left: 6px solid #33cc33;
            padding: 1rem;
            border-radius: 10px;
            margin-top: 1rem;
            font-size: 1.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)



# Load model and processor
@st.cache_resource
def load_model():
    args = Args()
    transform = get_inference_transform(args)
    tokenizer = Tokenizer(args)
    tokenizer.load_vocab('data/vocab.json')
    model = ExpertTransformer(args=args,tokenizer=tokenizer)
    checkpoint_path = os.path.join(args.project_root,args.from_pretrained)
    checkpoint = torch.load(checkpoint_path,map_location=args.device)
    model.load_state_dict(checkpoint['model'])
    model = model.to(args.device)

    return transform, model, tokenizer, args

transform, model, tokenizer, args = load_model()

st.title(" Retinal Image Captioning ")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
keywords = st.text_input("Enter keywords (comma-separated):", "")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2, col3 = st.columns([1, 8, 1])
    with col2:
        st.image(image, caption="Uploaded Image", width=500)
if st.button("Generate"):
    # Generate caption
    with st.spinner("Generating caption..."):
        keywords_list = [kw.strip() for kw in keywords.split(',') if kw.strip()]
        if keywords_list:
            raw_keywords = f" <SEP> ".join(keywords_list)  
        else:
            raw_keywords = "<SEP>" 
        
        keyword_tokens = tokenizer.encode_keywords(raw_keywords)
        keyword_tokens = torch.tensor(keyword_tokens, dtype=torch.long)
        keyword_tokens = keyword_tokens.unsqueeze(0)
        keyword_tokens = keyword_tokens.to(args.device)
        image = transform(image).clone().detach()
        image = image.unsqueeze(0)
        image = image.to(args.device)
        model.eval()
        with torch.cuda.amp.autocast():

            out = model.generate(image,keyword_tokens)
            out = out[0]
    


    st.markdown(f'<div class="green-box">{out}</div>', unsafe_allow_html=True)


