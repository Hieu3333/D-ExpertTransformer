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

# def overlay_heatmap_on_image(att_map, image_tensor):
#     """
#     att_map: (1, H, W) torch tensor
#     image_tensor: (3, H, W) torch tensor, values in [0, 1]
#     """
#     # Convert and resize
#     att_map = att_map.squeeze(0).cpu().numpy()  # (H, W)
#     att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-6)

#     heatmap = cv2.applyColorMap(np.uint8(255 * att_map), cv2.COLORMAP_JET)
#     heatmap = heatmap[..., ::-1] / 255.0  # BGR to RGB

#     # Convert image tensor to numpy
#     img_np = image_tensor.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
#     img_np = np.clip(img_np, 0, 1)

#     # Overlay
#     overlay = 0.6 * img_np + 0.4 * heatmap
#     overlay = np.clip(overlay, 0, 1)

#     plt.imshow(overlay)
#     plt.axis('off')
#     plt.title("Attention Overlay")
#     plt.show()


# Load your model and processor (you can change this to your own)
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

st.title("🖼️ Retinal Image Captioning Demo")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
keywords = st.text_input("Enter keywords (comma-separated):", "")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

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

            out = model.generate_v2(image,keyword_tokens)
            out = out[0]
    


    st.markdown(f"<h4>{out}</h4>", unsafe_allow_html=True)

