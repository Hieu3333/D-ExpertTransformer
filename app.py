import os
import sys
# model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model'))
# sys.path.append(model_path)

# modules_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'modules'))
# sys.path.append(modules_path)

import streamlit as st
from PIL import Image
import torch
from model.model import ExpertTransformer
from modules.tokenizer import Tokenizer
from Args import Args
from modules.utils import get_inference_transform


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
keywords = st.text_input("Enter keywords:", "")

if st.button("Generate"):
    # Load and display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

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


    st.markdown(f"<h3>{out}</h3>", unsafe_allow_html=True)

