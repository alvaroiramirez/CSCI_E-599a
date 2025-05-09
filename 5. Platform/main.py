# Imports and util files
import streamlit as st
import pandas as pd
import numpy as np
from utils.model_utils import load_model_check

# First action for displaying the pages
st.set_page_config(page_title="FEMA Disaster Image Damage Detection", layout="wide", initial_sidebar_state="expanded")


st.title("Main Page")
st.sidebar.success("Select a page to view the demonstration.")

st.write("This app demonstrates a pipeline of object detection, image cropping, and captioning on disaster images taken from an aerial view. The outputs include single photo inference outputs and batch inference outputs of JSON annotations with the bounding box data.")

st.sidebar.write("*If you upload and run inference on a single image, you can keep track of that image below. This is also what is currently on the Crop page.*")


# Sidebar will have the uploaded image
st.markdown("""
<style>
.sidebar-image-container img {
    border: 2px solid #aaa;
    border-radius: 6px;
    padding: 4px;
    background-color: #f8f8f8;
}
</style>
""", unsafe_allow_html=True)

if "last_uploaded_image" in st.session_state:
    st.sidebar.image(
        st.session_state["last_uploaded_image"],
        caption=None,
        use_container_width=True
    )
    #display its filename
if "last_filename" in st.session_state:
    st.sidebar.markdown(
        f"**Filename:** `{st.session_state['last_filename']}`"
    )

# Link to Google Drive with YOLO Model
st.markdown("""
### Download Model
[Download the model here](https://drive.google.com/file/d/1YK_ykzgCNtuUHJayYXkhaLDOs3Mdmvr0/view?usp=drive_link) and upload it below.
""", unsafe_allow_html=True)

st.write("To run this app, you will need to upload a YOLO model to run inference with. The one above is a fine-tuned YOLOv11 model but you may provide your own.")

# Upload model
uploaded_model = st.file_uploader("Upload your model (.pt, .pth, etc)", type=["pt", "pth"])
if uploaded_model is not None:
    with st.spinner("Loading uploaded model... please wait."):
        
        model, device = load_model_check(uploaded_model)
        st.session_state["model"] = model
        st.session_state["device"] = device
    st.success("**YOLO model loaded and ready for use.**")

st.write(
    """
    Below this, you will see a number of hyperlinks to direct you to the previous bodies of research or groups that provided data that inspired this project.
    
    **Example Research and Sources (final product may not reflect this):**
    - [Original LADIv2 Paper Done By MIT on arXiv](https://arxiv.org/abs/2406.02780)
    - [LADIv2 Repository on HuggingFace](https://huggingface.co/datasets/MITLL/LADI-v2-dataset)
        
    """)



