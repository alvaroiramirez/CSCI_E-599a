# streamlit-fema
Streamlit content for the final FEMA project work.


This repository is for utilizing the Streamlit multimodal integration tool designed to assist FEMA with damage detection, annotation, localization, and captioning for images obtained from aerially sourced images.

As part of the deliverable for our sponsor, we have delivered an integrated multimodal platform built with Streamlit that streamlines object detection, image annotation, and image captioning for disaster damage assessment. 

The tool supports both single-image and batch processing workflows, automatically generating COCO-style annotations and enabling users to selectively download cropped regions and annotated datasets. 

Each detection is visually represented with bounding boxes and category labels. A demo for the captioning module is included to teach how to use the OpenAI GPT-4o MLLM to describe detected damages. 

The platform includes an interactive interface with guidance, an embedded video tutorial for using the OpenAI GPT-4o captioning model, and exportable outputs, making it accessible for all end users.

The single image object detection tool is accessible through the "Single Upload Object Detection" and "Cropping" pages. 

The batch object detction tool is accessible through the "Batch Object Detection" page.

The captioning demonstration is accessible through the "Captioning (Video Demonstration)" page.

Please note that this implementation may be optimally run on Google Colab with a T4 GPU for inference though it was tested and developed and debugged all with only CPU processing accessible.  
