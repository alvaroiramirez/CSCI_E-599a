import streamlit as st
from PIL import Image, ImageDraw
import os
import json
import zipfile
from io import BytesIO

# utils imports
from utils.image_utils import crop_bbox, draw_bounding_boxes
from utils.zip_utils import clean_annotation

# Labels for the damage
CATEGORY_LABELS = {
    1: "Affected building",
    2: "Major damage"
}


# Save the image that has the bounding boxes
def ds_bbox_image(image, annotations):
    copy = image.copy()
    art = ImageDraw.Draw(copy)
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        color = "purple" if ann["category_id"] == 1 else "yellow"
        art.rectangle([x, y, x + w, y + h], outline = color, width=3)
    buffer = BytesIO()
    copy.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer
    

# Start of page
st.title("Cropping Tool")

# Checking for the uploaded image from the ob2page.py
if "last_uploaded_image" in st.session_state and "last_detections" in st.session_state and "last_annotations" in st.session_state:
    image = st.session_state["last_uploaded_image"]
    detections = st.session_state["last_detections"]
    coco_data = st.session_state["last_annotations"]
    imgname = st.session_state["last_filename"]

    # Show full image with bounding boxes
    image_with_boxes = draw_bounding_boxes(image.copy(), detections)
    st.image(image_with_boxes, caption="Detected Objects", use_container_width=True)

    # The cropped images - show and make them selectable for download
    st.subheader("Cropped Regions")
    
    # Select All Toggle for the cropped images
    select_all = st.checkbox("Select All Crops", value = False)

    # Put the selectable images in an expander to keep things easy to look at
    with st.expander("View selectable crops for download", expanded=False):

        # List containing the cropped images that we selectively include
        chosen_crop_ids = []

        # Saving the cropped images to the session state
        # Refresh crops when the image changes
        if (
            "last_filename" not in st.session_state
            or st.session_state.get("cropped_images_filename") != st.session_state["last_filename"]
        ):
            # Regenerate crops for the new image
            st.session_state["cropped_images"] = [
                crop_bbox(image, det["bbox"]) for det in detections
            ]
            st.session_state["cropped_images_filename"] = st.session_state["last_filename"]        
    
        for i, (det, cropped) in enumerate(zip(detections, st.session_state["cropped_images"])):
        
            col1, col2 = st.columns([1,3])
        
            # Column 1 - the checkboxes
            with col1:
                include = st.checkbox(
                    f"Include Crop {i+1}",
                    key=f"include_{i}",
                    value=select_all
                )
                if include:
                    chosen_crop_ids.append(i)

            # Cropped images themselves
            with col2:
                #cropped = crop_bbox(image, det["bbox"])
                label = CATEGORY_LABELS.get(det["category_id"], "Unknown")
                st.image(cropped, caption=f"Crop {i+1}: {label}", width=200)

    if chosen_crop_ids:
        zip_buffer = BytesIO()

        # To the annotations, add whether or not they were chosen to be downloaded
        copied_annotations = []
        for i, anno in enumerate(coco_data["annotations"]):
            anno2 = anno.copy()
            anno2["selected"] = i in chosen_crop_ids
            copied_annotations.append(anno2)

        new_coco_export = {
            "images": coco_data["images"],
            "annotations": copied_annotations,
            "categories": coco_data["categories"]
        }

        # The zipfile generated
        clean_filename = clean_annotation(imgname)
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            
            # Save the new COCO JSON
            coco_json_str = json.dumps(new_coco_export, indent=2)
            zip_file.writestr(f"{clean_filename}_annotations.json", coco_json_str)

            # Save the picture of the image with the bounding boxes
            saved_bbox_image = ds_bbox_image(image, detections)
            zip_file.writestr(f"{clean_filename}_bboxes.jpg", saved_bbox_image.read())

            # Save the crops that we chose
            for i in chosen_crop_ids:
                crop = st.session_state["cropped_images"][i]
                cropimg_bytes = BytesIO()
                crop.save(cropimg_bytes, format="JPEG")
                cropimg_bytes.seek(0)

                zip_file.writestr(f"{clean_filename}_crop_{i}.jpg", cropimg_bytes.read())
                
        # Download ZIP file
        st.download_button(
            label="Download ZIP",
            data=zip_buffer.getvalue(),
            file_name=f"{clean_filename}_bbox_bundle.zip",
            mime="application/zip"
        )
    else:
        st.info("Choose at least one image to download.")

else:
    st.warning("No image or detections found. Please run object detection first.")
