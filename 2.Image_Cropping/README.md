**SOURCE**
------

Upload the images to the `source` folder. You can obtain the images from our Roboflow account. If this is the case, select the `COCO format` and download the dataset as a zip file to your computer. Unpack the zip file and use it as your source folder.



**DETECTION**
---------

**Description**

This program reads the images in `source` and extracts the bounding boxes in separate image files. It also creates a copy of the original image with the bounding boxes drawn on it.


**Preparation**

Store the images and their JSON files (bounding boxes coordinates) in the `source` folder. You can use a different folder name for the source data. In that case, update the folder name in `source_base_dir` in the source code.


**Results**

The output will be stored in the `processed` folder located in the same folder where the JSON file is. The program processes all subfolders in `source`, if any.
