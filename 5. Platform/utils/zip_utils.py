import os

# when making an annotation, clean it of the filetype (e.g. not img1.jpg_annotations but img1_annotations)
def clean_annotation(filename):
    return os.path.splitext(filename)[0]

