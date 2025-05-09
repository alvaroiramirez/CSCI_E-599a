# -*- coding: utf-8 -*-
"""Streamlit Colab

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1RVfsYLt7mwZVr-N6MwM3ms3XeQuPmDNu

# Capstone: Streamlit Multimodal Tool Access Code

Note that this is meant to access the code while on Google Colab. Upload it to Colab and do these lines of code.

It is using the github version which is most recent.

# Github Access
"""

!git clone "https://github.com/yuvrajpuri/streamlit-fema.git"

"""# Github Pulls for Updates"""

# Commented out IPython magic to ensure Python compatibility.
# %cd "/content/streamlit-fema/Final Project Streamlit Demo"
!git pull origin main

"""# Mandatory Installations"""

!pip install -r requirements.txt

"""# Grab IP for using to run"""

!wget -q -O - ipv4.icanhazip.com

"""# Streamlit Runner

You use your IP in the localtunnel to run the website (as a temporary thing) as it has yet to be deployed to Streamlit Cloud.
"""

!cd "/content/streamlit-fema/Final Project Streamlit Demo" && streamlit run streamlit_demo.py & npx localtunnel --port 8501