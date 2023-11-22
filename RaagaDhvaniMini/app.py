import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import platform
import time
import pathlib
import os

import streamlit as st

## menu
st.set_page_config(page_title = "Raagadhvani",page_icon="🎶",
    layout="wide")


#title of page
st.title("Raagadhvani 🎵")
                                                                                                                                                   


#hiding menu
st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

st.sidebar.success("Menu")


# Define colors
header_color = "#9b59b6"  
info_color = "#3498db"    

# Header
st.markdown(
    f"<h1 style='color: {header_color};'>Welcome to the Carnatic Music Experience! 🎶✨🎵</h1>",
    unsafe_allow_html=True
)

# Introduction
st.markdown(
    f"<p style='color: {info_color}; font-size: 18px; text-align: center;'>"
    "Carnatic music is a classical tradition in Southern India known for its rich melody and intricate rhythm patterns."
    " Explore the beauty of Carnatic music through music generation and raga identification."
    "</p>",
    unsafe_allow_html=True
)

# Interesting Facts
st.info(
    "### Interesting Facts:"
    "\n- Carnatic music has a history spanning over 2000 years."
    "\n- Ragas are the melodic modes used in Carnatic music, each with its own unique mood and emotion."
    "\n- The mridangam and the violin are commonly used instruments in Carnatic music performances."
    "\n- There are over 300 ragas in Carnatic music, each offering a distinct musical experience."
    "\n- Some famous Carnatic music composers include Tyagaraja, Muthuswami Dikshitar, and Syama Sastri."
)

# Call to Action
st.markdown(
    "<p style='color: #2ecc71; font-size: 20px; text-align: center;'>"
    "Let's dive into the world of Carnatic music!"
    "</p>",
    unsafe_allow_html=True
)


st.title("MUSIC PLAYER")
st.write("Play any audio file you like!")
audio_file = st.file_uploader("Choose a music file (mp3 or wav)", type=["mp3", "wav"])
if audio_file:
    st.audio(audio_file, format='audio/wav')