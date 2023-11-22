# RaagaDhvani1.0

# Raagadhvani Project README

## Overview

Raagadhvani is an innovative project that utilizes audio processing techniques and machine learning algorithms to automatically identify and compose music in traditional Indian classical Raagas. This project caters to musicians and music enthusiasts, providing an AI-driven solution for recognizing Raagas in input audio files and generating subsequent compositions based on the identified Raaga.

## Features

### Raaga Identification:
- The project employs deep learning algorithms to classify the Raaga of input musical compositions.
- Functions as an AI Music Tutor to evaluate the closeness of a student's input music to the reference Raaga.

### Composition Generation:
- Utilizes recurrent neural network-based LSTM (Long Short-Term Memory) algorithms to recognize and encode long-term patterns in the input Music.
- Generates new sequences of notes in a similar compositional style to the input Raaga.

### Web-Based Interface:
- A user-friendly web interface allows users to input Carnatic music and receive the new sequence of notes from the same Raaga.

## Architecture

The proposed model consists of two main steps:

1. **Raaga Classification:**
   - Identifies the Raaga from the input musical composition.
   - Doubles as an AI Music Tutor Application, enabling users to calibrate elements of their input file with built-in references.

2. **Composition Generation:**
   - Generates subsequent compositions of notes and motifs based on the identified Raaga for the required duration.

## Getting Started

To use Raagadhvani, follow these steps:


### Usage:### Installation:
1. Clone the repository to your local machine.
2. Install the required dependencies using the provided `requirements.txt` file.

1. Run the Raaga Classifier to identify the Raaga of your input music sample.
2. Use the identified Raaga to generate a new sequence of notes and motifs using the Composition Generator.

### Web Interface:
- Access the web-based interface by navigating to the provided URL.
- Input your Carnatic music and receive the newly generated composition.


Installation Steps: 
-------------------------------------------------------------------------------------------------------
Clone the Repository

Install the requirements.txt 
```
pip install -r requirements.txt 
```

Run the python file to execute 
```
python -m streamlit run app.py
```



