# NOMA AI - Skin Cancer Prediction

## Overview

NOMA AI is a Streamlit application that utilizes a convolutional neural network (CNN) model to predict the presence of melanoma and various non-cancerous skin conditions. The application aims to provide an intuitive interface for users to upload images of skin lesions and receive predictions about potential health issues.

## Features

- **Model Predictions**: Predicts the presence of:
  - **Melanoma**
  - **Non-Cancerous Skin Conditions**:
    - Nevus
    - Seborrheic Keratosis
    - Actinic Keratosis
    - Basal Cell Carcinoma
    - Squamous Cell Carcinoma
    - Dermatofibroma
    - Psoriasis
    - Eczema
    - Normal Skin

- **User-Friendly Interface**: Allows users to easily upload images and receive predictions.

- **Grad-CAM Visualization**: Provides visual explanations of the model's predictions through Grad-CAM overlays, highlighting the areas of the image that contribute most to the decision.

## Installation

To run the Streamlit app locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-directory>
