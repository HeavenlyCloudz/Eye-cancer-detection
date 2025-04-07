# OPTI AI - Eye Disease Prediction

## Overview

OPTI AI is a Streamlit application that utilizes a convolutional neural network (CNN) model to predict the presence of various eye conditions, including eye cancer and ten different non-cancerous eye diseases. The application aims to provide an intuitive interface for users to upload images of eyes and receive predictions about potential health issues.

## Features

- **Model Predictions**: Predicts the presence of:
  - **Eye Cancer**
  - **Non-Cancerous Eye Diseases**:
    - Glaucoma
    - Cataract
    - Myopia
    - Background Diabetic Retinopathy
    - Central Retinal Vein Occlusion
    - Optic Atrophy
    - Disc Swelling and Abnormality
    - Preretinal Hemorrhage
    - Hypertensive Retinopathy
    - Age Related Macular Degeneration
  - **Normal Eyes**

- **User-Friendly Interface**: Allows users to easily upload images and receive predictions.

- **Grad-CAM Visualization**: Provides visual explanations of the model's predictions through Grad-CAM overlays, highlighting the areas of the image that contribute most to the decision.

## Installation

To run the Streamlit app locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-directory>
