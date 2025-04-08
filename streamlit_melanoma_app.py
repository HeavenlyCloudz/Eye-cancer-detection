import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import cv2
import os
import zipfile
import tempfile

# Constants
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
BATCH_SIZE = 32
MODEL_FILENAME = 'eye_cancer_model.keras'

# Class labels
CLASS_LABELS = [
    "Normal",  # 0
    "Eye Cancer",  # 1
    "Glaucoma",  # 2
    "Cataract",  # 3
    "Myopia",  # 4
    "Background Diabetic Retinopathy",  # 5
    "Central Retinal Vein Occlusion",  # 6
    "Optic Atrophy",  # 7
    "Disc Swelling and Abnormality",  # 8
    "Preretinal Hemorrhage",  # 9
    "Hypertensive Retinopathy",  # 10
    "Age Related Macular Degeneration"  # 11
]

# Load data function
def load_data(train_dir, val_dir, test_dir):
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20,
                                       width_shift_range=0.2, height_shift_range=0.2,
                                       shear_range=0.2, zoom_range=0.2,
                                       horizontal_flip=True, fill_mode='nearest')
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    try:
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
        return train_generator, val_generator, test_generator
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

# Build model function
def build_model(num_classes):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=x)
    
    for layer in base_model.layers:
        layer.trainable = False  # Freeze base layers
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Grad-CAM function
def grad_cam(model, img_array, class_idx):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.output, model.layers[-2].output]
    )
    with tf.GradientTape() as tape:
        model_output, conv_layer_output = grad_model(np.array([img_array]))
        loss = model_output[:, class_idx]
    
    grads = tape.gradient(loss, conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_layer_output = conv_layer_output[0]
    heatmap = conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    
    return heatmap.numpy()

# Streamlit app layout
st.title("OPTI AI - Eye Disease Prediction")

# Upload a zip file containing the dataset
uploaded_zip = st.file_uploader("Upload Dataset Zip File", type=["zip"])

if uploaded_zip is not None:
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Save the uploaded zip file
        zip_path = os.path.join(tmpdirname, uploaded_zip.name)
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.getbuffer())
        
        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdirname)
        
        # Set directories for training, validation, and testing
        train_dir = os.path.join(tmpdirname, "train")
        val_dir = os.path.join(tmpdirname, "val")
        test_dir = os.path.join(tmpdirname, "test")

        if st.button("Load Data and Train"):
            train_generator, val_generator, test_generator = load_data(train_dir, val_dir, test_dir)
            
            if train_generator and val_generator:
                num_classes = len(CLASS_LABELS)  # Set number of classes based on defined labels
                
                # Check if the model already exists
                if os.path.exists(MODEL_FILENAME):
                    model = tf.keras.models.load_model(MODEL_FILENAME)
                    st.success("Model loaded successfully!")
                else:
                    model = build_model(num_classes=num_classes)
                    model.fit(train_generator, validation_data=val_generator, epochs=5)
                    model.save(MODEL_FILENAME)
                    st.success("Model trained and saved successfully!")

                # Evaluate on test data
                test_loss, test_accuracy = model.evaluate(test_generator)
                st.write(f"Test Accuracy: {test_accuracy:.2f}")
                st.write(f"Test Loss: {test_loss:.2f}")

# Image upload and prediction
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
    img_array = tf.keras.preprocessing.image.img_to_array(image) / 255.0
    st.image(image, caption="Uploaded Image", use_column_width=True)

    predictions = model.predict(np.expand_dims(img_array, axis=0))
    predicted_class_idx = np.argmax(predictions)
    predicted_class = CLASS_LABELS[predicted_class_idx]
    st.write(f"Predicted Class: {predicted_class}")

    # Grad-CAM overlay
    heatmap = grad_cam(model, img_array, predicted_class_idx)
    heatmap = cv2.resize(heatmap, (IMAGE_WIDTH, IMAGE_HEIGHT))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img_array * 255
    st.image(superimposed_img.astype(np.uint8), caption="Grad-CAM Overlay", use_column_width=True)

# Download model button
if os.path.exists(MODEL_FILENAME):
    with open(MODEL_FILENAME, "rb") as f:
        st.download_button(label="Download Trained Model", data=f, file_name=MODEL_FILENAME, mime="application/octet-stream")

# Function to collect feedback
def collect_feedback():
    st.title(":rainbow[Feedback] Form")
    
    # Add a text area for the feedback
    feedback = st.text_area("Please share your feedback to improve this app💕", "", height=150)
    
    # Add a submit button
    if st.button("Submit Feedback"):
        if feedback:
            st.success("Thank you for your feedback🫶!")
            # Save feedback to a file, database, or send it via email, etc.
            save_feedback(feedback)
        else:
            st.error("Please enter some feedback before submitting😡.")

# Function to save feedback (can be customized to store feedback anywhere)
def save_feedback(feedback):
    # Example: Save to a text file (or database)
    with open("user_feedback.txt", "a") as f:
        f.write(f"Feedback: {feedback}\n{'-'*50}\n")
    st.info("Your feedback has been recorded.")

# Show the feedback form
collect_feedback()
