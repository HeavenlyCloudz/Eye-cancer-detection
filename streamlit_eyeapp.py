import streamlit as st
import snowflake.connector
import os
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.cm as cm 

# Constants
IMAGE_HEIGHT, IMAGE_WIDTH = 224, 224
MODEL_FILE = 'eye_cancer_detection_model.keras'
BATCH_SIZE = 32
EPOCHS = 10
base_data_dir = os.path.join(os.getcwd(), 'data')
train_data_dir = os.path.join(base_data_dir, 'train')
val_data_dir = os.path.join(base_data_dir, 'val')
test_data_dir = os.path.join(base_data_dir, 'test')

last_conv_layer_name = 'top_conv'
is_new_model = False

# Function to calculate class weights
def calculate_class_weights(train_generator):
    labels = train_generator.classes
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
    return class_weights_dict

def compute_class_weights(train_generator):
    class_weights = calculate_class_weights(train_generator)
    total_class_weight = sum(class_weights.values())
    imbalance_ratio = max(class_weights.values()) / total_class_weight
    return class_weights, imbalance_ratio

# Focal Loss Function
def focal_loss(alpha=0.25, gamma=2.0):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        loss = alpha * tf.pow(1 - y_pred, gamma) * cross_entropy
        return tf.reduce_mean(loss)
    return focal_loss_fixed

def create_efficientnet_model(input_shape=(224, 224, 3), learning_rate=1e-3):
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers[-50:]:
        layer.trainable = True

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    predictions = layers.Dense(3, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def load_model_file():
    global is_new_model
    if os.path.exists(MODEL_FILE):
        try:
            model = load_model(MODEL_FILE, custom_objects={"focal_loss": focal_loss})
            for layer in model.layers[-50:]:
                layer.trainable = True
            st.success("Model loaded successfully!")
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    else:
        st.warning("No saved model found. Creating a new model.")
        is_new_model = True
        return create_efficientnet_model()

model = load_model_file()

@tf.function(input_signature=[tf.TensorSpec(shape=[None, 224, 224, 3], dtype=tf.float32)])
def predict(input_tensor):
    return model(input_tensor)

def preprocess_image(img_path):
    try:
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((224, 224))
        img_array = np.asarray(img, dtype=np.float32)
        if img_array.shape != (224, 224, 3):
            raise ValueError(f"Unexpected shape after resizing: {img_array.shape}")
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

def main():
    image_tensor = tf.random.uniform((1, 224, 224, 3))
    result = predict(image_tensor)
    print(result)

def load_data(train_dir, val_dir, batch_size):
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20,
                                       width_shift_range=0.2, height_shift_range=0.2,
                                       shear_range=0.2, zoom_range=0.2,
                                       horizontal_flip=True, fill_mode='nearest')
    val_datagen = ImageDataGenerator(rescale=1./255)
    try:
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
            batch_size=batch_size,
            class_mode='categorical'
        )
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
            batch_size=batch_size,
            class_mode='categorical'
        )
        return train_generator, val_generator
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

def print_layer_names():
    try:
        base_model = EfficientNetB0(include_top=False, weights='', input_shape=(224, 224, 3))
        return [layer.name for layer in base_model.layers]
    except Exception as e:
        st.error(f"Error in print_layer_names: {str(e)}")
        return []

def plot_training_history(history):
    try:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
         # Plot accuracy with yellow and purple
        ax[0].plot(history.history['accuracy'], label='Train Accuracy', color='yellow')
        ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy', color='purple')
        ax[0].set_title('Model Accuracy')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Accuracy')
        ax[0].legend()

        # Plot loss with yellow and purple
        ax[1].plot(history.history['loss'], label='Train Loss', color='yellow')
        ax[1].plot(history.history['val_loss'], label='Validation Loss', color='purple')
        ax[1].set_title('Model Loss')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Loss')
        ax[1].legend()

        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting training history: {str(e)}")
        
def train(train_dir, val_dir):
    global model
    train_generator, val_generator = load_data(train_dir, val_dir, BATCH_SIZE)
    if not train_generator or not val_generator:
        st.error("Failed to load training or validation data.")
        return

    class_weights = calculate_class_weights(train_generator)
    imbalance_ratio = max(class_weights.values()) / sum(class_weights.values())
    loss_function = focal_loss(alpha=0.25, gamma=2.0) if imbalance_ratio > 1.5 else 'categorical_crossentropy'

    if not model._is_compiled:
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        class_weight=class_weights
    )

    model.save(MODEL_FILE)
    st.success("Model saved successfully!")
    st.write("Training completed.")

def test_model(model):
    test_datagen = ImageDataGenerator(rescale=1./255)
    try:
        test_generator = test_datagen.flow_from_directory(
            test_data_dir,
            target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )

        test_loss, test_accuracy = model.evaluate(test_generator)
        st.sidebar.write(f"Test Loss: {test_loss:.4f}")
        st.sidebar.write(f"Test Accuracy: {test_accuracy:.4f}")

        y_pred = model.predict(test_generator)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = test_generator.classes
        cmatrix = confusion_matrix(y_true, y_pred_classes)

        class_labels = list(test_generator.class_indices.keys())

        st.sidebar.write("Class-wise Metrics:")
        for i, label in enumerate(class_labels):
            tp = cmatrix[i, i]
            fp = sum(cmatrix[:, i]) - tp
            fn = sum(cmatrix[i, :]) - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            st.sidebar.write(f"**{label}** - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cmatrix, annot=True, fmt='d', cmap='Yellows',
                    xticklabels=class_labels, yticklabels=class_labels, ax=ax)
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error during testing: {str(e)}")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    try:
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)

            if pred_index is None:
                pred_index = tf.argmax(preds[0])

            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, last_conv_layer_output), axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        heatmap /= tf.reduce_max(heatmap) if tf.reduce_max(heatmap) > 0 else 1

        # Resize the heatmap to match the original image size
        heatmap = cv2.resize(heatmap.numpy(), (IMAGE_WIDTH, IMAGE_HEIGHT))
        return heatmap

    except Exception as e:
        st.error(f"Error generating Grad-CAM heatmap: {str(e)}")
        return None

def display_gradcam(img, heatmap, alpha=0.4):
    try:
        # Ensure img is in the correct format (BGR to RGB if needed)
        if img.shape[2] == 3:  # Check if the image has 3 channels
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img

        heatmap = np.uint8(255 * heatmap)

        # Use the updated method to get the colormap
        jet = plt.colormaps['jet']
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        jet_heatmap = np.uint8(jet_heatmap * 255)
        jet_heatmap = cv2.cvtColor(jet_heatmap, cv2.COLOR_RGB2BGR)

        # Resize the heatmap to the original image size
        jet_heatmap = cv2.resize(jet_heatmap, (img_rgb.shape[1], img_rgb.shape[0]))

        superimposed_img = cv2.addWeighted(jet_heatmap, alpha, img_rgb, 1 - alpha, 0)
        return superimposed_img

    except Exception as e:
        st.error(f"Error displaying Grad-CAM: {str(e)}")
        return None

# Streamlit UI
st.title("Eye Cancer Detection👀")
st.markdown(
    """
    <style>
    body {
        background-color: #CBC3E3; /* Light purple color */
    }
    .section {
        background-image: url('https://www.cancer.org/adobe/dynamicmedia/deliver/dm-aid--9f4bfdfd-41b9-4175-a40b-804fda3661c2/eye.jpg?preferwebp=true&quality=82');
        background-size: cover; 
        background-repeat: no-repeat;
        background-position: center;
        padding: 60px; 
        border-radius: 10px;
        color: purple; 
        margin: 20px 0;
        height: 400px; 
    }
    .sidebar .sidebar-content {
        background-color: #ADD8E6; 
        color: yellow; 
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="section">', unsafe_allow_html=True)
st.header("Thank you for using OPTI👁️")
st.write("CNNs are the preferred network for detecting eye cancer due to their ability to process image data. They can perform tasks such as classification, segmentation, and object recognition. In the case of eye cancer detection, CNNs have shown promise in surpassing traditional methods and offering a more efficient and accurate approach to early diagnosis.")
st.markdown('</div>', unsafe_allow_html=True)
st.markdown("Visit [OPTI](https://readymag.website/u4174625345/5256774/) for more information.")
st.markdown("Visit my [GitHub](https://github.com/HeavenlyCloudz/Eye-cancer-detection.git) repository for insight on my code.")

# Sidebar controls
st.sidebar.title("Controls🎮")

# Load the model
model = load_model_file()

# Hyperparameter inputs
epochs = st.sidebar.number_input("Number of epochs for training", min_value=1, max_value=100, value=10)
batch_size = st.sidebar.number_input("Batch size", min_value=1, max_value=64, value=BATCH_SIZE)
 
 # Add input for number of evaluations during testing
eval_epochs = st.sidebar.number_input("Number of evaluations for testing", min_value=1, max_value=10, value=1)

# Button to train model
if st.sidebar.button("Train Model"):
    if model is not None:
        with st.spinner("Training the model🤖..."):
            train_generator, val_generator = load_data(train_data_dir, val_data_dir, BATCH_SIZE)

            # Ensure generators are loaded
            if train_generator is not None and val_generator is not None:
                y_train = train_generator.classes
                class_labels = np.unique(y_train)
                weights = compute_class_weight('balanced', classes=class_labels, y=y_train)
                class_weights = {i: weights[i] for i in range(len(class_labels))}

                # Calculate imbalance ratio
                class_0_count = np.sum(y_train == 0)
                class_1_count = np.sum(y_train == 1)
                imbalance_ratio = (
                    max(class_0_count, class_1_count) / min(class_0_count, class_1_count)
                    if min(class_0_count, class_1_count) > 0
                    else 1
                )

                # Determine loss function based on imbalance
                if imbalance_ratio > 1.5:
                    loss_function = focal_loss(alpha=0.25, gamma=2.0)
                    st.sidebar.write(f"Detected significant class imbalance (ratio: {imbalance_ratio:.2f}). Using Focal Loss.")
                else:
                    loss_function = 'categorical_crossentropy'
                    st.sidebar.write(f"Class balance is acceptable (ratio: {imbalance_ratio:.2f}). Using Categorical Cross-Entropy.")

                # Add ReduceLROnPlateau Callback
                reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

                # Compile the model with the selected loss function
                optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
                model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

                # Train the model
                history = model.fit(
                    train_generator,
                    validation_data=val_generator,
                    epochs=epochs,
                    class_weight=class_weights,
                    callbacks=[reduce_lr]
                )

                # Save model
                model.save(MODEL_FILE)
                st.success("Model trained and saved successfully!")
                plot_training_history(history)

                # Add a download button for the model file after training completes
                with open(MODEL_FILE, "rb") as f:
                    model_data = f.read()
                st.download_button(
                    label="Download Trained Model",
                    data=model_data,
                    file_name=MODEL_FILE,
                    mime="application/octet-stream"
                )
    else:
        st.error("Model is not available for training. Please check model initialization.")

# Button to test model
if st.sidebar.button("Test Model"):
    if model:
        with st.spinner("Testing the model📝..."):
            for _ in range(eval_epochs):  # Repeat testing as per user input
                test_model(model)
    else:
        st.warning("No model found. Please train the model first.")

# Function to process and predict image
def process_and_predict(image_path, model, last_conv_layer_name):
    try:
        # Preprocess the image
        processed_image = preprocess_image(image_path)

        if processed_image is not None and model:
            # Make prediction
            predictions = model.predict(processed_image)[0]  # For multi-class, get the full prediction
            predicted_class = np.argmax(predictions)  # Get the class with the highest probability
            confidence = predictions[predicted_class]  # Confidence of the predicted class
            confidence_percentage = confidence * 100  # Convert to percentage

            # Class labels for the three categories
            class_labels = ["Eye Cancer", "Non-Cancerous Eye Disease", "Normal Eyes"]
            result = class_labels[predicted_class]

            # Display Prediction Result
            st.subheader("Prediction Result:")
            st.write(f"**{result}**")
            st.write(f"**Confidence: {confidence_percentage:.2f}%**")  # Show confidence

            # Add description based on result
            if result == 'Eye Cancer':
                st.write("**Note:** The model has determined the presence of eye cancer. Please consult with a healthcare professional for further assessment.")
            elif result == 'Non-Cancerous Eye Disease':
                st.write("**Note:** The model has determined the presence of a non-cancerous eye disease. This could be one of several benign conditions, but you should still consult a professional for further clarification.")
            elif result == 'Normal Eyes':
                st.write("**Note:** The model has determined your eyes are normal. However, regular eye check-ups are recommended for maintaining eye health.")

            # Additional symptoms check for eye diseases
            if result != 'Normal Eyes':
                symptoms = [
                    "Vision loss",
                    "Blurred vision",
                    "Eye pain",
                    "Redness or swelling",
                    "Sensitivity to light",
                    "Double vision",
                    "Seeing flashes or floaters"
                ]

                # Multi-select for symptoms
                selected_symptoms = st.multiselect("Please select any symptoms you are experiencing:", symptoms)

                # Done button
                if st.button("Done"):
                    # Check how many symptoms are selected
                    if len(selected_symptoms) > 3:
                        st.warning("Even if it isn't cancer, these symptoms could indicate other eye conditions. Please contact a healthcare provider.")
                    elif len(selected_symptoms) == 3:
                        st.warning("These symptoms could possibly point to other eye conditions. Be sure to consult a healthcare provider.")
                    elif len(selected_symptoms) > 0:
                        st.success("You have selected a manageable number of symptoms. Monitor your health and consult a healthcare provider if necessary.")
                    else:
                        st.info("No symptoms selected. If you are feeling unwell, please consult a healthcare provider.")


            # Generate Grad-CAM heatmap
            try:
                heatmap = make_gradcam_heatmap(processed_image, model, last_conv_layer_name)

                if heatmap is not None:
                    uploaded_image = Image.open(image_path)  # Open with PIL

                    # Convert PIL image to numpy array for OpenCV compatibility
                    uploaded_image_np = np.array(uploaded_image)

                    superimposed_img = display_gradcam(uploaded_image_np, heatmap)

                    # Show images
                    st.image(image_path, caption='Uploaded Image', use_container_width=True)

                    if superimposed_img is not None:
                        st.image(superimposed_img, caption='Superimposed Grad-CAM', use_container_width=True)
                    else:
                        st.warning("Grad-CAM generation failed.")

                    uploaded_image.close()  # Close the PIL image
                else:
                    st.warning("Grad-CAM generation returned None.")

            except Exception as e:
                st.error(f"Error displaying Grad-CAM: {str(e)}")

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")

    finally:
        # Ensure cleanup of the image file
        if os.path.exists(image_path):
            try:
                os.remove(image_path)
            except Exception as e:
                st.warning(f"Error removing image file: {str(e)}")

# Load Model
last_conv_layer_name = 'top_conv'

# Normal Image Upload
uploaded_file = st.sidebar.file_uploader("Upload your image (JPG, PNG)", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1]
    temp_filename = f"temp_image.{file_extension}"

    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())

    process_and_predict(temp_filename, model, last_conv_layer_name)

# Mobile Capture Option
st.sidebar.header("Take a Picture")
photo = st.sidebar.file_uploader("Capture a photo", type=["jpg", "jpeg", "png"])
if photo is not None:
    file_extension = photo.name.split('.')[-1]
    captured_filename = f"captured_image.{file_extension}"

    with open(captured_filename, "wb") as f:
        f.write(photo.getbuffer())

    process_and_predict(captured_filename, model, last_conv_layer_name)

# Clear cache button
if st.button("Clear Cache"):
    st.cache_data.clear()  # Clear the cache
    st.success("Cache cleared successfully!🎯")

if st.sidebar.button("Show Layer Names"):
    st.write("Layer names in EfficientNetB0:")
    layer_names = print_layer_names()
    st.text("\n".join(layer_names))

if __name__ == "__main__":
    main()
