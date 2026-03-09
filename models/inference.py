import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from models.grad_cam import make_gradcam_heatmap, save_and_display_gradcam
import cv2

# Define globals for loaded model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'resnet_pneumonia_model.h5')
_model = None

def get_model():
    """Lazy load the Keras model"""
    global _model
    if _model is None:
        try:
            if os.path.exists(MODEL_PATH):
                _model = tf.keras.models.load_model(MODEL_PATH)
                print(f"Loaded production model from {MODEL_PATH}")
            else:
                raise FileNotFoundError("No 'resnet_pneumonia_model.h5' found in the models directory. Please run the training script.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    return _model

def preprocess_image(img_path, target_size=(224, 224)):
    """Preprocess image for ResNet50"""
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    # Expand dimensions for batch size of 1
    img_array = np.expand_dims(img_array, axis=0)
    # Re-scale matching the ImageDataGenerator rescale=1./255
    img_array = img_array / 255.0
    return img_array

def find_last_conv_layer(model):
    """Dynamically find the last convolutional layer in the model for Grad-CAM."""
    # Specifically target ResNet50's final conv block output for accurate heatmaps
    try:
        return model.get_layer("conv5_block3_out").name
    except ValueError:
        pass
        
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4: # Typically conv layers output (batch, height, width, channels)
            return layer.name
    raise ValueError("Could not find a convolutional layer in the model.")

def process_and_predict(img_path, heatmap_save_path):
    """
    Complete pipeline: Preprocess -> Predict -> GradCAM -> Save Heatmap -> Segment Lung
    Returns: prediction_label, confidence_score, severity, left_affected, right_affected
    """
    model = get_model()
    
    # 1. Preprocess
    img_array = preprocess_image(img_path)
    
    # 2. Predict
    # Assuming output is sigmoid binary classification (0=Normal, 1=Pneumonia)
    # Adjust indexing if using softmax categorical
    preds = model.predict(img_array)
    
    # Check shape to determine if it's binary or categorical
    if preds.shape[-1] == 1:
        # Binary Classification
        prob_pneumonia = float(preds[0][0])
        is_pneumonia = prob_pneumonia > 0.5
        confidence = prob_pneumonia * 100 if is_pneumonia else (1 - prob_pneumonia) * 100
        pred_label = "Pneumonia" if is_pneumonia else "Normal"
        heatmap_pred_index = 0
    else:
        # Categorical (assumes 0=Normal, 1=Pneumonia)
        class_idx = np.argmax(preds[0])
        confidence = float(preds[0][class_idx]) * 100
        pred_label = "Pneumonia" if class_idx == 1 else "Normal"
        heatmap_pred_index = class_idx

    try:
        # 3. Grad-CAM
        last_conv_layer_name = find_last_conv_layer(model)
        
        # We drop the preprocess_input because Grad-CAM takes the raw inputs expected by the model
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=heatmap_pred_index)
        
        # 4. Save Heatmap Overlay
        save_and_display_gradcam(img_path, heatmap, cam_path=heatmap_save_path)
        
        # 5. Lung Region Segmentation & Area Calculation
        # Resize heatmap to standard 224x224
        resized_heatmap = cv2.resize(heatmap, (224, 224))
        
        # Threshold to isolate highly active regions (opacities)
        active_regions = (resized_heatmap > 0.4).astype(float)
        
        # Split into left and right halves 
        # (In PA X-rays, the patient's Right lung is on the left side of the image)
        mid = 112
        patient_right_lung = active_regions[:, :mid]
        patient_left_lung = active_regions[:, mid:]
        
        # Calculate percentage (normalized roughly to standard lung bounding box)
        # Using a bounding box factor of ~50% since the lung doesn't occupy the entire half
        right_affected = min((np.sum(patient_right_lung) / patient_right_lung.size) * 100 * 2.0, 100.0)
        left_affected = min((np.sum(patient_left_lung) / patient_left_lung.size) * 100 * 2.0, 100.0)
        
    except Exception as e:
        print(f"Warning: Grad-CAM or Segmentation failed: {e}")
        # If Grad-CAM fails, just copy the original image
        import shutil
        shutil.copy(img_path, heatmap_save_path)
        left_affected = 0.0
        right_affected = 0.0

    # Severity Scoring
    if pred_label == 'Normal':
        severity = 'None'
        left_affected = 0.0
        right_affected = 0.0
    else:
        avg_affected = (left_affected + right_affected) / 2
        if avg_affected <= 30:
            severity = 'Mild'
        elif avg_affected <= 70:
            severity = 'Moderate'
        else:
            severity = 'Severe'

    return pred_label, confidence, severity, left_affected, right_affected
