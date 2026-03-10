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

def apply_lung_segmentation(img_array):
    """
    Extracts the lung region to remove ribs, background, and markers.
    Uses OpenCV to create a bounding mask over the central chest cavity.
    """
    # Convert back to 0-255 uint8 format for OpenCV
    img_uint8 = (img_array[0] * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    
    # Threshold to find the main body area (remove pure black background and noise)
    _, binary = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    
    # Morphological operations to clean up mask
    kernel = np.ones((5,5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.zeros_like(gray)
    if contours:
        # Keep the largest contour (the chest cavity / lungs)
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    else:
        # Fallback if contour fails
        mask.fill(255)
        
    # Smooth the mask edges slightly
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    
    # Convert mask to 3 channels and normalize back to 0-1
    mask_3d = cv2.merge([mask, mask, mask]).astype(np.float32) / 255.0
    
    # Apply the mask to original image, forcing background to pure black
    segmented_img = img_array[0] * mask_3d
    return np.expand_dims(segmented_img, axis=0)

def predict_with_tta(model, img_array):
    """
    Test Time Augmentation (TTA).
    Generates variations (original, flipped, zoomed, rotated) and averages the predictions.
    This provides more stable and accurate predictions.
    """
    h, w = img_array[0].shape[:2]
    
    # 1. Original
    img_orig = img_array[0]
    
    # 2. Horizontal Flip
    img_flip = cv2.flip(img_orig, 1)
    
    # 3. Zoom (Crop center 80% and resize)
    crop_h, crop_w = int(h * 0.1), int(w * 0.1)
    img_zoom = img_orig[crop_h:h-crop_h, crop_w:w-crop_w]
    img_zoom = cv2.resize(img_zoom, (w, h))
    
    # 4. Rotate 5 degrees
    M = cv2.getRotationMatrix2D((w//2, h//2), 5, 1.0)
    img_rot = cv2.warpAffine(img_orig, M, (w, h))
    
    # Stack into a batch
    batch = np.array([img_orig, img_flip, img_zoom, img_rot])
    
    # Predict all 4 augmentations simultaneously
    preds = model.predict(batch)
    
    # Average the confidence scores
    avg_pred = np.mean(preds, axis=0, keepdims=True)
    return avg_pred

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
    
    # 1. Preprocess & Segment
    img_array = preprocess_image(img_path)
    segmented_array = apply_lung_segmentation(img_array)
    
    # 2. Predict with Test Time Augmentation (TTA)
    # Assuming output is sigmoid binary classification (0=Normal, 1=Pneumonia)
    # Adjust indexing if using softmax categorical
    preds = predict_with_tta(model, segmented_array)
    
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
        # Use the segmented_array so Grad-CAM only highlights the lungs
        heatmap = make_gradcam_heatmap(segmented_array, model, last_conv_layer_name, pred_index=heatmap_pred_index)
        
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
