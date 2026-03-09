import os
import tensorflow as tf
from tensorflow.keras import layers, models

def create_dummy_model():
    """
    Creates a tiny, fast-training dummy model with fake weights 
    capable of running Grad-CAM to test the web application pipeline
    without waiting for a 5-hour ResNet training job.
    """
    print("Generating Dummy Model for Web UI Testing...")
    
    # Input matching ResNet50 target size
    inputs = tf.keras.Input(shape=(224, 224, 3))
    
    # A few conv layers so Grad-CAM has strong spatial features to hook into
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='target_conv_layer')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)
    
    # Binary output: Normal(0) vs Pneumonia(1). We zero the kernel 
    # and use a high positive bias so the untrained dummy model consistently flags Pneumonia for UI testing.
    outputs = layers.Dense(1, activation='sigmoid', 
                           kernel_initializer=tf.keras.initializers.Zeros(),
                           bias_initializer=tf.keras.initializers.Constant(5.0))(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Determine save path
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(os.path.dirname(curr_dir), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    save_path = os.path.join(models_dir, 'dummy_model.h5')
    
    # Save the untrained model
    model.save(save_path)
    print(f"Successfully saved Dummy Model to: {save_path}")
    print("The web app will use this fallback model if pneumonia_resnet_model.h5 doesn't exist.")

if __name__ == '__main__':
    create_dummy_model()
