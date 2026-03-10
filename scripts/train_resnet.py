import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

def build_model():
    input_shape = (224, 224, 3)
    
    # 1. Load ResNet50 pretrained on ImageNet without top layers
    # This leaves the final spatial convolution layers exposed for Grad-CAM
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # 2. Freeze the base layers so we only train our new classifier
    for layer in base_model.layers:
        layer.trainable = False
        
    # 3. Add custom classification layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(1, activation='sigmoid')(x) # Binary classification: Normal vs Pneumonia
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # We do NOT compile the model here anymore, we will compile it at the start of each Stage
    return model

def train_model():
    # Defining paths based on required folder structure
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, 'dataset')
    
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')
    # test_dir = os.path.join(dataset_dir, 'test') # Test dir can be used for evaluation if needed
    
    if not os.path.exists(train_dir):
        print(f"Error: Dataset directory not found at {train_dir}")
        print("Please ensure you have placed the Kaggle dataset in the 'dataset' folder.")
        return
        
    # Data preprocessing and augmentation
    # Using rescale=1./255, small rotation, zoom, and horizontal flip for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        zoom_range=0.1,
        horizontal_flip=True
    )
    
    # Validation data should only be rescaled, not augmented
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    batch_size = 16
    
    print("Loading training data...")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary'
    )
    
    print("Loading validation data...")
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary'
    )
    
    model = build_model()
    model.summary()
    
    # --- STAGE 1: Train Classifier Head (Backbone Frozen) ---
    print("\n--- STAGE 1: Training Classifier Head (Backbone Frozen) ---")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    
    history_stage1 = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=5,
        validation_data=val_generator,
        validation_steps=val_generator.samples // batch_size
    )
    
    # --- STAGE 2: Progressive Unfreezing (Top 50 layers) ---
    print("\n--- STAGE 2: Progressive Unfreezing (Top 50 layers) ---")
    for layer in model.layers[-50:]:
        # Keep BatchNormalization layers frozen to prevent weight destruction
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    
    history_stage2 = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=10,
        validation_data=val_generator,
        validation_steps=val_generator.samples // batch_size
    )
    
    # --- STAGE 3: Full Fine-tuning with Very Low Learning Rate ---
    print("\n--- STAGE 3: Full Fine-Tuning (All layers) ---")
    for layer in model.layers:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
            
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    
    history_stage3 = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=5,
        validation_data=val_generator,
        validation_steps=val_generator.samples // batch_size
    )
    
    # Save the trained model
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_save_path = os.path.join(models_dir, 'resnet_pneumonia_model.h5')
    
    model.save(model_save_path)
    print(f"\nTraining complete! Model saved successfully to:")
    print(f"-> {model_save_path}")

if __name__ == '__main__':
    train_model()
