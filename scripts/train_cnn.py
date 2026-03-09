import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def build_basic_cnn(input_shape=(224, 224, 3)):
    """Builds a custom CNN model from scratch."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5), # Regularization
        Dense(1, activation='sigmoid') # Binary Output
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    return model

def train():
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset_split')
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    
    if not os.path.exists(train_dir):
        print("Dataset not found. Please run prepare_data.py first.")
        return

    # Basic Data Augmentation for CNN
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    batch_size = 32
    
    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=(224, 224), batch_size=batch_size, class_mode='binary'
    )
    
    val_gen = val_datagen.flow_from_directory(
        val_dir, target_size=(224, 224), batch_size=batch_size, class_mode='binary'
    )
    
    model = build_basic_cnn()
    
    # Save best model
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)
    checkpoint_path = os.path.join(models_dir, 'pneumonia_basic_cnn.h5')
    
    callback_list = [
        ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_accuracy'),
        EarlyStopping(patience=5, restore_best_weights=True)
    ]
    
    print("Training Basic Custom CNN...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20,
        callbacks=callback_list
    )
    
    print(f"Basic CNN Training Complete. Model saved to {checkpoint_path}")

if __name__ == '__main__':
    train()
