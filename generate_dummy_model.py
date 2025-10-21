import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import sys
import numpy as np

# Define the structure the main script expects (48x48 grayscale input, 7 emotion outputs)
def create_dummy_model():
    """Creates an optimized dummy model with the expected I/O dimensions and saves it as emotion_detection_model.h5."""
    
    # Input Shape: (48, 48, 1) -> 48x48 pixel grayscale image
    # Output Shape: 7 classes (emotions)
    
    print("Creating optimized dummy model structure...")
    print("This model will provide random but realistic predictions for testing.")
    
    model = Sequential([
        # Optimized CNN structure for better performance
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')  # 7 output classes for the emotions
    ])

    # Compile the model with optimizations
    model.compile(
        optimizer='adam', 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    # Create some dummy training data to make the model more realistic
    print("Training model with dummy data for more realistic predictions...")
    dummy_x = np.random.random((1000, 48, 48, 1)).astype('float32')
    dummy_y = np.random.random((1000, 7)).astype('float32')
    
    # Normalize dummy_y to sum to 1 (like one-hot encoded labels)
    dummy_y = dummy_y / np.sum(dummy_y, axis=1, keepdims=True)
    
    # Train for a few epochs to make predictions more stable
    model.fit(dummy_x, dummy_y, epochs=5, batch_size=32, verbose=0)
    
    # Save the model with the correct name
    model.save('emotion_detection_model.h5')
    print("\n✅ SUCCESS: 'emotion_detection_model.h5' created!")
    print("📊 Model Summary:")
    model.summary()
    print("\n🚀 You can now run 'python emotion_detector.py' to test the webcam functionality!")
    print("💡 NOTE: This is a dummy model; for real emotion detection, train with actual emotion datasets.")

if __name__ == '__main__':
    try:
        create_dummy_model()
    except Exception as e:
        print(f"❌ Error creating dummy model: {e}")
        print("🔧 Ensure you have TensorFlow installed: pip install tensorflow")
        print("📦 Or install all requirements: pip install -r requirements.txt")
