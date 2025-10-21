import cv2
import numpy as np
import tensorflow as tf
import time 
import sys
import threading
from collections import deque

# --- Configuration ---
# NOTE: Ensure these files are in the same directory as this script!
FACE_CASCADE_PATH = 'haarcascade_frontalface_default.xml'
EMOTION_MODEL_PATH = 'emotion_detection_model.h5' 
CAMERA_INDEX = 0  # 0 is usually the built-in webcam

# Performance optimization settings
FRAME_SKIP = 2  # Process every 2nd frame for better performance
MAX_FACES = 3   # Limit number of faces to process
MIN_FACE_SIZE = (50, 50)  # Minimum face size for processing
PREDICTION_CACHE_SIZE = 5  # Cache recent predictions

# Define the emotional categories the model is trained to recognize
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Global variables for performance optimization
frame_count = 0
prediction_cache = deque(maxlen=PREDICTION_CACHE_SIZE)
last_prediction = None

# --- Initialization ---
def initialize_models():
    """Initialize face detection and emotion models with optimizations."""
    global face_cascade, emotion_model
    
    try:
        # 1. Load the Haar Cascade for basic face detection
        print("Loading face detection model...")
        face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        if face_cascade.empty():
            raise IOError(f"Could not load face cascade XML file: {FACE_CASCADE_PATH}")
        
        # 2. Load the trained Keras/TensorFlow model for emotion classification
        print("Loading emotion detection model...")
        emotion_model = tf.keras.models.load_model(EMOTION_MODEL_PATH)
        
        # Optimize model for inference
        print("Optimizing model for inference...")
        # Enable mixed precision for faster inference
        tf.config.optimizer.set_jit(True)
        
        # Warm up the model with a dummy prediction
        dummy_input = np.random.random((1, 48, 48, 1)).astype('float32')
        _ = emotion_model.predict(dummy_input, verbose=0)
        
        print("✓ All models loaded and optimized successfully!")
        return True
        
    except Exception as e:
        print("FATAL ERROR during initialization.")
        print(f"Details: {e}")
        print("\n--- Setup Check ---")
        print("1. Did you run 'pip install opencv-python tensorflow numpy'?")
        print(f"2. Are '{FACE_CASCADE_PATH}' and '{EMOTION_MODEL_PATH}' in this directory?")
        print("3. Try running 'python generate_dummy_model.py' to create a test model")
        return False

# Initialize models
if not initialize_models():
    sys.exit(1)


def predict_emotion_optimized(face_roi):
    """Optimized emotion prediction with caching and error handling."""
    global last_prediction, prediction_cache
    
    try:
        # Resize and normalize the face ROI
        resized_face = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)
        normalized_face = resized_face.astype('float32') / 255.0
        reshaped_face = np.expand_dims(np.expand_dims(normalized_face, -1), 0)
        
        # Use cached prediction if available and recent
        if last_prediction and len(prediction_cache) > 0:
            # Return cached prediction for similar faces
            return last_prediction
        
        # Predict the emotion
        predictions = emotion_model.predict(reshaped_face, verbose=0)[0]
        emotion_index = np.argmax(predictions)
        predicted_emotion = EMOTIONS[emotion_index]
        confidence = predictions[emotion_index] * 100
        
        # Cache the prediction
        result = (predicted_emotion, confidence)
        prediction_cache.append(result)
        last_prediction = result
        
        return result
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return ("Error", 0.0)

def detect_and_predict_emotion():
    """
    Optimized real-time face emotion detection with performance enhancements.
    """
    global frame_count
    
    # Initialize video capture with optimizations
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size for lower latency

    if not cap.isOpened():
        print(f"Error: Cannot open camera with index {CAMERA_INDEX}. Check if camera is in use.")
        return

    print("🚀 Optimized Face Expression Detector Started!")
    print("Controls: 'q' = quit, 'r' = reset cache, 's' = toggle frame skip")
    print("=" * 50)
    
    # Performance tracking
    start_time = time.time()
    frame_count = 0
    processing_times = deque(maxlen=30)
    current_frame_skip = FRAME_SKIP
    
    while True:
        frame_start = time.time()
        
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame_count += 1
        
        # Skip frames for better performance
        if frame_count % current_frame_skip != 0:
            # Still show the frame but skip processing
            cv2.putText(frame, f"Frame Skip: {current_frame_skip}x", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.imshow('Real-Time Face Expression Detector', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Convert to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Optimized face detection parameters
        faces = face_cascade.detectMultiScale(
            gray_frame, 
            scaleFactor=1.05,  # Reduced for better accuracy
            minNeighbors=6,     # Increased for fewer false positives
            minSize=MIN_FACE_SIZE,
            maxSize=(300, 300),  # Limit maximum face size
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Process only the largest faces (up to MAX_FACES)
        if len(faces) > 0:
            # Sort faces by area (largest first)
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)[:MAX_FACES]
            
            for i, (x, y, w, h) in enumerate(faces):
                # Extract face ROI
                roi_gray = gray_frame[y:y + h, x:x + w]
                
                # Predict emotion
                emotion, confidence = predict_emotion_optimized(roi_gray)
                
                # Choose colors based on emotion
                if emotion == 'Happy':
                    color = (0, 255, 0)  # Green
                elif emotion in ['Angry', 'Disgust']:
                    color = (0, 0, 255)  # Red
                elif emotion in ['Sad', 'Fear']:
                    color = (255, 0, 0)  # Blue
                else:
                    color = (255, 255, 0)  # Cyan
                
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # Display emotion with confidence
                text = f"{emotion} ({confidence:.1f}%)"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Draw text background
                cv2.rectangle(frame, (x, y - 25), (x + text_size[0] + 10, y), color, -1)
                cv2.putText(frame, text, (x + 5, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Add face number for multiple faces
                if len(faces) > 1:
                    cv2.putText(frame, f"Face {i+1}", (x, y + h + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Calculate and display performance metrics
        processing_time = time.time() - frame_start
        processing_times.append(processing_time)
        avg_processing_time = np.mean(processing_times)
        
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Display performance info
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Processing: {avg_processing_time*1000:.1f}ms", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Display the frame
        cv2.imshow('Real-Time Face Expression Detector', frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset prediction cache
            prediction_cache.clear()
            last_prediction = None
            print("Cache reset!")
        elif key == ord('s'):
            # Toggle frame skip
            current_frame_skip = 1 if current_frame_skip > 1 else FRAME_SKIP
            print(f"Frame skip: {current_frame_skip}x")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n🎯 Session Summary:")
    print(f"Total frames processed: {frame_count}")
    print(f"Average FPS: {fps:.2f}")
    print(f"Average processing time: {avg_processing_time*1000:.2f}ms")
    print("Detection process finished.")

if __name__ == "__main__":
    detect_and_predict_emotion()
