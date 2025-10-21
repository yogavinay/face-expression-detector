#!/usr/bin/env python3
"""
Test script to verify the face expression detector setup.
Run this before using the main application.
"""

import sys
import os
import cv2
import numpy as np

def test_imports():
    """Test if all required packages are installed."""
    print("Testing imports...")
    try:
        import tensorflow as tf
        print("OK TensorFlow:", tf.__version__)
    except ImportError:
        print("ERROR TensorFlow not found")
        return False
    
    try:
        import cv2
        print("OK OpenCV:", cv2.__version__)
    except ImportError:
        print("ERROR OpenCV not found")
        return False
    
    try:
        import numpy as np
        print("OK NumPy:", np.__version__)
    except ImportError:
        print("ERROR NumPy not found")
        return False
    
    return True

def test_files():
    """Test if required files exist."""
    print("\nChecking required files...")
    
    required_files = [
        'haarcascade_frontalface_default.xml',
        'emotion_detection_model.h5'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"OK {file}")
        else:
            print(f"ERROR {file} - MISSING")
            missing_files.append(file)
    
    return missing_files

def test_camera():
    """Test if camera is accessible."""
    print("\nTesting camera access...")
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("OK Camera accessible")
                cap.release()
                return True
            else:
                print("ERROR Camera accessible but can't read frames")
                cap.release()
                return False
        else:
            print("ERROR Camera not accessible")
            return False
    except Exception as e:
        print(f"ERROR Camera error: {e}")
        return False

def test_model():
    """Test if the emotion model can be loaded."""
    print("\nTesting emotion model...")
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model('emotion_detection_model.h5')
        print("OK Emotion model loaded successfully")
        
        # Test prediction
        dummy_input = np.random.random((1, 48, 48, 1)).astype('float32')
        prediction = model.predict(dummy_input, verbose=0)
        print(f"OK Model prediction test passed (shape: {prediction.shape})")
        return True
    except Exception as e:
        print(f"ERROR Model loading failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Face Expression Detector - Setup Test")
    print("=" * 50)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test files
    missing_files = test_files()
    if missing_files:
        all_passed = False
        print(f"\n💡 Missing files: {', '.join(missing_files)}")
        if 'emotion_detection_model.h5' in missing_files:
            print("   Run: python generate_dummy_model.py")
    
    # Test camera
    if not test_camera():
        all_passed = False
    
    # Test model (only if file exists)
    if os.path.exists('emotion_detection_model.h5'):
        if not test_model():
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("SUCCESS: All tests passed! You're ready to run the detector.")
        print("   Run: python emotion_detector.py")
    else:
        print("WARNING: Some tests failed. Please fix the issues above.")
        print("   Check the README.md for troubleshooting tips.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
