#!/usr/bin/env python3
"""
Test script to verify the environment setup for the Fruit and Vegetable Classification project
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✓ Pandas {pd.__version__}")
    except ImportError as e:
        print(f"✗ Pandas import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ Matplotlib")
    except ImportError as e:
        print(f"✗ Matplotlib import failed: {e}")
        return False
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__}")
    except ImportError as e:
        print(f"✗ TensorFlow import failed: {e}")
        return False
    
    try:
        import keras
        print(f"✓ Keras {keras.__version__}")
    except ImportError as e:
        print(f"✗ Keras import failed: {e}")
        return False
    
    try:
        from keras.preprocessing.image import load_img, img_to_array
        print("✓ Keras image preprocessing")
    except ImportError as e:
        print(f"✗ Keras image preprocessing import failed: {e}")
        return False
    
    try:
        from keras.applications.mobilenet_v2 import preprocess_input
        print("✓ MobileNetV2 preprocessing")
    except ImportError as e:
        print(f"✗ MobileNetV2 preprocessing import failed: {e}")
        return False
    
    try:
        import streamlit as st
        print("✓ Streamlit")
    except ImportError as e:
        print(f"✗ Streamlit import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print("✓ Pillow (PIL)")
    except ImportError as e:
        print(f"✗ Pillow import failed: {e}")
        return False
    
    try:
        import requests
        print("✓ Requests")
    except ImportError as e:
        print(f"✗ Requests import failed: {e}")
        return False
    
    try:
        from bs4 import BeautifulSoup
        print("✓ BeautifulSoup")
    except ImportError as e:
        print(f"✗ BeautifulSoup import failed: {e}")
        return False
    
    return True

def test_data_structure():
    """Test if the data directory structure exists"""
    print("\nTesting data structure...")
    
    required_dirs = ['input/train', 'input/test', 'input/validation', 'upload_images']
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            # Count files in directory
            if dir_path.startswith('input/'):
                file_count = 0
                for root, dirs, files in os.walk(dir_path):
                    file_count += len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                print(f"✓ {dir_path} exists with {file_count} image files")
            else:
                print(f"✓ {dir_path} exists")
        else:
            print(f"✗ {dir_path} does not exist")
            if dir_path == 'upload_images':
                try:
                    os.makedirs(dir_path)
                    print(f"  Created {dir_path}")
                except Exception as e:
                    print(f"  Failed to create {dir_path}: {e}")

def test_gpu():
    """Test if GPU is available"""
    print("\nTesting GPU availability...")
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ GPU available: {len(gpus)} device(s)")
            for gpu in gpus:
                print(f"  - {gpu.name}")
        else:
            print("⚠ No GPU devices found, will use CPU")
    except Exception as e:
        print(f"✗ GPU test failed: {e}")

def main():
    print("Fruit and Vegetable Classification - Environment Test")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test data structure
    test_data_structure()
    
    # Test GPU
    test_gpu()
    
    print("\n" + "=" * 50)
    if imports_ok:
        print("✓ Environment setup looks good!")
        print("\nNext steps:")
        print("1. Run 'python fruit_veg_class.py' to train the model")
        print("2. Run 'streamlit run App.py' to start the web application")
    else:
        print("✗ Some imports failed. Please install missing packages:")
        print("  pip install -r requirements.txt")

if __name__ == "__main__":
    main() 