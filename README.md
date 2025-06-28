# Fruit and Vegetable Recognition System

A deep learning-based image classification system that can identify 36 different types of fruits and vegetables using Convolutional Neural Networks (CNN) and transfer learning.

## About the Application

This project implements a real-time fruit and vegetable classification system with the following capabilities:

- **Image Classification**: Automatically identifies 36 different fruits and vegetables from uploaded images
- **Category Detection**: Distinguishes between fruits and vegetables
- **Nutritional Information**: Provides calorie information for identified items
- **User-Friendly Interface**: Web-based application built with Streamlit for easy interaction
- **Real-Time Processing**: Instant classification results with confidence scores

## CNN Algorithm Overview

The system uses a **Convolutional Neural Network (CNN)** based on **MobileNetV2** architecture with transfer learning:

### Architecture Details:
- **Base Model**: MobileNetV2 pre-trained on ImageNet dataset
- **Input Processing**: Images resized to 224x224x3 RGB format
- **Feature Extraction**: Leverages pre-trained convolutional layers for robust feature detection
- **Classification Head**: Custom dense layers for 36-class classification
- **Transfer Learning**: Fine-tunes pre-trained weights for fruit/vegetable recognition

### Key CNN Components:
1. **Convolutional Layers**: Extract hierarchical features (edges, textures, shapes)
2. **Pooling Layers**: Reduce spatial dimensions while preserving important features
3. **Activation Functions**: ReLU for non-linearity
4. **Dropout**: Prevents overfitting during training
5. **Dense Layers**: Final classification with softmax activation

### Training Strategy:
- **Data Augmentation**: Rotation, zoom, flip, and shift transformations
- **Transfer Learning**: Freezes base model layers, trains only classification head
- **Optimization**: Adam optimizer with categorical crossentropy loss
- **Validation**: 80-10-10 split (train-validation-test)

## Dataset

The system is trained on a comprehensive dataset containing:
- **36 Classes**: 15 fruits and 21 vegetables
- **Image Quality**: High-resolution images with various lighting conditions
- **Data Split**: Training (80%), Validation (10%), Test (10%)

### Supported Classes:

**Fruits (15)**: Apple, Banana, Bell Pepper, Chilli Pepper, Grapes, Jalepeno, Kiwi, Lemon, Mango, Orange, Paprika, Pear, Pineapple, Pomegranate, Watermelon

**Vegetables (21)**: Beetroot, Cabbage, Capsicum, Carrot, Cauliflower, Corn, Cucumber, Eggplant, Garlic, Ginger, Lettuce, Onion, Peas, Potato, Raddish, Soy Beans, Spinach, Sweetcorn, Sweetpotato, Tomato, Turnip

## Project Structure

```
Fruit_Vegetable_Recognition_New/
├── input/
│   ├── train/          # Training images (36 classes)
│   ├── test/           # Test images (36 classes)
│   └── validation/     # Validation images (36 classes)
├── upload_images/      # Directory for uploaded images
├── App.py             # Streamlit web application
├── fruit_veg_class.py  # Model training script
├── FV2.h5             # Trained CNN model
├── class_indices.json # Class labels mapping
└── README.md          # This file
```

## Installation and Setup

### Prerequisites
- Python 3.7+
- TensorFlow 2.15.0
- Keras 2.15.0
- Streamlit
- Other dependencies (see requirements.txt)

### Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   streamlit run App.py
   ```

3. **Access the Web Interface**:
   - Open your browser and go to `http://localhost:8501`
   - Upload an image of a fruit or vegetable
   - Get instant classification results with nutritional information

## Model Performance

The CNN model typically achieves:
- **Training Accuracy**: 95-98%
- **Validation Accuracy**: 90-95%
- **Test Accuracy**: 85-92%
- **Inference Time**: < 1 second per image

## Technical Features

### Image Processing Pipeline:
1. **Preprocessing**: Resize to 224x224, normalize pixel values (0-1)
2. **Feature Extraction**: CNN layers extract hierarchical features
3. **Classification**: Dense layers with softmax for class prediction
4. **Post-processing**: Confidence scoring and category assignment

### Web Application Features:
- **Real-time Upload**: Drag-and-drop or file browser interface
- **Instant Results**: Immediate classification with confidence display
- **Nutritional Data**: Calorie information with fallback database
- **Responsive Design**: Works on desktop and mobile devices

## Usage Examples

### Web Interface:
1. Upload an image through the web interface
2. View the classification result (fruit/vegetable category)
3. See the predicted item name
4. Get calorie information per 100 grams

### Programmatic Usage:
```python
from keras.models import load_model
from keras.utils import load_img, img_to_array
import numpy as np

# Load the trained model
model = load_model('FV2.h5')

# Load class labels
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)
labels = {v: k for k, v in class_indices.items()}

# Predict function
def predict_image(image_path):
    img = load_img(image_path, target_size=(224, 224, 3))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    predicted_class = prediction.argmax(axis=-1)[0]
    return labels[predicted_class]
```

## Future Enhancements

Potential improvements for the system:
- **Multi-language Support**: Interface in different languages
- **Nutritional Database**: Expanded nutritional information
- **Recipe Suggestions**: Recipe recommendations based on identified items
- **Mobile App**: Native mobile application
- **Batch Processing**: Multiple image classification
- **API Integration**: RESTful API for third-party applications

## Technical Requirements

- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: 2GB free space for model and dataset
- **Processing**: CPU or GPU (GPU recommended for training)
- **Network**: Internet connection for calorie data fetching

## License

This project is developed for educational and research purposes. The CNN model and web application are designed to demonstrate deep learning capabilities in computer vision applications.

---

*This system demonstrates the power of Convolutional Neural Networks in real-world image classification tasks, providing an accessible interface for users to interact with advanced AI technology.*
