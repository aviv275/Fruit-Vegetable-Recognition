import streamlit as st
from PIL import Image
import numpy as np
import json
import os
import random

# Page configuration
st.set_page_config(
    page_title="Morcadona Smart Scale", 
    page_icon="🛒", 
    layout="centered"
)

# Custom CSS for Morcadona branding
st.markdown("""
<style>
    .main-header {
        color: #006830;
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    
    .stButton > button {
        min-height: 60px;
        font-size: 20px;
        border-radius: 12px;
        font-weight: bold;
    }
    
    .primary-button {
        background-color: #006830 !important;
        color: white !important;
        border: none !important;
    }
    
    .secondary-button {
        background-color: #FFB400 !important;
        color: #006830 !important;
        border: none !important;
    }
    
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 12px;
        border-left: 4px solid #006830;
    }
    
    .weight-badge {
        background-color: #006830;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Try to import TensorFlow/Keras with error handling
try:
    from keras.utils import load_img, img_to_array
    from keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    st.error(f"Error importing TensorFlow/Keras: {e}")
    TENSORFLOW_AVAILABLE = False

# Try to import Gemini with error handling
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Price per kg lookup (€/kg)
PRICE_PER_KG = {
    "apple": 1.90, "banana": 1.45, "beetroot": 1.20, "bell pepper": 2.25, 
    "cabbage": 0.99, "capsicum": 2.25, "carrot": 1.15, "cauliflower": 1.80,
    "chilli pepper": 3.50, "corn": 1.60, "cucumber": 1.30, "eggplant": 2.10,
    "garlic": 4.20, "ginger": 5.50, "grapes": 2.80, "jalepeno": 3.50,
    "kiwi": 3.20, "lemon": 2.40, "lettuce": 1.10, "mango": 3.80,
    "onion": 0.85, "orange": 1.95, "paprika": 4.50, "pear": 2.10,
    "peas": 2.80, "pineapple": 2.50, "pomegranate": 3.90, "potato": 0.99,
    "raddish": 1.25, "soy beans": 2.40, "spinach": 2.20, "sweetcorn": 1.60,
    "sweetpotato": 1.80, "tomato": 1.80, "turnip": 1.15, "watermelon": 1.20
}

# Load the model
@st.cache_resource
def load_classification_model():
    if not TENSORFLOW_AVAILABLE:
        return None
    
    if os.path.exists('FV2.h5'):
        try:
            model = load_model('FV2.h5')
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    else:
        st.error("Model file 'FV2.h5' not found.")
        return None

# Load class indices
@st.cache_data
def load_class_labels():
    if os.path.exists('class_indices.json'):
        with open('class_indices.json', 'r') as f:
            class_indices = json.load(f)
        return {v: k for k, v in class_indices.items()}
    else:
        # Fallback to hardcoded labels
        return {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 
                5: 'capsicum', 6: 'carrot', 7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 
                10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger', 14: 'grapes', 
                15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce', 19: 'mango', 
                20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 
                25: 'pineapple', 26: 'pomegranate', 27: 'potato', 28: 'raddish', 
                29: 'soy beans', 30: 'spinach', 31: 'sweetcorn', 32: 'sweetpotato', 
                33: 'tomato', 34: 'turnip', 35: 'watermelon'}

def get_weight():
    """Simulate scale weight reading (100g - 1.5kg)"""
    return random.randint(100, 1500)

def get_price_per_kg(item_name):
    """Get price per kg for an item"""
    item_lower = item_name.lower()
    return PRICE_PER_KG.get(item_lower, random.uniform(1.0, 3.0))

def predict_item(image_path, model, labels):
    """Predict the item in the image"""
    if model is None:
        return None, 0.0
        
    try:
        img = load_img(image_path, target_size=(224, 224, 3))
        img = img_to_array(img)
        img = img / 255
        img = np.expand_dims(img, [0])
        answer = model.predict(img, verbose=0)
        y_class = answer.argmax(axis=-1)
        confidence = float(answer.max())
        y = int(y_class[0])
        result = labels[y]
        return result.capitalize(), confidence
    except Exception as e:
        st.error(f"Error predicting image: {e}")
        return None, 0.0

def get_recipes(item_name, gemini_model):
    """Get Spanish recipes for the item using Gemini"""
    if gemini_model is None:
        return None
    
    try:
        prompt = f"""Give me two easy Spanish recipes that use: {item_name}.
Return each recipe as: title, ingredients list, instructions (≤100 words). No nutrition."""
        
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error getting recipes: {e}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">morcadona</h1>', unsafe_allow_html=True)
    
    # Initialize Gemini model
    gemini_model = None
    
    # Configure Gemini API from secrets
    if GEMINI_AVAILABLE:
        try:
            # Try to get API key from secrets
            api_key = st.secrets.get("GEMINI_API_KEY")
            if api_key:
                try:
                    # Configure Gemini API
                    genai.configure(api_key=api_key)
                    # Create model instance
                    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                    st.success("✅ Gemini AI configured from secrets!")
                except AttributeError:
                    # Fallback for different API versions
                    try:
                        gemini_model = genai.GenerativeModel('gemini-1.5-flash', api_key=api_key)
                        st.success("✅ Gemini AI configured from secrets!")
                    except Exception as e:
                        st.error(f"Error configuring Gemini: {e}")
                        gemini_model = None
                except Exception as e:
                    st.error(f"Error configuring Gemini: {e}")
                    gemini_model = None
            else:
                # Fallback to sidebar input if not in secrets
                with st.sidebar:
                    st.header("🔑 API Configuration")
                    api_key = st.text_input("Enter Gemini API Key", type="password", 
                                           help="Get your API key from https://makersuite.google.com/app/apikey")
                    if api_key:
                        try:
                            genai.configure(api_key=api_key)
                            gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                            st.success("API key configured!")
                        except Exception as e:
                            st.error(f"Error configuring Gemini: {e}")
                            gemini_model = None
                    else:
                        st.info("Enter API key to enable recipe features")
        except Exception as e:
            st.error(f"Error accessing secrets: {e}")
            gemini_model = None
    else:
        # Show warning in sidebar if Gemini not available
        with st.sidebar:
            st.header("🔑 API Configuration")
            st.warning("Gemini library not available. Install with: pip install google-generativeai")
    
    # Load model and labels
    model = load_classification_model()
    labels = load_class_labels()
    
    if model is None:
        st.warning("Model not available. Please ensure FV2.h5 is in the current directory.")
        return
    
    # Image input section
    st.markdown("### 📸 Place item on scale and capture image")
    
    # Try camera input first, fallback to file uploader
    camera_input = st.camera_input("Place item on scale and tap ⬆️")
    
    if camera_input is None:
        file_input = st.file_uploader("Or upload an image", type=["jpg", "png", "jpeg"])
        img_input = file_input
    else:
        img_input = camera_input
    
    if img_input is not None:
        # Display image
        img = Image.open(img_input).resize((400, 400))
        
        # Create container for image and weight badge
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.image(img, use_container_width=True)
        
        with col2:
            # Simulate weight reading
            weight_g = get_weight()
            st.markdown(f'<div class="weight-badge">{weight_g} g</div>', unsafe_allow_html=True)
        
        # Save image for processing
        os.makedirs('./upload_images', exist_ok=True)
        save_image_path = './upload_images/temp_image.jpg'
        with open(save_image_path, "wb") as f:
            f.write(img_input.getbuffer())
        
        # Predict item
        predicted_item, confidence = predict_item(save_image_path, model, labels)
        
        if predicted_item:
            # Display results
            st.markdown("---")
            
            # Item detection result
            confidence_pct = int(confidence * 100)
            st.markdown(f"### Detected item: {predicted_item} {get_emoji(predicted_item)} {confidence_pct}%")
            
            # Pricing information
            price_per_kg = get_price_per_kg(predicted_item)
            subtotal = (weight_g / 1000) * price_per_kg
            
            # Metrics display
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Weight", f"{weight_g} g")
            with col2:
                st.metric("Price", f"€{price_per_kg:.2f}/kg")
            with col3:
                st.metric("Subtotal", f"€{subtotal:.2f}")
            
            # Action buttons
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("✅ Confirm", key="confirm", use_container_width=True):
                    st.success(f"✅ {predicted_item} added to cart - €{subtotal:.2f}")
            
            with col2:
                if st.button("🍽 Get a recipe", key="recipe", use_container_width=True):
                    st.session_state.show_recipes = True
            
            # Recipe expander
            if st.session_state.get('show_recipes', False):
                with st.expander("🍽 Recipes", expanded=True):
                    if gemini_model:
                        with st.spinner("Getting Spanish recipes..."):
                            recipes = get_recipes(predicted_item, gemini_model)
                        
                        if recipes:
                            st.markdown(recipes)
                            
                            # Download button
                            recipe_text = f"Spanish Recipes for {predicted_item}\n\n{recipes}"
                            st.download_button(
                                label="📥 Download Recipes",
                                data=recipe_text,
                                file_name=f"{predicted_item}_spanish_recipes.txt",
                                mime="text/plain"
                            )
                        else:
                            st.warning("Could not fetch recipes. Please check your API key.")
                    else:
                        st.info("Please configure Gemini API key in secrets.toml or sidebar to get recipes.")
        else:
            st.warning("Could not detect item. Please try again with a clearer image.")

def get_emoji(item_name):
    """Get emoji for item"""
    emoji_map = {
        'Apple': '🍎', 'Banana': '🍌', 'Beetroot': '🫘', 'Bell pepper': '🫑', 
        'Cabbage': '🥬', 'Capsicum': '🫑', 'Carrot': '🥕', 'Cauliflower': '🥦',
        'Chilli pepper': '🌶️', 'Corn': '🌽', 'Cucumber': '🥒', 'Eggplant': '🍆',
        'Garlic': '🧄', 'Ginger': '🫚', 'Grapes': '🍇', 'Jalepeno': '🌶️',
        'Kiwi': '🥝', 'Lemon': '🍋', 'Lettuce': '🥬', 'Mango': '🥭',
        'Onion': '🧅', 'Orange': '🍊', 'Paprika': '🫑', 'Pear': '🍐',
        'Peas': '🫛', 'Pineapple': '🍍', 'Pomegranate': '石榴', 'Potato': '🥔',
        'Raddish': '🥬', 'Soy beans': '🫘', 'Spinach': '🥬', 'Sweetcorn': '🌽',
        'Sweetpotato': '🍠', 'Tomato': '🍅', 'Turnip': '🥬', 'Watermelon': '🍉'
    }
    return emoji_map.get(item_name, '🥬')

if __name__ == "__main__":
    main()
