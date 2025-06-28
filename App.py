import streamlit as st
from PIL import Image
import requests
from bs4 import BeautifulSoup
import numpy as np
from keras.utils import load_img, img_to_array
from keras.models import load_model
import json
import os

# Load the model
if os.path.exists('FV2.h5'):
    model = load_model('FV2.h5')
else:
    st.error("Model file 'FV2.h5' not found. Please run the training script first.")
    st.stop()

# Load class indices
if os.path.exists('class_indices.json'):
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)
    # Create reverse mapping
    labels = {v: k for k, v in class_indices.items()}
else:
    # Fallback to hardcoded labels if JSON doesn't exist
    labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot',
              7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger',
              14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce',
              19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple',
              26: 'pomegranate', 27: 'potato', 28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn',
              32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'}

fruits = ['Apple', 'Banana', 'Bell Pepper', 'Chilli Pepper', 'Grapes', 'Jalepeno', 'Kiwi', 'Lemon', 'Mango', 'Orange',
          'Paprika', 'Pear', 'Pineapple', 'Pomegranate', 'Watermelon']
vegetables = ['Beetroot', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Corn', 'Cucumber', 'Eggplant', 'Ginger',
              'Lettuce', 'Onion', 'Peas', 'Potato', 'Raddish', 'Soy Beans', 'Spinach', 'Sweetcorn', 'Sweetpotato',
              'Tomato', 'Turnip']


def fetch_calories(prediction):
    # Hardcoded calorie data as fallback
    calorie_data = {
        'apple': '52 calories',
        'banana': '89 calories', 
        'beetroot': '43 calories',
        'bell pepper': '31 calories',
        'cabbage': '25 calories',
        'capsicum': '31 calories',
        'carrot': '41 calories',
        'cauliflower': '25 calories',
        'chilli pepper': '40 calories',
        'corn': '86 calories',
        'cucumber': '16 calories',
        'eggplant': '25 calories',
        'garlic': '149 calories',
        'ginger': '80 calories',
        'grapes': '62 calories',
        'jalepeno': '29 calories',
        'kiwi': '61 calories',
        'lemon': '29 calories',
        'lettuce': '15 calories',
        'mango': '60 calories',
        'onion': '40 calories',
        'orange': '47 calories',
        'paprika': '282 calories',
        'pear': '57 calories',
        'peas': '84 calories',
        'pineapple': '50 calories',
        'pomegranate': '83 calories',
        'potato': '77 calories',
        'raddish': '16 calories',
        'soy beans': '147 calories',
        'spinach': '23 calories',
        'sweetcorn': '86 calories',
        'sweetpotato': '86 calories',
        'tomato': '18 calories',
        'turnip': '28 calories',
        'watermelon': '30 calories'
    }
    
    try:
        # Try web scraping first
        url = 'https://www.google.com/search?&q=calories in ' + prediction
        req = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).text
        scrap = BeautifulSoup(req, 'html.parser')
        
        # Try multiple possible selectors
        calories_element = scrap.find("div", class_="BNeawe iBp4i AP7Wnd")
        if not calories_element:
            calories_element = scrap.find("div", class_="BNeawe s3v9rd AP7Wnd")
        if not calories_element:
            calories_element = scrap.find("span", class_="BNeawe s3v9rd AP7Wnd")
            
        if calories_element and calories_element.text:
            return calories_element.text
        else:
            # Fallback to hardcoded data
            prediction_lower = prediction.lower()
            if prediction_lower in calorie_data:
                return calorie_data[prediction_lower]
            else:
                return "Calorie information not available"
                
    except Exception as e:
        # Fallback to hardcoded data on any error
        prediction_lower = prediction.lower()
        if prediction_lower in calorie_data:
            return calorie_data[prediction_lower]
        else:
            st.error("Can't fetch the Calories")
            print(f"Error fetching calories: {e}")
            return "Calorie information not available"


def prepare_image(img_path):
    if model is None:
        st.error("Model not loaded properly")
        return None
        
    try:
        img = load_img(img_path, target_size=(224, 224, 3))
        img = img_to_array(img)
        img = img / 255
        img = np.expand_dims(img, [0])
        answer = model.predict(img)
        y_class = answer.argmax(axis=-1)
        print(y_class)
        y = " ".join(str(x) for x in y_class)
        y = int(y)
        res = labels[y]
        print(res)
        return res.capitalize()
    except Exception as e:
        st.error(f"Error predicting image: {e}")
        return None


def run():
    st.title("Fruits🍍-Vegetable🍅 Classification")
    img_file = st.file_uploader("Choose an Image", type=["jpg", "png", "jpeg"])
    if img_file is not None:
        img = Image.open(img_file).resize((250, 250))
        st.image(img, use_container_width=False)
        save_image_path = './upload_images/' + img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        # if st.button("Predict"):
        if img_file is not None:
            result = prepare_image(save_image_path)
            if result is None:
                st.error("Failed to predict the image")
                return
                
            if result in vegetables:
                st.info('**Category : Vegetables**')
            else:
                st.info('**Category : Fruit**')
            st.success("**Predicted : " + result + '**')
            cal = fetch_calories(result)
            if cal:
                st.warning('**' + cal + '(100 grams)**')


run()
