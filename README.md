# Morcadona Smart Scale 🍏

A modern, self-service produce scale kiosk for fruit and vegetable recognition, pricing, and instant Spanish recipe suggestions—Mercadona-style!

---

## Features

- **Camera or Upload**: Instantly capture or upload a photo of your produce.
- **AI Recognition**: Identifies 36 fruits and vegetables using a deep learning model.
- **Simulated Weight**: Demo mode simulates a real scale (100g–1.5kg).
- **Live Pricing**: Calculates price per kg and subtotal for each item.
- **Recipe Suggestions**: Get Spanish recipes for your produce (Gemini AI integration).
- **Tablet-Optimized UI**: Clean, touch-friendly, and brandable for real-world kiosks.
- **No Confirm Button**: Streamlined checkout—just scan and get info.

---

## Quick Start

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Add Your Gemini API Key

1. Get your key from: https://makersuite.google.com/app/apikey
2. Edit `.secrets.toml` in the project root:

   ```toml
   GEMINI_API_KEY = "your_gemini_api_key_here"
   ```

### 3. Run the App

```bash
streamlit run App.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Usage

1. **Place produce on the scale and use the camera** (or upload an image).
2. **See instant recognition**: Detected item, emoji, confidence, weight, price/kg, and subtotal.
3. **Tap "Get a recipe"** for two easy Spanish recipes (requires Gemini API key).
4. **No Confirm button**: The app is designed for seamless, kiosk-style flow.

---

## Project Structure

```
Fruit_Vegetable_Recognition_New/
├── App.py                # Main Streamlit app
├── FV2.h5                # Trained Keras model
├── class_indices.json    # Class label mapping
├── requirements.txt      # Python dependencies
├── .secrets.toml         # Gemini API key (not in git)
├── input/                # (Optional) Training/test images
├── upload_images/        # Temporary uploads
└── ...                   # Other scripts and docs
```

---

## Supported Produce

- **Fruits**: Apple, Banana, Bell Pepper, Chilli Pepper, Grapes, Jalepeno, Kiwi, Lemon, Mango, Orange, Paprika, Pear, Pineapple, Pomegranate, Watermelon
- **Vegetables**: Beetroot, Cabbage, Capsicum, Carrot, Cauliflower, Corn, Cucumber, Eggplant, Garlic, Ginger, Lettuce, Onion, Peas, Potato, Raddish, Soy Beans, Spinach, Sweetcorn, Sweetpotato, Tomato, Turnip

---

## Customization

- **Branding**: Easily change colors, logo, and UI text in `App.py` and CSS.
- **Real Scale Integration**: Replace the `get_weight()` function with real hardware input.
- **Pricing**: Edit the `PRICE_PER_KG` dictionary in `App.py` for your store's prices.

---

## Requirements

- Python 3.7+
- TensorFlow 2.15.0, Keras 2.15.0
- Streamlit, Pillow, numpy, google-generativeai, etc. (see `requirements.txt`)

---

## Security

- **API keys**: Never commit `.secrets.toml` to git.
- **User uploads**: Images are stored temporarily in `upload_images/`.

---

## License

For educational and demonstration use. Not for commercial deployment without further security and privacy review.

---

## Credits

Inspired by Mercadona's self-service produce scales.  
AI model and app by [Your Name/Team].

---

**Enjoy your smart produce scale!**
