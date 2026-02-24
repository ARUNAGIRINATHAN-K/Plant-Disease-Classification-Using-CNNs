"""
app.py — Hugging Face Spaces entry point
Plant Disease Classification using EfficientNetB3
"""

import os
import numpy as np
import gradio as gr
import tensorflow as tf
from tensorflow import keras
from PIL import Image

# ── Constants ──────────────────────────────────────────────────────────────────
IMG_SIZE   = (224, 224)
MODEL_PATH = os.environ.get('MODEL_PATH', 'plant_disease_model.h5')

CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
    'Apple___healthy', 'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# ── Disease information ────────────────────────────────────────────────────────
DISEASE_INFO = {
    '_healthy': {
        'description': '✅ The plant appears **HEALTHY** with no visible disease signs.',
        'treatment'  : 'No treatment required. Continue standard care.',
        'prevention' : 'Maintain regular watering, fertilisation, and pest monitoring.',
        'severity'   : 'None 🟢'
    },
    'Apple___Apple_scab': {
        'description': '🍎 **Apple Scab** — fungal disease (Venturia inaequalis). '
                       'Causes olive-green to brown scab-like lesions on leaves and fruit.',
        'treatment'  : 'Apply fungicides (captan, myclobutanil, or sulfur) every 7–10 days.',
        'prevention' : 'Plant resistant varieties; ensure good air circulation.',
        'severity'   : 'Moderate 🟡'
    },
    'Apple___Black_rot': {
        'description': '🍎 **Apple Black Rot** — caused by Botryosphaeria obtusa. '
                       'Brown rotting spots on fruit; cankers on branches.',
        'treatment'  : 'Prune infected wood; apply copper-based or captan fungicides.',
        'prevention' : 'Avoid wounding trees; ensure good drainage.',
        'severity'   : 'High 🟠'
    },
    'Apple___Cedar_apple_rust': {
        'description': '🍎 **Cedar Apple Rust** — bright orange-yellow spots on apple leaves.',
        'treatment'  : 'Apply myclobutanil or mancozeb at pink bud stage.',
        'prevention' : 'Plant rust-resistant varieties; distance from junipers.',
        'severity'   : 'Moderate 🟡'
    },
    'Tomato___Early_blight': {
        'description': '🍅 **Tomato Early Blight** — dark concentric-ring lesions on older leaves.',
        'treatment'  : 'Apply chlorothalonil or copper fungicide every 7 days.',
        'prevention' : 'Rotate crops; mulch soil; avoid overhead watering.',
        'severity'   : 'Moderate 🟡'
    },
    'Tomato___Late_blight': {
        'description': '🍅 **Tomato Late Blight** — large, water-soaked greasy lesions. '
                       'Caused by Phytophthora infestans. Highly destructive.',
        'treatment'  : 'Apply mancozeb or metalaxyl immediately. Destroy infected plants.',
        'prevention' : 'Scout regularly; avoid overhead watering; use resistant varieties.',
        'severity'   : 'Critical 🔴'
    },
    'Tomato___Leaf_Mold': {
        'description': '🍅 **Tomato Leaf Mold** — pale yellow spots above; olive mold below leaves.',
        'treatment'  : 'Improve ventilation; apply copper or chlorothalonil fungicide.',
        'prevention' : 'Keep humidity <85%; prune lower leaves.',
        'severity'   : 'Moderate 🟡'
    },
    'Tomato___Septoria_leaf_spot': {
        'description': '🍅 **Septoria Leaf Spot** — small circular spots with dark borders on lower leaves.',
        'treatment'  : 'Apply chlorothalonil or copper-based fungicide; remove affected leaves.',
        'prevention' : 'Crop rotation; mulch; avoid wetting foliage.',
        'severity'   : 'Moderate 🟡'
    },
    'Tomato___Bacterial_spot': {
        'description': '🍅 **Bacterial Spot** — small water-soaked spots turning brown with yellow halos.',
        'treatment'  : 'Apply copper bactericide + mancozeb.',
        'prevention' : 'Use disease-free transplants; avoid overhead irrigation.',
        'severity'   : 'High 🟠'
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'description': '🍅 **TYLCV** — whitefly-transmitted virus. Upward leaf curling, '
                       'yellowing, stunted growth.',
        'treatment'  : 'No cure. Remove infected plants; control whiteflies.',
        'prevention' : 'Use resistant varieties; install insect-proof netting.',
        'severity'   : 'Critical 🔴'
    },
    'Potato___Early_blight': {
        'description': '🥔 **Potato Early Blight** — dark concentric lesions on older leaves.',
        'treatment'  : 'Apply chlorothalonil or mancozeb every 7–10 days.',
        'prevention' : 'Crop rotation; certified seed; maintain plant nutrition.',
        'severity'   : 'Moderate 🟡'
    },
    'Potato___Late_blight': {
        'description': '🥔 **Potato Late Blight** — rapidly spreading water-soaked lesions; '
                       'white mold on leaf underside.',
        'treatment'  : 'Apply metalaxyl + mancozeb immediately.',
        'prevention' : 'Plant resistant varieties; monitor weather forecasts.',
        'severity'   : 'Critical 🔴'
    },
    'Grape___Black_rot': {
        'description': '🍇 **Grape Black Rot** — brown leaf lesions; fruit shrivels to black mummies.',
        'treatment'  : 'Apply mancozeb or myclobutanil from bud break.',
        'prevention' : 'Prune for open canopy; destroy mummies before budbreak.',
        'severity'   : 'High 🟠'
    },
    'Corn_(maize)___Common_rust_': {
        'description': '🌽 **Corn Common Rust** — brick-red oval pustules on both leaf surfaces.',
        'treatment'  : 'Apply triazole or strobilurin fungicides at early detection.',
        'prevention' : 'Plant resistant hybrids; early planting.',
        'severity'   : 'Moderate 🟡'
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'description': '🌽 **Northern Corn Leaf Blight** — long cigar-shaped grayish lesions.',
        'treatment'  : 'Apply propiconazole or azoxystrobin at early tasseling.',
        'prevention' : 'Resistant hybrids; crop rotation; tillage.',
        'severity'   : 'Moderate 🟡'
    },
    'Orange___Haunglongbing_(Citrus_greening)': {
        'description': '🍊 **HLB / Citrus Greening** — incurable bacterial disease spread by psyllids. '
                       'Yellow mottling, lopsided bitter fruit, eventual tree death.',
        'treatment'  : 'No cure. Remove and destroy infected trees immediately.',
        'prevention' : 'Certified disease-free stock; psyllid control; quarantine.',
        'severity'   : 'Critical 🔴'
    },
    '__default__': {
        'description': '⚠️ Disease detected. Consult a local agricultural extension officer.',
        'treatment'  : 'Isolate affected plants; apply broad-spectrum fungicide/bactericide.',
        'prevention' : 'Practice crop rotation; maintain field hygiene; use certified seeds.',
        'severity'   : 'Unknown ⚪'
    },
}

# ── Load model ─────────────────────────────────────────────────────────────────
print(f'Loading model from: {MODEL_PATH}')
try:
    model = keras.models.load_model(MODEL_PATH)
    print('✅ Model loaded successfully!')
except Exception as e:
    print(f'⚠️  Model not found ({e}). Running in demo mode without predictions.')
    model = None

# ── Helper functions ───────────────────────────────────────────────────────────
def format_class(raw: str) -> str:
    parts = raw.split('___')
    crop  = parts[0].replace('_', ' ').strip()
    cond  = parts[1].replace('_', ' ').strip() if len(parts) > 1 else ''
    return f'{crop}  |  {cond}' if cond else crop


def get_disease_info(class_name: str) -> dict:
    if class_name in DISEASE_INFO:
        return DISEASE_INFO[class_name]
    for key in DISEASE_INFO:
        if key in class_name:
            return DISEASE_INFO[key]
    if 'healthy' in class_name.lower():
        return DISEASE_INFO['_healthy']
    return DISEASE_INFO['__default__']


def predict_disease(image):
    if image is None:
        return 'No image provided', '—', {}, '⚠️ Please upload a leaf image.', ''

    if model is None:
        return ('Demo Mode', '—', {},
                '⚠️ Model file not loaded. Upload `plant_disease_model.h5` to the Space.',
                '')

    # Preprocess
    img  = image.convert('RGB').resize(IMG_SIZE)
    arr  = np.array(img, dtype=np.float32) / 255.0
    inp  = np.expand_dims(arr, axis=0)

    # Inference
    probs    = model.predict(inp, verbose=0)[0]
    top3_idx = np.argsort(probs)[::-1][:3]

    best_idx  = top3_idx[0]
    best_name = CLASS_NAMES[best_idx]
    best_conf = float(probs[best_idx])

    top3_dict = {
        format_class(CLASS_NAMES[i]): round(float(probs[i]) * 100, 2)
        for i in top3_idx
    }

    info = get_disease_info(best_name)
    info_md = (
        f'### 🔬 Disease Details\n\n'
        f'**Severity:** {info["severity"]}\n\n'
        f'{info["description"]}'
    )
    treatment_md = (
        f'### 💊 Treatment\n{info["treatment"]}\n\n'
        f'### 🛡️ Prevention\n{info["prevention"]}'
    )

    return format_class(best_name), f'{best_conf*100:.2f}%', top3_dict, info_md, treatment_md


# ── Gradio UI ──────────────────────────────────────────────────────────────────
css = """
#title    { text-align:center; color:#2d6a4f; }
#subtitle { text-align:center; color:#52796f; font-size:1rem; margin-bottom:8px; }
.footer   { text-align:center; color:#aaa; font-size:0.8rem; margin-top:8px; }
"""

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue   = "green",
        secondary_hue = "blue",
        font          = [gr.themes.GoogleFont('Inter'), 'Arial', 'sans-serif']
    ),
    css   = css,
    title = '🌿 Plant Disease Classification'
) as demo:

    gr.Markdown('# 🌿 Plant Disease Classification System', elem_id='title')
    gr.Markdown(
        'Upload a leaf image to instantly detect plant disease using AI.  \n'
        'Powered by **EfficientNetB3** trained on PlantVillage (38 classes).',
        elem_id='subtitle'
    )

    with gr.Row():
        with gr.Column(scale=4):
            img_input   = gr.Image(type='pil', label='📷 Upload Leaf Image',
                                   sources=['upload', 'clipboard'], height=300)
            predict_btn = gr.Button('🔍 Analyse Leaf', variant='primary', size='lg')
            gr.ClearButton(components=[img_input], value='🗑️ Clear', size='sm')

        with gr.Column(scale=6):
            with gr.Row():
                pred_out = gr.Textbox(label='🌿 Detected Disease', interactive=False, scale=3)
                conf_out = gr.Textbox(label='📊 Confidence',       interactive=False, scale=1)
            top3_out = gr.JSON(label='🏆 Top-3 Predictions (%)')
            with gr.Accordion('🔬 Disease Information', open=True):
                info_out = gr.Markdown('_Upload an image and click Analyse._')
            with gr.Accordion('💊 Treatment & Prevention', open=False):
                treat_out = gr.Markdown('')

    with gr.Accordion('ℹ️ How to Use', open=False):
        gr.Markdown("""
        1. **Upload** a clear close-up photo of a plant leaf (JPG / PNG)
        2. Click **Analyse Leaf**
        3. View the detected disease, confidence, and top-3 predictions
        4. Expand the accordions for treatment and prevention advice

        > **Supported crops:** Apple · Blueberry · Cherry · Corn · Grape · Orange ·
        Peach · Pepper · Potato · Raspberry · Soybean · Squash · Strawberry · Tomato
        """)

    gr.Markdown(
        '<p class="footer">🌿 Plant Disease Classification &nbsp;|&nbsp; '
        'EfficientNetB3 + PlantVillage &nbsp;|&nbsp; Built with TensorFlow & Gradio</p>'
    )

    predict_btn.click(predict_disease, inputs=[img_input],
                      outputs=[pred_out, conf_out, top3_out, info_out, treat_out])
    img_input.upload(predict_disease, inputs=[img_input],
                     outputs=[pred_out, conf_out, top3_out, info_out, treat_out])

if __name__ == '__main__':
    demo.launch()
