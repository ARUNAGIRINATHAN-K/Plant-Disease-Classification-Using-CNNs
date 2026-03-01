import gradio as gr
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import json
from torchvision import transforms

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

num_classes = len(class_names)

# Create model
# Create model EXACTLY like training
model = timm.create_model(
    "efficientnet_b0",
    pretrained=False,
    num_classes=len(class_names)
)

# Load trained weights
model.load_state_dict(
    torch.load("efficientnet_plant_best.pth", map_location=device),
    strict=True
)

model.to(device)
model.eval()

# IMPORTANT: Match Kaggle validation transform exactly
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def predict(image):
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)[0]

    top_prob, top_idx = torch.max(probs, 0)

    predicted_class = class_names[top_idx.item()]
    confidence = top_prob.item() * 100

    # Create plot with title
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis("off")
    ax.set_title(f"{predicted_class} ({confidence:.2f}%)")

    return fig

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Plot(),
    title="🌿 Plant Disease Classification",
    description="Upload a leaf image to detect disease using EfficientNet-B0"
)

interface.launch()