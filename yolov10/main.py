import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
from ultralytics import YOLOv10

TRAINED_MODEL_PATH = 'D:\\AIO\\AIO-Project\\Helmet-Detection-YOLOv10\\yolov10\\best.pt'
model = YOLOv10(TRAINED_MODEL_PATH)

# Load the trained weights into your model
checkpoint = torch.load('D:\\AIO\\AIO-Project\\Helmet-Detection-YOLOv10\\yolov10\\best.pt')
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.eval()

# Define the transformation
def transform_image(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

# Streamlit UI
st.title("Helmet Detection with YOLOv10")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Detecting...")

    # Transform the image
    input_tensor = transform_image(image)

    # Get predictions
    with torch.no_grad():
        predictions = model([input_tensor])  # Adjust this line based on how your model expects input

    # Display predictions (this part is highly dependent on your model's output format)
    # You might need to adjust this to fit your model's specific output format
    for prediction in predictions:
        # Example of accessing prediction data (adjust according to your model's output)
        boxes = prediction['boxes']
        labels = prediction['labels']
        scores = prediction['scores']

        # Example: Print boxes - you would add drawing functionality here
        for box in boxes:
            st.write(f"Box: {box}")

# Note: This is a very basic example. You'll need to adjust the prediction and image drawing parts
# according to your model's specifics and how you want to display the results.