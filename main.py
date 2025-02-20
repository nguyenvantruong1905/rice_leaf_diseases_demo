import torch
from fastapi import FastAPI, File, UploadFile
import uvicorn
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import io
from timm import create_model

model = create_model("vit_small_patch16_224", pretrained=True, num_classes=3)
model.load_state_dict(torch.load("final_model.pth", map_location=torch.device("cpu")))
model.eval()

app = FastAPI()


def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0)  # add batch dimension

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    processed_image = preprocess_image(image)
    with torch.no_grad():
        prediction = model(processed_image)
        # softmax
        probabilities = F.softmax(prediction, dim=1)

        # (confidence) and predicted class
        confidence, predicted = torch.max(probabilities, 1)

        print(f"Predicted Label: {predicted}")
        print(f"Confidence Score: {confidence}")
        return {
            "predicted_class": predicted.item(),
            "confidence": confidence.item()}
            

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
