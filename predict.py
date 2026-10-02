import argparse
import torch
import torch.nn.functional as F
from PIL import Image

from animalvision.config import Config
from animalvision.model import AnimalVisionModel
from animalvision.data import get_data_transforms

CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def predict_image(image_path, model_path="best_model.pth", top_k=3):
    _, val_transform = get_data_transforms()
    image = Image.open(image_path).convert('RGB')
    input_tensor = val_transform(image).unsqueeze(0).to(Config.DEVICE)

    model = AnimalVisionModel(num_classes=Config.NUM_CLASSES)
    state_dict = torch.load(model_path, map_location=Config.DEVICE)
    
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
        
    model.load_state_dict(state_dict)
    model.to(Config.DEVICE)
    model.eval()

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        
    top_probs, top_indices = torch.topk(probabilities, k=top_k)

    print(f"\n--- Prediction Results for: {image_path} ---")
    for i in range(top_k):
        class_name = CLASSES[top_indices[i].item()]
        confidence = top_probs[i].item() * 100
        print(f"{i+1}. {class_name:<12} | Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on an image using AnimalVision.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file.")
    parser.add_argument("--model", type=str, default="best_model.pth", help="Path to saved model weights.")
    
    args = parser.parse_args()
    predict_image(args.image, model_path=args.model)