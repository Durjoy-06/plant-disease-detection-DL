import torch
import torchvision.transforms as transforms
from PIL import Image
import sys
from models.cnn_model import MyCNN

# ক্লাস লিস্ট (আপনার ট্রেইন্ড ডিরেক্টরি সিকোয়েন্স অনুযায়ী)
# এখানে ডেমো হিসেবে আপনার ক্লাসের নামগুলো লোড করার কোড দেওয়া হলো

def predict_image(image_path, model_path, num_classes=38):
    # ট্রান্সফর্মেশন
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    
    # ইমেজ লোড
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0) # ব্যাচ ডাইমেনশন যোগ করা
    
    # মডেল লোড
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MyCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # প্রেডিকশন
    with torch.no_grad():
        outputs = model(image_tensor.to(device))
        _, predicted = torch.max(outputs, 1)
        
    print(f"Predicted Class Index: {predicted.item()}")
    return predicted.item()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
    else:
        img_path = sys.argv[1]
        predict_image(img_path, "plant_disease_model.pth")
