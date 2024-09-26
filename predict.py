import torch 
from PIL import Image 
from torchvision import transforms
from models.model import TumorClassifier
from config import config
import os 
import csv



def load_model():
    model = TumorClassifier(num_classes=config["num_classes"])
    model.load_state_dict(torch.load("brain_tumor_classifier.pth"))
    model.eval()
    return model


def predict_image(model, image_path):
    
    # preprocess the image 
    transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.485,0.406], [0.229,0.224,0.225])])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)


    # Prediction 
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output,1)
    
    return "Tumor" if predicted.item() == 1 else "No Tumor"

def predict_on_dataset(test_dir):
    model = load_model()
    results = []

    for subfolder in os.listdir(test_dir):
        subfolder_path = os.path.join(test_dir, subfolder)
        if os.path.isdir(subfolder_path):  # Check if it's a directory
            for img_name in os.listdir(subfolder_path):
                if img_name.endswith('.jpg'):  # Process only supported image formats
                    img_path = os.path.join(subfolder_path, img_name)
                    prediction = predict_image(model, img_path)
                    results.append((f"{subfolder}/{img_name}", prediction))  # Include subfolder in result
                    print(f"Image: {subfolder}/{img_name}, Prediction: {prediction}")

    return results

    


if __name__ == "__main__":
    test_directory = "/home/gs165/Downloads/TestingMRI"
    predictions = predict_on_dataset(test_directory)

    with open('predictions.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Image Name', 'Prediction'])  # Write header

        for img_name, pred in predictions:
            csv_writer.writerow([img_name, pred])  # Write each prediction

    print("Predictions saved to predictions.csv")