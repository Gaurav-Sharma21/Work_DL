import torch 
from torch import nn, optim 
from torch.utils.data import DataLoader
from data.dataset import BrainTumorDataset
from models.model import TumorClassifier
from utils.preprocess import get_transforms
from utils.evaluate import evaluate
from config import config

print("Training started")
def train():
    dataset = BrainTumorDataset(data_dir=config["data_dir"], transform=get_transforms())
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    # Loading the model 
    model = TumorClassifier(num_classes=config["num_classes"])
    model.to(config["device"])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = config["learning_rate"])

    # Training the Loop 
    for epoch in range(config["num_epochs"]):
        model.train()
        running_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(config["device"]), labels.to(config["device"])
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss, accuracy = evaluate(model, dataloader, criterion, config["device"])
        print(f"Epoch [{epoch + 1}/ {config['num_epochs']}], Loss: {avg_loss}, Accuracy: {accuracy}")
    
    torch.save(model.state_dict(), "brain_tumor_classifier.pth")

print("Training Finished")
if __name__ == "__main__":
    train()