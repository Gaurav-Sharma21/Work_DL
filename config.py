import torch
config = {
    "data_dir" : "/home/gs165/Downloads/TrainingMRI",
    "batch_size" : 32,
    "num_epochs" : 25,
    "learning_rate" : 1e-4,
    "num_classes": 2,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}