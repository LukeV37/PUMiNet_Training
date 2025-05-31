from models.Denoising_AE import Denoising_AE 
from scripts.eval_model import get_predictions_DAE

import pickle
import yaml
import torch
import torch.nn as nn

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
print(config)

# Load configuration
in_sample = config["in_sample"]
eval_model_path = config["eval_model"]
out_path = config["out_path"]

# Load data
with open( in_sample, 'rb') as f:
    data = pickle.load(f) # keys: X_train, y_train, X_val, y_val, X_test, y_test

# Loading Test Data
X_test = data["X_test"]
y_test = data["y_test"]

# Initialize and load model
model = Denoising_AE(embed_dim=config["embed_dim"],
                     num_heads=config["num_heads"],
                     latent_dim=config["latent_dim"])
# Load the trained model weights
model.load_state_dict(torch.load(eval_model_path, map_location=torch.device('cpu')))

# Get device
print("GPU Available: ", torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print()

# Loss functions
trk_loss_fn = nn.MSELoss() # for calculating avg test loss

# Test Data
data_for_eval = [X_test, y_test]

# Get Predictions
get_predictions_DAE(model, data_for_eval, trk_loss_fn, device, out_path)
