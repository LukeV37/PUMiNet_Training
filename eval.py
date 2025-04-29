from models.PUMiNet import *
from scripts.eval_model import *

import pickle
import yaml
import torch
import torch.nn as nn

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
print(config)

# Load configuration
in_sample = config["in_sample"]
eval_model = config["eval_model"]
out_path = config["out_path"]

# Load data
with open( in_sample, 'rb') as f:
    data = pickle.load(f)
X_train, y_train, X_val, y_val, X_test, y_test = data

# Load model
model = torch.load(eval_model, weights_only=False)

# Get device
print("GPU Available: ", torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print()

# Loss functions
jet_loss_fn = nn.MSELoss()
trk_loss_fn = nn.BCELoss()
loss_fns = [jet_loss_fn, trk_loss_fn]

# Test Data
data = [X_test, y_test]

# Get Predictions
get_predictions(model, data, loss_fns, device, out_path)
