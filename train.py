import pickle
import os
import sys
import time
import matplotlib.pyplot as plt
import yaml

from scripts.training_loop import train
from models.Denoising_AE import *

import torch
import torch.optim as optim

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
print(config)

# Load configuration
Epochs = config["epochs"]
in_sample = config["in_sample"]
out_path = config["out_path"]
embed_dim = config["embed_dim"]
num_heads = config["num_heads"]
latent_dim = config["latent_dim"]
learning_rate = config["learning_rate"]
lr_step_size = config["lr_step_size"]
lr_step_gamma = config["lr_step_gamma"]

try:
    os.mkdir(out_path)
except OSError as error:
    print(error)
    print("Please try cleaning out your workspace or use new path! :)")
    sys.exit(1)

with open( in_sample, 'rb') as f:
    data = pickle.load(f)
X_train, y_train, X_val, y_val, X_test, y_test = [data["X_train"], data["y_train"], data["X_val"], data["y_val"], data["X_test"], data["y_test"]]

# Loss functions
loss_fn = nn.MSELoss()

# Get Instance of the model
model = Denoising_AE(embed_dim, num_heads, latent_dim)

print(model)
print()
print("Trainable Parameters :", sum(p.numel() for p in model.parameters() if p.requires_grad))
print()

# Pass some data to the model and print outputs
Event_no = 0

PU_weights = model(X_train[Event_no])

print("Test Case MSE Loss:", loss_fn(PU_weights,y_train[Event_no]))
print()

print("GPU Available: ", torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print()

###################
### Train Model ###
###################

# Initialize
model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
data = [X_train, y_train, X_val, y_val]

# Train
start_time = time.time()
combined_history = train(model, loss_fn, optimizer, device, data, out_path, lr_step_size, lr_step_gamma, Epochs)
end_time = time.time()
print()
print("Training Time: ", round(end_time-start_time,1),"(s)")
torch.save(model,out_path+'/model_final.torch')

# Plot loss
plt.figure()
plt.plot(combined_history[:,0], label="Train")
plt.plot(combined_history[:,1], label="Val")
plt.title('Loss')
plt.legend()
plt.yscale('log')
plt.savefig(out_path+"/Loss_Curve.png")
#plt.show()
