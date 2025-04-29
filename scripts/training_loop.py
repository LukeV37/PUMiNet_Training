import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

### Define Training Loop
def train(model, loss_fns, optimizer, device, data, out_dir, step=15, gam=0.1, epochs=40):
    # Unpack arguments
    jet_loss_fn,trk_loss_fn=loss_fns
    X_train, y_train, X_val, y_val=data
    step_size=step
    gamma=gam

    # Initialize training history
    combined_history = []

    # Calculate number of events
    num_train = len(X_train)
    num_val = len(X_val)

    # Initiliaze learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    ###########################
    ### Begin training loop ###
    ###########################
    print("Begin Training Loop...")    
    for e in range(epochs):
        model.train()
        cumulative_loss_train = 0

        # Update model weights on training data
        for i in range(num_train):
            optimizer.zero_grad()

            jet_pred, trk_pred = model(X_train[i][0].to(device), X_train[i][1].to(device), X_train[i][2].to(device))

            jet_loss=jet_loss_fn(jet_pred, y_train[i][0].to(device))
            trk_loss=trk_loss_fn(trk_pred, y_train[i][1].to(device))

            loss = jet_loss+trk_loss

            loss.backward()
            optimizer.step()

            cumulative_loss_train+=loss.detach().cpu().numpy().mean()

        cumulative_loss_train = cumulative_loss_train / num_train

        # Evaluate model and track metrics on validation data
        model.eval()
        cumulative_loss_val = 0
        for i in range(num_val):
            jet_pred, trk_pred = model(X_val[i][0].to(device), X_val[i][1].to(device), X_val[i][2].to(device))
            
            jet_loss=jet_loss_fn(jet_pred, y_val[i][0].to(device))
            trk_loss=trk_loss_fn(trk_pred, y_val[i][1].to(device))

            loss = jet_loss+trk_loss

            cumulative_loss_val+=loss.detach().cpu().numpy().mean()

        cumulative_loss_val = cumulative_loss_val / num_val
        combined_history.append([cumulative_loss_train, cumulative_loss_val])

        # Increment scheduler
        scheduler.step()

        # Print metrics at the end of each epoch
        if e%1==0:
            print('\tEpoch:',e+1,'\tTrain Loss:',round(cumulative_loss_train,6),'\tVal Loss:',round(cumulative_loss_val,6))

        if (e+1)%step_size==0:
            print("\t\tReducing Step Size by ", gamma)
            
        torch.save(model,out_dir+"/model_Epoch_"+str(e+1)+".torch")

    return np.array(combined_history)
