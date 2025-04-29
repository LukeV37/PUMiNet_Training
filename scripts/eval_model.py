import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors

from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

import pickle
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def ATLAS_roc(y_true, y_pred):
    sig = (y_true==1)
    bkg = ~sig

    sig_eff = []
    bkg_eff = []

    thresholds = np.linspace(0,0.999,100)

    for threshold in thresholds:
        sig_eff.append(((y_pred[sig] > threshold).sum() / y_true[sig].shape[0]))
        bkg_eff.append(((y_pred[bkg] > threshold).sum()  / y_true[bkg].shape[0]))

    bkg_rej = [1/x for x in bkg_eff]
    return np.array(sig_eff), np.array(bkg_rej), thresholds

def roc(y_true, y_pred):
    sig = (y_true==1)
    bkg = ~sig

    sig_eff = []
    fake_rate = []

    thresholds = np.linspace(0,0.97,100)

    for threshold in thresholds:
        sig_eff.append(((y_pred[sig] > threshold).sum() / y_true[sig].shape[0]))
        fake_rate.append(((y_pred[bkg] > threshold).sum()  / y_true[bkg].shape[0]))

    return np.array(sig_eff), np.array(fake_rate), thresholds

def get_metrics(y_true, y_pred, threshold):
    y_Pred = np.array(y_pred > threshold).astype(int)
    y_True = np.array(y_true > threshold).astype(int)
    x1,y1, thresholds1 = ATLAS_roc(y_True, y_pred)
    x2,y2, thresholds2 = roc(y_True, y_pred)
    AUC = roc_auc_score(y_True, y_Pred)
    BA = accuracy_score(y_True, y_Pred)
    f1 = f1_score(y_True, y_Pred)
    return x1,y1,x2,y2,thresholds1,thresholds2,AUC,BA,f1

def get_predictions(model, data, loss_fns, device, out_path):
    
    X_test, y_test = data
    jet_loss_fn, trk_loss_fn = loss_fns

    ### Evaluate Model

    model.eval()
    model.to(device)

    cumulative_loss_test = 0
    cumulative_MSE_test = 0
    cumulative_BCE_test = 0

    Efrac_pred_labels = []
    Efrac_true_labels = []

    Mfrac_pred_labels = []
    Mfrac_true_labels = []

    trk_pred_labels = []
    trk_true_labels = []

    num_test = len(X_test)
    for i in range(num_test):
        jet_pred, trk_pred = model(X_test[i][0].to(device), X_test[i][1].to(device), X_test[i][2].to(device))

        jet_loss=jet_loss_fn(jet_pred, y_test[i][0].to(device))
        trk_loss=trk_loss_fn(trk_pred, y_test[i][1].to(device))

        loss = jet_loss+trk_loss

        cumulative_loss_test+=loss.detach().cpu().numpy().mean()

        for j in range(jet_pred.shape[0]):
            Efrac_pred_labels.append(float(jet_pred[j][0].detach().cpu().numpy()))
            Efrac_true_labels.append(float(y_test[i][0][j][0].detach().numpy()))
            Mfrac_pred_labels.append(float(jet_pred[j][1].detach().cpu().numpy()))
            Mfrac_true_labels.append(float(y_test[i][0][j][1].detach().numpy()))

        for j in range(trk_pred.shape[0]):
            trk_pred_labels.append(float(trk_pred[j][0].detach().cpu().numpy()))
            trk_true_labels.append(float(y_test[i][1][j][0].detach().numpy()))

    cumulative_loss_test = cumulative_loss_test / num_test

    print()
    print("Test Loss:\t", cumulative_loss_test)
    print()
    print("Efrac R2:\t", r2_score(Efrac_true_labels, Efrac_pred_labels))
    print("Efrac MAE:\t", mean_absolute_error(Efrac_true_labels, Efrac_pred_labels))
    print("Efrac RMSE:\t", root_mean_squared_error(Efrac_true_labels, Efrac_pred_labels))
    print()
    print("Mfrac R2:\t", r2_score(Mfrac_true_labels, Mfrac_pred_labels))
    print("Mfrac MAE:\t", mean_absolute_error(Mfrac_true_labels, Mfrac_pred_labels))
    print("Mfrac RMSE:\t", root_mean_squared_error(Mfrac_true_labels, Mfrac_pred_labels))
    print()

    plt.figure()
    plt.hist(Efrac_true_labels,histtype='step',color='r',label='True Efrac Distribution',bins=50,range=(0,1))
    plt.hist(Efrac_pred_labels,histtype='step',color='b',label='Predicted Efrac Distribution',bins=50,range=(0,1))
    plt.title("Predicted Efrac Distribution using Attention Model (\u03BC=60)")
    plt.legend()
    plt.yscale('log')
    plt.xlabel('Efrac',loc='right')
    plt.savefig(out_path+"/Efrac_1d.png")
    #plt.show()

    plt.figure()
    plt.title("Efrac Distribution using Attention Model (\u03BC=60)")
    plt.hist2d(Efrac_pred_labels,Efrac_true_labels, bins=100,norm=mcolors.PowerNorm(0.2))
    plt.xlabel('Predicted Efrac',loc='right')
    plt.ylabel('True Efrac',loc='top')
    plt.savefig(out_path+"/Efrac_2d.png")
    #plt.show()

    plt.figure()
    plt.hist(Mfrac_true_labels,histtype='step',color='r',label='True Mfrac Distribution',bins=50,range=(0,1))
    plt.hist(Mfrac_pred_labels,histtype='step',color='b',label='Predicted Mfrac Distribution',bins=50,range=(0,1))
    plt.title("Predicted Mfrac Distribution using Attention Model (\u03BC=60)")
    plt.legend()
    plt.yscale('log')
    plt.xlabel('Mfrac',loc='right')
    plt.savefig(out_path+"/Mfrac_1d.png")
    #plt.show()

    plt.figure()
    plt.title("Mfrac Distribution using Attention Model (\u03BC=60)")
    plt.hist2d(Mfrac_pred_labels,Mfrac_true_labels, bins=100,norm=mcolors.PowerNorm(0.2))
    plt.xlabel('Predicted Mfrac',loc='right')
    plt.ylabel('True Mfrac',loc='top')
    plt.savefig(out_path+"/Mfrac_2d.png")
    #plt.show()

    trk_true_labels = np.array(trk_true_labels)
    trk_pred_labels = np.array(trk_pred_labels)

    sig = trk_true_labels==0
    bkg = ~sig

    plt.figure()
    plt.hist(trk_pred_labels[sig],histtype='step',color='r',label='HS Prediction',bins=50,range=(0,1))
    plt.hist(trk_pred_labels[bkg],histtype='step',color='b',label='PU Prediction',bins=50,range=(0,1))
    plt.title("Predicted Track Distribution using Attention Model (\u03BC=60)")
    plt.legend()
    plt.yscale('log')
    plt.xlabel('isPU',loc='right')
    plt.savefig(out_path+"/Trk_1d.png")
    #plt.show()
    
    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(16,9), gridspec_kw={'height_ratios': [3, 1]})

    x1,y1,x1_v2,y1_v2,th1,th1_v2,AUC1,BA1,f11 = get_metrics(np.array(trk_true_labels), np.array(trk_pred_labels), 0.5)

    ax1.set_title("Track isPU ATLAS ROC Curve")
    ax1.set_xlabel("sig eff",loc='right')
    ax1.set_ylabel("bkg rej")

    ax1.plot(x1,y1, label="Attention",color='m')
    AUC1 = "Attention Model AUC: " + str(round(AUC1,4))
    ax1.text(0.51,8,AUC1)

    x = 1-np.flip(th1)
    ratio1 = np.interp(x,np.flip(x1),np.flip(y1))/np.interp(x,np.flip(x1),np.flip(y1))
    ax2.plot(x,ratio1,linestyle='--',color='m')

    # General Plot Settings
    ax1.legend()
    ax1.set_yscale('log')
    ax1.grid(which='both')
    ax2.grid(which='both')
    ax1.set_xlim(0.5,1)
    ax2.set_xlim(0.5,1)
    plt.savefig(out_path+"/Trk_ATLAS_ROC.png")
    #plt.show()

    print("Binary Accuracy: ", BA1, "\tF1 Score: ", f11)
    print()
    print("Done!")