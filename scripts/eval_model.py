import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def get_predictions_DAE(model, data_test, loss_fn, device, out_path, num_epsilon_steps=400):
    print("Starting evaluation...")
    model.eval()
    model.to(device)

    X_test_list, y_test_list = data_test

    all_pred_signal_tracks_list = []
    all_true_signal_tracks_list = []
    all_pred_pileup_tracks_list = []
    # True pileup tracks are all zeros, so we don't need to store them explicitly, just their predictions

    l2_norms_pred_for_true_signal = []
    l2_norms_pred_for_true_pileup = []

    sum_of_event_mean_losses = 0.0
    num_events_processed = 0

    with torch.no_grad():
        for i in tqdm(range(len(X_test_list)), desc="Evaluating Events"):
            X_event = X_test_list[i].to(device)
            y_event = y_test_list[i].to(device)

            pred_tracks = model(X_event)
            
            # Calculate mean loss for the current event
            event_mean_loss = loss_fn(pred_tracks, y_event)
            
            sum_of_event_mean_losses += event_mean_loss.item()
            num_events_processed += 1

            pred_tracks_cpu = pred_tracks.cpu()
            y_event_cpu = y_event.cpu()

            for trk_idx in range(y_event_cpu.size(0)):
                true_track = y_event_cpu[trk_idx]
                pred_track = pred_tracks_cpu[trk_idx]
                
                # L2 norm of the predicted track
                pred_norm = torch.norm(pred_track, p=2).item()
                # A track is pileup if all its true features are zero
                is_true_pileup = torch.all(true_track == 0).item()

                if is_true_pileup:
                    all_pred_pileup_tracks_list.append(pred_track)
                    l2_norms_pred_for_true_pileup.append(pred_norm)
                else:
                    all_pred_signal_tracks_list.append(pred_track)
                    all_true_signal_tracks_list.append(true_track)
                    l2_norms_pred_for_true_signal.append(pred_norm)
    
    avg_of_event_mean_losses = sum_of_event_mean_losses / num_events_processed if num_events_processed > 0 else 0
    print(f"\nAverage of Per-Event Mean Test MSE Loss: {avg_of_event_mean_losses:.6f}")
    
    # Initialize metrics for per-track evaluation
    mse_signal, mae_signal, mse_pileup, mae_pileup = np.nan, np.nan, np.nan, np.nan

    # --- Convert lists to tensors for metric calculations ---
    if all_pred_signal_tracks_list:
        all_pred_signal_t = torch.stack(all_pred_signal_tracks_list)
        all_true_signal_t = torch.stack(all_true_signal_tracks_list)
        
        mse_signal = F.mse_loss(all_pred_signal_t, all_true_signal_t).item()
        mae_signal = F.l1_loss(all_pred_signal_t, all_true_signal_t).item()
        print("\nSignal Tracks Reconstruction (Per-Track Metrics):")
        print(f"  MSE: {mse_signal:.6f}")
        print(f"  MAE: {mae_signal:.6f}")
    else:
        print("\nNo signal tracks found in the test set for reconstruction evaluation.")

    if all_pred_pileup_tracks_list:
        all_pred_pileup_t = torch.stack(all_pred_pileup_tracks_list)
        zeros_for_pileup = torch.zeros_like(all_pred_pileup_t)
        
        mse_pileup = F.mse_loss(all_pred_pileup_t, zeros_for_pileup).item()
        mae_pileup = F.l1_loss(all_pred_pileup_t, zeros_for_pileup).item()
        print("\nPileup Tracks Suppression (Per-Track Metrics, distance from zero vector):")
        print(f"  MSE: {mse_pileup:.6f}")
        print(f"  MAE: {mae_pileup:.6f}")
    else:
        print("\nNo pileup tracks found in the test set for suppression evaluation.")

     # --- ROC Curve Data Preparation ---
    if not l2_norms_pred_for_true_signal and not l2_norms_pred_for_true_pileup:
        print("\nNo tracks available to generate ROC curve or L2 norm histograms.")
        return

    # True labels for ROC: 1 for signal, 0 for pileup
    y_true_roc = np.array([1] * len(l2_norms_pred_for_true_signal) + [0] * len(l2_norms_pred_for_true_pileup))
    # Predicted scores for ROC: L2 norm of the predicted track
    y_score_roc = np.array(l2_norms_pred_for_true_signal + l2_norms_pred_for_true_pileup)

    roc_signal_preservation_rates = []
    roc_pileup_false_positive_rates = []
    roc_points_data = []
    
     # Determine a sensible range for epsilon thresholds based on norms
    min_norm = 0
    max_norm = np.max(y_score_roc) if y_score_roc.size > 0 else 1.0 
    if max_norm == 0: max_norm = 1.0  # handle case where all norms are 0
    
    epsilon_thresholds = np.linspace(min_norm, max_norm, num_epsilon_steps)
    if num_epsilon_steps == 1 and max_norm > min_norm : # Ensure at least two points if range allows for linspace
         epsilon_thresholds = np.array([min_norm,max_norm])
    elif num_epsilon_steps == 1 and max_norm == min_norm: # handles edge case where all norms are same
         epsilon_thresholds = np.array([min_norm])

    print("\nCalculating ROC points...")
    for epsilon in tqdm(epsilon_thresholds, desc="Calculating ROC points", leave=False):
        predicted_as_signal = (y_score_roc >= epsilon)
        
        tp = np.sum((predicted_as_signal == 1) & (y_true_roc == 1))
        fn = np.sum((predicted_as_signal == 0) & (y_true_roc == 1))
        tn = np.sum((predicted_as_signal == 0) & (y_true_roc == 0))
        fp = np.sum((predicted_as_signal == 1) & (y_true_roc == 0))

        signal_preservation_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        pileup_false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0 # This is 1 - Pileup Suppression Rate

        roc_signal_preservation_rates.append(signal_preservation_rate)
        roc_pileup_false_positive_rates.append(pileup_false_positive_rate)
        roc_points_data.append({'epsilon': epsilon, 'tpr': signal_preservation_rate, 'fpr': pileup_false_positive_rate})
    
    
    # --- Print specific rates for a chosen epsilon threshold --- # i need to decide on a sensible epsilon threshold: maybe median of the pileup L2 norms?
    chosen_epsilon = 1.0 
    predicted_as_signal_chosen_eps = (y_score_roc >= chosen_epsilon)
    tp_chosen = np.sum((predicted_as_signal_chosen_eps == 1) & (y_true_roc == 1))
    fn_chosen = np.sum((predicted_as_signal_chosen_eps == 0) & (y_true_roc == 1))
    tn_chosen = np.sum((predicted_as_signal_chosen_eps == 0) & (y_true_roc == 0))
    fp_chosen = np.sum((predicted_as_signal_chosen_eps == 1) & (y_true_roc == 0))

    tpr_chosen = tp_chosen / (tp_chosen + fn_chosen) if (tp_chosen + fn_chosen) > 0 else 0.0 # this is signal preservation rate or (TPR)
    fnr_chosen = fn_chosen / (tp_chosen + fn_chosen) if (tp_chosen + fn_chosen) > 0 else 0.0 # this is signal false negative rate or (FNR = 1 - TPR)
    tnr_chosen = tn_chosen / (tn_chosen + fp_chosen) if (tn_chosen + fp_chosen) > 0 else 0.0 # this is pileup suppression rate or (TNR)
    fpr_chosen = fp_chosen / (fp_chosen + tn_chosen) if (fp_chosen + tn_chosen) > 0 else 0.0 # this is pileup false positive rate or (FPR = 1 - TNR)
    


    print(f"\nMetrics at Epsilon = {chosen_epsilon:.3f} (threshold on L2 norm to be considered signal):")
    print(f"  Signal Preservation Rate (TPR): {tpr_chosen:.4f}") # Out of all true signal tracks, how many were correctly predicted as signal
    print(f"  Signal False Negative Rate (FNR = 1 - TPR): {fnr_chosen:.4f}") # Out of all true signal tracks, how many were incorrectly predicted as pileup
    print(f"  Pileup Suppression Rate (TNR):  {tnr_chosen:.4f}") # Out of all true pileup tracks, how many were correctly predicted as pileup
    print(f"  Pileup False Positive Rate (FPR = 1 - TNR): {fpr_chosen:.4f}")  # Out of all true pileup tracks, how many were incorrectly predicted as signal
    
    
	# # Print all ROC data points
    # print("\n--- ROC Data Points ---")
    # print("Epsilon   | TPR     | FPR")
    # print("--------------------------")
    # for point in roc_points_data:
    #     print(f"{point['epsilon']:<9.4f} | {point['tpr']:<7.4f} | {point['fpr']:<7.4f}")
    
    # Save the printed metrics to a text file
    metrics_save_path = os.path.join(out_path, "DAE_metrics.txt")
    with open(metrics_save_path, 'w') as f:
        f.write(f"Average of Per-Event Mean Test MSE Loss: {avg_of_event_mean_losses:.6f}\n")
        f.write("\nSignal Tracks Reconstruction (Per-Track Metrics):\n")
        f.write(f"  MSE: {mse_signal if not np.isnan(mse_signal) else 'N/A':.6f}\n")
        f.write(f"  MAE: {mae_signal if not np.isnan(mae_signal) else 'N/A':.6f}\n")
        f.write("\nPileup Tracks Suppression (Per-Track Metrics, distance from zero vector):\n")
        f.write(f"  MSE: {mse_pileup if not np.isnan(mse_pileup) else 'N/A':.6f}\n")
        f.write(f"  MAE: {mae_pileup if not np.isnan(mae_pileup) else 'N/A':.6f}\n")
        f.write(f"\nMetrics at Epsilon = {chosen_epsilon:.3f} (threshold on L2 norm to be considered signal):\n")
        f.write(f"  Signal Preservation Rate (TPR): {tpr_chosen:.4f}\n")
        f.write(f"  Pileup Suppression Rate (TNR):  {tnr_chosen:.4f}\n")
        f.write("\n\n--- ROC Data Points ---\n")
        f.write("Epsilon   | TPR     | FPR\n")
        f.write("---------------------------\n")
        for point in roc_points_data:
            f.write(f"{point['epsilon']:<9.4f} | {point['tpr']:<7.4f} | {point['fpr']:<7.4f}\n")
            # Only save until epsilon is 50
            if point['epsilon'] > 50.0:
                break
    print(f"\nMetrics saved to {metrics_save_path}")

    
    # --- Plotting ---
    os.makedirs(out_path, exist_ok=True)

     # ROC Curve
    plt.figure(figsize=(10, 8))
    plt.plot(roc_pileup_false_positive_rates, roc_signal_preservation_rates, marker='.', label='Model ROC')
    plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Random Guess')
    plt.xlabel("Pileup False Positive Rate (1 - Pileup Suppression Rate)")
    plt.ylabel("Signal Preservation Rate (TPR)")
    plt.title("ROC Curve: Denoising Autoencoder")
    plt.legend()
    plt.grid(True)
    roc_save_path = os.path.join(out_path, "DAE_ROC_curve.png")
    plt.savefig(roc_save_path)
    plt.close()
    print(f"ROC curve saved to {roc_save_path}")

    
    # L2 Norm Histograms
    plt.figure(figsize=(10, 6))
    if l2_norms_pred_for_true_signal:
        # plt.hist(l2_norms_pred_for_true_signal, bins=50, alpha=0.7, label='True Signal Tracks (Predicted L2 Norms)', density=True)
        plt.hist(l2_norms_pred_for_true_signal, bins=50, alpha=0.7, label='True Signal Tracks (Predicted L2 Norms)', density=False)
    if l2_norms_pred_for_true_pileup:
        # plt.hist(l2_norms_pred_for_true_signal, bins=50, alpha=0.7, label='True Signal Tracks (Predicted L2 Norms)', density=True)
        plt.hist(l2_norms_pred_for_true_pileup, bins=50, alpha=0.7, label='True Pileup Tracks (Predicted L2 Norms)', density=False)
    plt.xlabel("L2 Norm of Predicted Track Vector")
    # plt.ylabel("Density")
    plt.ylabel("Counts (Log Scale)")
    plt.title("Distribution of Predicted L2 Norms")
    plt.yscale('log') 
    plt.legend()
    plt.grid(True)
    hist_save_path = os.path.join(out_path, "DAE_L2_norm_histograms.png")
    plt.savefig(hist_save_path)
    plt.close()
    print(f"L2 norm histograms saved to {hist_save_path}")

    print("\nEvaluation finished.")