import matplotlib.pyplot as plt
import pickle

results_path = "../results/tgn-attn-gcn.pkl"  # Replace with the actual path
results = pickle.load(open(results_path, "rb"))

epoch_times = results["epoch_times"]
total_epoch_times = results["total_epoch_times"]
mean_losses = results["train_losses"]
val_aps = results["val_aps"]
nn_val_aps = results["new_nodes_val_aps"]

# Plotting
epochs = range(1, len(epoch_times) + 1)

# Plot total epoch times
plt.figure(figsize=(10, 6))
plt.plot(epochs, total_epoch_times, label='Total Epoch Time')
plt.xlabel('Epoch')
plt.ylabel('Time (s)')
plt.title('Total Epoch Time over Epochs')
plt.legend()
plt.show()

# Plot mean losses
plt.figure(figsize=(10, 6))
plt.plot(epochs, mean_losses, label='Epoch Mean Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Epoch Mean Loss over Epochs')
plt.legend()
plt.show()

# Plot validation APs
plt.figure(figsize=(10, 6))
plt.plot(epochs, val_aps, label='Validation AP')
plt.plot(epochs, nn_val_aps, label='New Nodes Validation AP')
plt.xlabel('Epoch')
plt.ylabel('Average Precision')
plt.title('Validation AP and New Nodes Validation AP over Epochs')
plt.legend()
plt.show()