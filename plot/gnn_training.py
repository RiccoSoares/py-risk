import re
import matplotlib.pyplot as plt

# Initialize lists to store the extracted data
epochs = []
policy_loss = []
value_loss = []

# Define a regular expression to match the log lines
pattern = re.compile(r"Epoch (\d+)/200, Policy Loss: ([0-9.]+), Value Loss: ([0-9.]+)")

# Read the log file and parse the data
with open('train_small_maps_output.log', 'r') as file:
    for line in file:
        match = pattern.match(line)
        if match:
            epochs.append(int(match.group(1)))
            policy_loss.append(float(match.group(2)))
            value_loss.append(float(match.group(3)))

# Plot Policy Loss
plt.figure(figsize=(12, 6))
plt.plot(epochs, policy_loss, label='Policy Loss')
plt.xlabel('Epoch')
plt.ylabel('Policy Loss')
plt.title('Policy Loss over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('plot/policy_loss_plot.png')

# Plot Value Loss
plt.figure(figsize=(12, 6))
plt.plot(epochs, value_loss, label='Value Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Value Loss')
plt.title('Value Loss over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('plot/value_loss_plot.png')

