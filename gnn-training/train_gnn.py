import pickle
import torch
import random
import argparse
import torch
from torch.optim import Adam
from risk.replay_buffer import ReplayBuffer
from risk.nn import Model12

# Function to train the policy and value network
def train_policy_value_network(network, replay_buffer, epochs=30, learning_rate=0.001):
    optimizer = Adam(network.parameters(), lr=learning_rate)
    criterion_policy = torch.nn.KLDivLoss(reduction='batchmean')
    criterion_value = torch.nn.MSELoss()

    for epoch in range(epochs):
        random.shuffle(replay_buffer.buffer)  # Shuffle the buffer to ensure random batches

        for i, exp in enumerate(replay_buffer.buffer):
            state, target_policy, target_value = exp
            graph_features, global_features, edges, moves = state

            #Forward pass
            predicted_value, predicted_policy = network(graph_features, global_features, edges, moves)

            #Convert target policy and target value to tensors
            target_policy = torch.tensor(target_policy, dtype=torch.float32, device=predicted_policy.device)
            target_value = torch.tensor(target_value, dtype=torch.float32, device=predicted_value.device)

            #Calculate losses
            policy_loss = criterion_policy(predicted_policy, target_policy)
            value_loss = criterion_value(predicted_value.squeeze(), target_value)
            total_loss = policy_loss + value_loss

            #Zero the gradients
            optimizer.zero_grad()

            #Backward pass (compute gradients)
            total_loss.backward()

            #Update the parameters
            optimizer.step()

            print(f"Epoch {epoch+1}/{epochs}, Step {i+1}/{len(replay_buffer.buffer)}, Policy Loss: {policy_loss.item()}, Value Loss: {value_loss.item()}, Total Loss: {total_loss.item()}")

    print("Training complete.")
    return network

def main(args):
    # Load the replay buffer
    replay_buffer = ReplayBuffer()
    replay_buffer.load(args.replay_buffer_path)

    # Initialize and load the model
    network = Model12()
    with open(args.model_path, 'rb') as f:
        network.load_state_dict(pickle.load(f))

    # Train the network
    trained_network = train_policy_value_network(network, replay_buffer, epochs=args.epochs, learning_rate=args.learning_rate)

    # Save the trained network
    with open(args.trained_model_path, 'wb') as f:
        pickle.dump(trained_network.state_dict(), f)
    print('Model training complete and saved.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the GNN")
    parser.add_argument("--replay_buffer_path", type=str, required=True, help="Path to the replay buffer file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the initial model file")
    parser.add_argument("--trained_model_path", type=str, required=True, help="Path to save the trained model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for training")
    args = parser.parse_args()
    main(args)