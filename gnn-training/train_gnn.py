import pickle
import torch
import random
import argparse
from torch.optim import Adam
from risk.replay_buffer import ReplayBuffer
from risk.nn import Model15, Model12  # Include both models for comparison
from torch_geometric.data import Batch

def create_batches(replay_buffer, batch_size):
    random.shuffle(replay_buffer.buffer)
    for i in range(0, len(replay_buffer.buffer), batch_size):
        yield replay_buffer.buffer[i:i + batch_size]

def collate_batch(batch):
    state_data, move_probs, win_values = zip(*batch)
    
    # Using `Batch` from torch_geometric to batch data objects
    state_batch = Batch.from_data_list(state_data)
    
    target_policy = torch.stack(move_probs)
    target_value = torch.stack(win_values)
    
    return state_batch, target_policy, target_value

# Function to train the policy and value network
def train_policy_value_network(network, replay_buffers, epochs=30, batch_size=20, learning_rate=0.001):
    optimizer = Adam(network.parameters(), lr=learning_rate)
    criterion_policy = torch.nn.CrossEntropyLoss()
    criterion_value = torch.nn.MSELoss()

    for epoch in range(epochs):
        total_policy_loss = 0.0
        total_value_loss = 0.0

        for replay_buffer in replay_buffers:
            
            for batch in create_batches(replay_buffer, batch_size):
                state_batch, target_policy, target_value = collate_batch(batch)
                
                # Forward pass with batched data
                predicted_value, predicted_policy = network(state_batch)

                # Flatten the predicted value to match the target value shape
                predicted_value = predicted_value.view(-1)
                # Convert target_policy to appropriate form (index of target class)
                target_policy = target_policy.argmax(dim=1)

                # Calculate losses
                policy_loss = criterion_policy(predicted_policy, target_policy)
                value_loss = criterion_value(predicted_value, target_value)

                # Zero the gradients
                optimizer.zero_grad()

                # Perform a backward pass on the accumulated loss
                total_loss = policy_loss + value_loss
                total_loss.backward()

                # Update the parameters once per batch
                optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
        
        avg_policy_loss = total_policy_loss / len(replay_buffer.buffer)
        avg_value_loss = total_value_loss / len(replay_buffer.buffer)
        
        print(f"Epoch {epoch+1}/{epochs}, Policy Loss: {avg_policy_loss}, Value Loss: {avg_value_loss}")

    print("Training complete.")
    return network

def main(args):
    # Load the replay buffer
    replay_buffer = ReplayBuffer()
    replay_buffer.load(args.replay_buffer_path)
    print('Starting training with replay buffer of size:', len(replay_buffer.buffer))

    # Initialize and load the model
    network = Model15()  # Using the batched model
    with open(args.model_path, 'rb') as f:
        network.load_state_dict(pickle.load(f))

    # Train the network
    trained_network = train_policy_value_network(network, [replay_buffer], epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate)

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
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size for training")
    args = parser.parse_args()
    main(args)