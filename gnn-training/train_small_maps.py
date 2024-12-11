import pickle
import torch
import random
import argparse
from torch.optim import Adam
from risk.replay_buffer import ReplayBuffer
from risk.nn import Model15, Model12  # Include both models for comparison
from torch_geometric.data import Batch
from train_gnn import *

def main(args):
    # Load the replay buffer
    owl_exps = ReplayBuffer()
    owl_exps.load('replay-buffer/mcts300/owl_island.pkl')
    print('Loaded owl_island replay buffer of size:', len(owl_exps.buffer))

    simple_exps = ReplayBuffer()
    simple_exps.load('replay-buffer/mcts300/simple.pkl')
    print('Loaded simple replay buffer of size:', len(simple_exps.buffer))

    banana_exps = ReplayBuffer()
    banana_exps.load('replay-buffer/mcts300/banana.pkl')
    print('Loaded banana replay buffer of size:', len(banana_exps.buffer))

    replay_buffers = [owl_exps, simple_exps, banana_exps]

    # Initialize and load the model
    network = Model15()  
    with open(args.model_path, 'rb') as f:
        network.load_state_dict(pickle.load(f))

    # Train the network
    trained_network = train_policy_value_network(network, replay_buffers, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate)

    # Save the trained network
    with open(args.trained_model_path, 'wb') as f:
        pickle.dump(trained_network.state_dict(), f)
    print('Model training complete and saved.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the GNN")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the initial model file")
    parser.add_argument("--trained_model_path", type=str, required=True, help="Path to save the trained model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size for training")
    args = parser.parse_args()
    main(args)