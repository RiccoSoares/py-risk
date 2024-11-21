# Training a GNN

The content in this dir is dedicated to scripts for training a policy and value GNN through self-play of an agent that plays with MCTS guided by this GNN, in a Reinforcement Learning framework. 


## Model initialization

To save random initialized weights run:
```sh
PYTHONPATH=$(pwd) python gnn-training/initialize_model.py
```
The weights will be saved in model-weights and correspond to the nn.Model12 policy and value architecture.


## Collecting experiences from self-play

Running the following will generate self play data, that will be saved in a Replay Buffer. It saves the board state, policy (moves probability distribution), and reward (evaluation of the position according to the own agent).
```sh
PYTHONPATH=$(pwd) python gnn-training/self-play.py \
    --iter 300 \
    --model-type MCTS \
    --output-dir results \
    --buffer-capacity 10000 \
    --num-games 100 \
    --model-path model-weights/model12_initial_weights.pkl \
```
Map played is currently Italy.

## Training the Neural Network based on the collected experiences

To load the ReplayBuffer data and train the GNN:
```sh
PYTHONPATH=$(pwd) python gnn-training/train_gnn.py \
    --replay_buffer_path results/replay_buffer_model12.pkl \
    --model_path model-weights/model12_initial_weights.pkl \
    --trained_model_path model-weights/model12_trained_weights.pkl \
    --epochs 50 \
    --learning_rate 0.001
```

## Evaluation

We can evaluate the training in matches of the trained agent against the untrained.
```sh
PYTHONPATH=$(pwd) python gnn-training/eval.py \
    --iter-1 150 \
    --iter-2 150 \
    --model-type-1 MCTS \
    --model-type-2 MCTS \
    --output-dir gnn-training/eval \
    --num-games 100 \
    --model-1-path model-weights/model12_initial_weights.pkl \
    --model-2-path model-weights/model12_trained_weights.pkl
```
