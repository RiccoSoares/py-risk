# Training a GNN

The content in this dir is dedicated to scripts for training a policy and value GNN through self-play of an agent that plays with MCTS guided by this GNN, in a Reinforcement Learning framework. 


## Model initialization

To save random initialized weights run:
```sh
PYTHONPATH=$(pwd) python gnn-training/initialize_model.py
```
The weights will be saved in model-weights and correspond to the nn.Model5 policy and value architecture.


## Collecting experiences from self-play

Running the following will generate self play data, that will be saved in a Replay Buffer. It saves the board state, policy (moves probability distribution), and reward (evaluation of the position according to the own agent).
```sh
PYTHONPATH=$(pwd) python gnn-training/self-play.py \       
    --iter-1 50 \
    --iter-2 50 \
    --model-type-1 MCTS \
    --model-type-2 MCTS \
    --output-dir results \
    --buffer-capacity 10000 \
    --num-games 4 \
    --model-path model-weights/initial_policy_value_model.pkl
```
Map played is currently Italy.

