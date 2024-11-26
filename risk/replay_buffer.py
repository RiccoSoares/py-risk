import pickle
from .game_types import MapState
from .orders import DeployOrder, AttackTransferOrder
import random

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        
    def add(self, experience):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.buffer, f)
    
    def load(self, filename):
        with open(filename, "rb") as f:
            self.buffer = pickle.load(f)
    
    def __len__(self):
        return len(self.buffer)

    def convert_moves(self, raw_moves):
        moves = []
        for move_set in raw_moves:
            converted_set = []
            for move in move_set:
                if move[0] == 'DeployOrder':
                    converted_set.append(DeployOrder(*move[1:]))
                elif move[0] == 'AttackTransferOrder':
                    converted_set.append(AttackTransferOrder(*move[1:]))
                else:
                    raise ValueError(f"Unknown move type: {move[0]}")
            moves.append(converted_set)
        return moves

    def collect_training_data(self, turns_data, mapstruct, p1, p2):
        for turn in turns_data:
            for idx in range(len(turn['moves'])):
                #Collect experience from each player's perspective
                player = idx + 1
                opponent = 1 if player == 2 else 2

                state = MapState(turn['armies'], turn['owner'], mapstruct)
                graph_features, global_features, edges = state.to_tensor(player, opponent)
                raw_moves = turn['moves'][idx]
                moves = self.convert_moves(raw_moves)
                state = (graph_features, global_features, edges, moves)

                move_probs = turn['move_probs'][idx]
                win_values = turn['win_value'][idx]

                experience = (state, move_probs, win_values)
                self.add(experience)