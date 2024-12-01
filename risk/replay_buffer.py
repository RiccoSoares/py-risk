import pickle
from .game_types import MapState
from .orders import DeployOrder, AttackTransferOrder
import random
from .data_loader import *

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
    
    def convert_to_data(self, state, mapstruct, moves, player, opponent):
        x1, x2, edges = state.to_tensor(player, opponent)
        graph_features, _, _ = state.to_tensor(player, opponent, full=False)
        i1, i2 = state.income(player), state.income(opponent)
        assert torch_geometric.utils.is_undirected(edges)

        mask, nodes, values, b_edges, b_mapping = mapstruct.bonusTensorAlt()

        z = torch.zeros(values.size(), dtype=torch.long)
        z.index_add_(0, mask, torch.ones(mask.size(), dtype=torch.long))

        orders = moves
        order_data = build_order_data(orders, state, x1)
        return StateData(
            map=mapstruct.id,
            num_nodes=len(mapstruct),
            num_bonuses=len(values),
            num_moves=len(orders),
            graph_data=x1,
            global_data=x2,
            graph_features=graph_features,
            graph_edges=edges,
            bonus_edges=b_edges,
            bonus_batch=mask,
            bonus_nodes=nodes,
            bonus_values=values,
            bonus_values_normed=values / z,
            bonus_mapping=b_mapping,
            income=torch.tensor([i1, i2]).view(1, -1),
            total_armies=graph_features[:,2:].sum(dim=0).view(1,-1),
            edge_index=edges,
            **order_data,
        )
        

    def collect_training_data(self, turns_data, mapstruct, p1, p2):
        for turn in turns_data:
            for idx in range(len(turn['moves'])):
                #Collect experience from each player's perspective
                player = idx + 1
                opponent = 1 if player == 2 else 2

                state = MapState(turn['armies'], turn['owner'], mapstruct)
                raw_moves = turn['moves'][idx]
                moves = self.convert_moves(raw_moves)
                data = self.convert_to_data(state, mapstruct, moves, player, opponent)

                move_probs = turn['move_probs'][idx]
                win_values = turn['win_value'][idx]

                experience = (data, move_probs, win_values)
                self.add(experience)