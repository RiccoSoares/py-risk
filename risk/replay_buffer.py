import pickle

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

    def collect_training_data(self, turns_data):
        for turn in turns_data:
            state = (turn['owner'], turn['armies'])

            move_probs = turn['move_probs']
            win_values = turn['win_value']

            # Loop through each player's data and store it in the replay buffer
            for player_idx in range(len(win_values)):
                policy = move_probs[player_idx]
                value = float(win_values[player_idx])
                experience = (state, policy, value)
                self.add(experience)