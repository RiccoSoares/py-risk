import torch
import pickle
from risk.nn import Model5

# Initialize the model
model = Model5()

# Save the initial model using pickle
with open('model-weights/initial_policy_value_model.pkl', 'wb') as f:
    pickle.dump(model.state_dict(), f)