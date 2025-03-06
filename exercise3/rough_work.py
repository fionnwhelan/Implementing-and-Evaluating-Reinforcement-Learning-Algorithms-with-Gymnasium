import torch

import torch.nn as nn

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # Input features -> 8 hidden units -> 4 output units
        self.fc1 = nn.Linear(5, 8)
        self.fc2 = nn.Linear(8, 4)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create the network with random weights
net = SimpleNet()

# Generate a random input tensor
input_tensor = torch.rand(5)  # 5 input features
print(f"Input: {input_tensor}")

# Forward pass through the network
with torch.no_grad():
    output = net(input_tensor)
    print(f"Output: {output}")
    
    # Get the index of the maximum value
    max_index = torch.argmax(output).item()
    print(f"Argmax: {max_index}")
    print(f'Argmax type: {type(max_index)}')