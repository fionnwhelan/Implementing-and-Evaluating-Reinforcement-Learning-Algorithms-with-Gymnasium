import pickle

# import torch

# import torch.nn as nn

# # Define a simple neural network
# class SimpleNet(nn.Module):
#     def __init__(self):
#         super(SimpleNet, self).__init__()
#         # Input features -> 8 hidden units -> 4 output units
#         self.fc1 = nn.Linear(5, 8)
#         self.fc2 = nn.Linear(8, 4)
#         self.relu = nn.ReLU()
        
#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# # Create the network with random weights
# net = SimpleNet()

# # Generate a random input tensor
# input_tensor = torch.rand(5)  # 5 input features
# print(f"Input: {input_tensor}")

# # Forward pass through the network
# with torch.no_grad():
#     output = net(input_tensor)
#     print(f"Output: {output}")
    
#     # Get the index of the maximum value
#     max_index = torch.argmax(output).item()
#     print(f"Argmax: {max_index}")
#     print(f'Argmax type: {type(max_index)}')


# Path to the pickle file
file_path = 'EX4_outputs/DDPG-Racetrack-sweep-results-ex4.pkl'

# Load the pickle file
try:
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    print(f"Successfully loaded data from {file_path}")
    print(f"Type of loaded data: {type(data)}")
    # Print basic information about the data
    print(f"Data summary: {data}")  # Print the full data object
    print(f"Data length: {len(data)} runs")
    
    for i in range(len(data)):
        run = data[i]
        value = getattr(run, 'config')
        print(value)
        print('Final return mean:', run.final_return_mean)
    # Inspect the first run object if data is not empty
#     if data and len(data) > 0:
#         first_run = data[0]

#         print(f"\nFirst run type: {type(first_run)}")
        
#         # Check for key attributes
#         attrs_to_check = ['config', 'metrics', 'results', 'summary', 'name', 'id']
#         for attr in attrs_to_check:
#             if hasattr(first_run, attr):
#                 value = getattr(first_run, attr)
#                 print(f"\nFirst run {attr}:")
#                 print(f"  Type: {type(value)}")
#                 print(f"  Value: {value}")
        
#         # If we have metrics data, analyze it
#         if hasattr(first_run, 'metrics') and first_run.metrics:
#             print("\nMetrics keys:", list(first_run.metrics.keys()))
            
#             # Show the first few values of a metric if available
#             for key in list(first_run.metrics.keys())[:2]:  # First 2 metrics
#                 values = first_run.metrics[key]
#                 print(f"\nMetric '{key}':")
#                 print(f"  Type: {type(values)}")
#                 print(f"  Length: {len(values) if hasattr(values, '__len__') else 'N/A'}")
#                 print(f"  First few values: {values[:5] if hasattr(values, '__getitem__') else values}")
#         print('Final return mean:', first_run.final_return_mean)
except FileNotFoundError:
    print(f"File not found: {file_path}")
except Exception as e:
    print(f"Error loading file: {str(e)}")