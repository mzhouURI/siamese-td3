import torch

# Example tensors with shape [20, 128, 4]
stacked_tensor_1 = torch.randn(20, 128, 4)
stacked_tensor_2 = torch.randn(20, 128, 4)

# Initialize an empty list to hold the concatenated tensors
combined_tensors = []

# Number of iterations (for example, concatenate 2 times)
num_iterations = 2

# Loop through the number of iterations
for _ in range(num_iterations):
    # Concatenate the tensors along the batch dimension (dim=0)
    combined_tensor = torch.cat((stacked_tensor_1.unsqueeze(0), stacked_tensor_2.unsqueeze(0)), dim=0)
    
    # Append the result to the list
    combined_tensors.append(combined_tensor)

# After the loop, you can stack all combined tensors into a single tensor if needed
final_combined_tensor = torch.cat(combined_tensors, dim=0)

print(final_combined_tensor.shape)  # Should print: torch.Size([4, 20, 128, 4])
