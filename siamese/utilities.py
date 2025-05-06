

# Function to generate batches
def get_batches(training_data, batch_size, indices):
    num_samples = len(training_data)
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        
        # Get the corresponding data for the batch
        batch_data = training_data[batch_indices]
        
        # Split into inputs (error_states, states) and targets (actions)
        batch_error_states = batch_data[:, :error_states.shape[1]]
        batch_states = batch_data[:, error_states.shape[1]:error_states.shape[1] + states.shape[1]]
        batch_actions = batch_data[:, -actions.shape[1]:]
        
        yield batch_error_states, batch_states, batch_actions