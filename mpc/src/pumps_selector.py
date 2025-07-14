

def update_trigger(current_index, trigger, n_minutes):
    """Update the pattern based on the current index.
    Change the pattern only if current_index is divisible by 60."""
    if current_index == 0:
        return trigger
    else: 
        # Define the cyclic complement sequence
        complements = [
            [1, 0, 1],  # To go from [0, 0, 1] -> [1, 0, 0]
            [1, 1, 0],  # To go from [1, 0, 0] -> [0, 1, 0]
            [0, 1, 1]   # To go from [0, 1, 0] -> [0, 0, 1]
        ]
        
        # Check if the index is divisible by 60
        if current_index % n_minutes == 0:
            # Calculate which complement to use
            complement_index = ((current_index // n_minutes) - 1) % 3
            # Update the trigger pattern by XORing with the complement
            new_pattern = xor_arrays(trigger, complements[complement_index])
            return new_pattern
        else:
            # If not divisible by 60, return the original trigger pattern
            return trigger

def xor_arrays(arr1, arr2):
    """Performs element-wise XOR between two arrays."""
    return [a ^ b for a, b in zip(arr1, arr2)]


def circular_right_shift(sequence):
    return [sequence[-1]] + sequence[:-1]