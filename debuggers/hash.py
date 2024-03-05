import numpy as np
import hashlib

def hash_numpy_array(array, hash_function='sha256'):
    """
    Computes the hash of a NumPy array.

    :param array: NumPy array to be hashed.
    :param hash_function: Name of the hash function to use (e.g., 'md5', 'sha1', 'sha256').
    :return: Hexadecimal hash of the array.
    """
    if hash_function not in hashlib.algorithms_available:
        raise ValueError(f"Hash function {hash_function} is not available.")

    # Convert the array to bytes
    array_bytes = array.tobytes()

    # Create a hash object and update it with the array bytes
    hash_obj = hashlib.new(hash_function)
    hash_obj.update(array_bytes)

    # Return the hexadecimal digest of the hash
    return hash_obj.hexdigest()

# Example usage:
arr = np.array([1, 2, 3])
print(hash_numpy_array(arr, 'sha256'))
