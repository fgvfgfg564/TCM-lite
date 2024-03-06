import numpy as np


def choose_items_with_probabilities(elements, probabilities, k):
    # Check if probabilities sum up to 1
    if not np.isclose(sum(probabilities), 1):
        raise ValueError("Probabilities should sum up to 1.")

    # Choose k items based on probabilities
    chosen_items = np.random.choice(elements, size=k, p=probabilities, replace=False)
    return chosen_items


# Example usage:
elements = [1, 2, 3, 4, 5]
probabilities = [0.1, 0.2, 0.3, 0.2, 0.2]  # Probabilities corresponding to elements
k = 6

chosen_items = choose_items_with_probabilities(elements, probabilities, k)
print("Chosen items:", chosen_items)
