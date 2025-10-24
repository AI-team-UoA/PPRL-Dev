import numpy as np
from typing import List


# Dice coefficient: 2 * |A ∩ B| / (|A| + |B|)

def _math_dice(intersecntion: int, cadinality_a : int, cardinality_b : int) -> float: 
    return float(2 * (intersecntion) / (cadinality_a + cardinality_b))




    
def _dice(blooms_1 : np.array, blooms_2 : np.array) -> np.array: 
    # Vectorized intersection and cardinality computation
    # Intersection: bitwise AND
        
    intersections = np.sum(blooms_1 & blooms_2, axis=2)  # (n_candidates, n_attributes)
        
    # Cardinalities
    card_1 = np.sum(blooms_1, axis=2)  # (n_candidates, n_attributes)
    card_2 = np.sum(blooms_2, axis=2)  # (n_candidates, n_attributes)

    denominators = card_1 + card_2
    # Avoid division by zero
    dice_scores = np.zeros_like(intersections, dtype=np.float64)
    valid_mask = denominators > 0
    dice_scores[valid_mask] = (2 * intersections[valid_mask]) / denominators[valid_mask]
    avg_similarities = np.mean(dice_scores, axis=1)


    return avg_similarities


def _scm(blooms_1 : np.array, blooms_2 : np.array) -> np.array: 
    # Vectorized intersection and cardinality computation
    # Intersection: bitwise AND


    xor_sum = np.bitwise_xor(blooms_1, blooms_2).sum(axis=2)
    matches = blooms_1.shape[2] - xor_sum
    
    
    scm_per_attr = matches / blooms_1.shape[2]
    
    avg_similarities = np.mean(scm_per_attr, axis=1)
    print(avg_similarities)


    return avg_similarities