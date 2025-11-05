import numpy as np
from scipy.special import softmax as scipy_softmax
from typing import List, Optional

# ============================================================================
# CHOICE MODEL (v3.2)
# ============================================================================

class ChoiceModel:
    """
    Implements discrete choice model for product selection.
    Based on multinomial logit with utility maximization.
    """
    
    @staticmethod
    def compute_choice_probabilities(utilities: np.ndarray) -> np.ndarray:
        """
        Compute choice probabilities using softmax.
        Input: utilities for all products
        Output: probability distribution over products
        """
        # Subtract max for numerical stability
        exp_utilities = np.exp(utilities - np.max(utilities))
        return exp_utilities / np.sum(exp_utilities)
    
    @staticmethod
    def sample_choices(probabilities: np.ndarray, n_choices: int) -> np.ndarray:
        """
        Sample n_choices products without replacement.
        Returns array of product indices.
        """
        if n_choices <= 0:
            return np.array([], dtype=int)
        
        n_products = len(probabilities)
        choices = []
        available_probs = probabilities.copy()
        
        for _ in range(min(n_choices, n_products)):
            if np.sum(available_probs) == 0:
                break
            
            # Normalize
            available_probs = available_probs / np.sum(available_probs)
            
            # Sample
            choice = np.random.choice(n_products, p=available_probs)
            choices.append(choice)
            
            # Remove chosen product
            available_probs[choice] = 0
        
        return np.array(choices, dtype=int)
