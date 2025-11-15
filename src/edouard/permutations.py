from itertools import permutations
from math import factorial
import numpy as np


def idx_to_perm(idx, m):
    """
    Maps an index to a canonical permutation of elements 0 to m-2.
    
    Canonical order:
    - First by length (0, 1, 2, ..., m-1)
    - Within each length, lexicographically
    
    Args:
        idx: The index of the permutation
        m: The parameter defining the range (elements are 0 to m-2)
    
    Returns:
        A tuple representing the permutation
    
    Example for m=3:
        idx_to_perm(0, 3) = ()
        idx_to_perm(1, 3) = (0,)
        idx_to_perm(2, 3) = (1,)
        idx_to_perm(3, 3) = (0, 1)
        idx_to_perm(4, 3) = (1, 0)
    """
    elements = list(range(m - 1))
    current_idx = 0
    
    # Generate permutations by length
    for length in range(m):
        # Get all permutations of this length
        perms = sorted(permutations(elements, length))
        
        # Check if our target index is in this length group
        if current_idx + len(perms) > idx:
            # Found the right length group
            perm_idx = idx - current_idx
            return perms[perm_idx]
        
        current_idx += len(perms)
    
    raise IndexError(f"Index {idx} out of range for m={m}")


def perm_to_idx(perm, m):
    """
    Maps a permutation to its canonical index.
    
    Args:
        perm: A tuple representing the permutation
        m: The parameter defining the range (elements are 0 to m-2)
    
    Returns:
        The index of the permutation
    """
    elements = list(range(m - 1))
    current_idx = 0
    length = len(perm)
    
    # Add counts for all shorter lengths
    for l in range(length):
        perms = list(permutations(elements, l))
        current_idx += len(perms)
    
    # Find position within permutations of this length
    perms_of_length = sorted(permutations(elements, length))
    current_idx += perms_of_length.index(perm)
    
    return current_idx


def update_perm_idx(idx, j, m):
    """
    Returns the index that results from appending element j to the permutation at idx.
    
    This is efficient - it works directly with indices without constructing permutations.
    
    Args:
        idx: The index of the original permutation
        j: The element to append (0 <= j < m-1)
        m: The parameter defining the range
    
    Returns:
        The index of the extended permutation
    
    Example:
        update_perm_idx(2, 0, 3) = 4
        because idx_to_perm(2, 3) = (1,), extending to (1, 0), which has index 4
    """
    # First, determine the length and position within length group of the current permutation
    elements = m - 1
    current_idx = 0
    length = 0
    
    # Find which length group idx belongs to
    for l in range(m):
        count = factorial(elements) // factorial(elements - l) if l <= elements else 0
        if current_idx + count > idx:
            length = l
            pos_in_length = idx - current_idx
            break
        current_idx += count
    
    # Now we need to:
    # 1. Figure out which permutation this is within its length group
    # 2. Determine what the extended permutation looks like
    # 3. Find the index of that extended permutation
    
    # The extended permutation will have length = length + 1
    new_length = length + 1
    
    # Count of permutations before the new length group
    idx_before_new_length = 0
    for l in range(new_length):
        count = factorial(elements) // factorial(elements - l) if l <= elements else 0
        idx_before_new_length += count
    
    # Now we need to find the position of the extended permutation within the new length group
    # We need to decode the original permutation to understand its structure
    
    if length == 0:
        # Original is empty (), extended is (j,)
        # Position in new length group is just j (since permutations of length 1 are sorted)
        return idx_before_new_length + j
    
    # For length > 0, we need to decode the permutation
    # Build the original permutation to understand which elements are used and in what order
    available = list(range(elements))
    perm_list = []
    remaining_pos = pos_in_length
    
    for pos in range(length):
        # How many perms can be formed with remaining positions?
        remaining_slots = length - pos - 1
        perms_per_choice = factorial(elements - pos - 1) // factorial(elements - length) if remaining_slots >= 0 else 1
        
        # Which element goes in this position?
        choice_idx = remaining_pos // perms_per_choice
        perm_list.append(available[choice_idx])
        available.pop(choice_idx)
        remaining_pos %= perms_per_choice
    
    # Check if j is already in the permutation
    if j in perm_list:
        raise ValueError(f"Element {j} is already in the permutation")
    
    # Create the extended permutation
    extended_perm = perm_list + [j]
    
    # Find the position of this extended permutation in the new length group
    # We need to find the lexicographic position
    available_new = list(range(elements))
    new_pos = 0
    
    for pos in range(new_length):
        elem = extended_perm[pos]
        elem_idx = available_new.index(elem)
        
        # Add count of all permutations that would come before this choice
        remaining_slots = new_length - pos - 1
        perms_per_choice = factorial(elements - pos - 1) // factorial(elements - new_length) if remaining_slots >= 0 else 1
        new_pos += elem_idx * perms_per_choice
        
        available_new.remove(elem)
    
    return idx_before_new_length + new_pos


def update_perm_idx_vectorized(idx_vec, j_vec, m):
    """
    Vectorized version of update_perm_idx.
    
    Returns indices that result from appending elements j_vec to permutations at idx_vec.
    
    Args:
        idx_vec: Array of indices of original permutations
        j_vec: Array of elements to append (0 <= j < m-1)
        m: The parameter defining the range
    
    Returns:
        Array of indices of extended permutations
    
    Example:
        update_perm_idx_vectorized(np.array([2, 1]), np.array([0, 1]), 3)
        returns np.array([4, 3])
    """
    idx_vec = np.asarray(idx_vec, dtype=int)
    j_vec = np.asarray(j_vec, dtype=int)
    
    elements = m - 1
    n = len(idx_vec)
    result = np.zeros(n, dtype=int)
    
    # Precompute factorial values
    fact = np.array([factorial(i) for i in range(elements + 1)])
    
    # For each index, determine its length group
    lengths = np.zeros(n, dtype=int)
    pos_in_length = np.zeros(n, dtype=int)
    
    for i in range(n):
        current_idx = 0
        for l in range(m):
            count = fact[elements] // fact[elements - l] if l <= elements else 0
            if current_idx + count > idx_vec[i]:
                lengths[i] = l
                pos_in_length[i] = idx_vec[i] - current_idx
                break
            current_idx += count
    
    # Process each element
    for i in range(n):
        length = lengths[i]
        new_length = length + 1
        
        # Count of permutations before the new length group
        idx_before_new_length = 0
        for l in range(new_length):
            count = fact[elements] // fact[elements - l] if l <= elements else 0
            idx_before_new_length += count
        
        if length == 0:
            # Original is empty (), extended is (j,)
            result[i] = idx_before_new_length + j_vec[i]
        else:
            # Decode the original permutation
            available = list(range(elements))
            perm_list = []
            remaining_pos = pos_in_length[i]
            
            for pos in range(length):
                remaining_slots = length - pos - 1
                perms_per_choice = fact[elements - pos - 1] // fact[elements - length]
                
                choice_idx = remaining_pos // perms_per_choice
                perm_list.append(available[choice_idx])
                available.pop(choice_idx)
                remaining_pos %= perms_per_choice
            
            # Check if j is already in the permutation
            if j_vec[i] in perm_list:
                raise ValueError(f"Element {j_vec[i]} is already in the permutation at index {i}")
            
            # Create the extended permutation
            extended_perm = perm_list + [j_vec[i]]
            
            # Find the position of this extended permutation in the new length group
            available_new = list(range(elements))
            new_pos = 0
            
            for pos in range(new_length):
                elem = extended_perm[pos]
                elem_idx = available_new.index(elem)
                
                remaining_slots = new_length - pos - 1
                perms_per_choice = fact[elements - pos - 1] // fact[elements - new_length]
                new_pos += elem_idx * perms_per_choice
                
                available_new.remove(elem)
            
            result[i] = idx_before_new_length + new_pos
    
    return result


class WeightVectorCalculator:
    """
    Optimized weight vector calculator that precomputes the structure of how
    each weight vector depends on keep factors.
    
    For each permutation index, we intern:
    - Which keep factors multiply together for each position
    - The structure of (1 - keep_factor) products
    
    This allows fast recalculation when only the keep_factor values change.
    """
    
    def __init__(self, num_winners, m):
        """
        Args:
            num_winners: Number of winners (L)
            m: Parameter for idx_to_perm (usually L+1)
        """
        self.num_winners = num_winners
        self.m = m
        self.num_perms = sum(factorial(num_winners) // factorial(l) 
                            for l in range(num_winners + 1))
        
        # Precompute the structure for each permutation
        # Each entry: list of (position, winner_indices_for_product, complement_indices)
        self.structures = {}
        
        for perm_idx in range(self.num_perms):
            self.structures[perm_idx] = self._compute_structure(perm_idx)
    
    def _compute_structure(self, perm_idx):
        """
        Compute the structure of how to calculate weights for a given permutation.
        
        Returns a list where each element describes one position in wt_vec:
        - (position, keep_idx, complement_indices)
        
        For position i with winner j:
          wt_vec[j] = keep_factors[j] * product((1 - keep_factors[k]) for k in complement_indices)
        For the final position (leftover):
          wt_vec[-1] = product((1 - keep_factors[k]) for k in complement_indices)
        """
        perm = idx_to_perm(perm_idx, self.m)
        structure = []
        
        # Track which winners have been processed (for complement calculation)
        for pos, winner_idx in enumerate(perm):
            # For this position, we multiply by keep_factors[winner_idx]
            # and by (1 - keep_factors[k]) for all k that came before
            complement_indices = list(perm[:pos])
            structure.append((winner_idx, winner_idx, complement_indices))
        
        # Final position gets leftover weight
        complement_indices = list(perm)
        structure.append((self.num_winners, None, complement_indices))
        
        return structure
    
    def make_wt_vec(self, perm_idx, keep_factors):
        """
        Fast weight vector calculation using precomputed structure.
        
        Args:
            perm_idx: Index of the permutation
            keep_factors: Array of keep factors for each winner
        
        Returns:
            Weight vector of length num_winners + 1
        """
        wt_vec = np.zeros(self.num_winners + 1)
        structure = self.structures[int(perm_idx)]
        
        for pos, keep_idx, complement_indices in structure:
            # Calculate the rolling weight from complements
            rolling_weight = 1.0
            for idx in complement_indices:
                rolling_weight *= (1 - keep_factors[idx])
            
            # Multiply by keep factor if not the final position
            if keep_idx is not None:
                wt_vec[pos] = rolling_weight * keep_factors[keep_idx]
            else:
                wt_vec[pos] = rolling_weight
        
        return wt_vec
    
    def make_wt_vec_vectorized(self, perm_indices, keep_factors):
        """
        Vectorized version that computes weight vectors for multiple permutations.
        
        Args:
            perm_indices: Array of permutation indices
            keep_factors: Array of keep factors for each winner
        
        Returns:
            2D array where each row is a weight vector
        """
        n = len(perm_indices)
        result = np.zeros((n, self.num_winners + 1))
        
        for i, perm_idx in enumerate(perm_indices):
            result[i] = self.make_wt_vec(perm_idx, keep_factors)
        
        return result
