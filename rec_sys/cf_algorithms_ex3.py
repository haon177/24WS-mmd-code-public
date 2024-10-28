# Artur Andrzejak, October 2024
# Algorithms for collaborative filtering

import numpy as np
import scipy.sparse as sp
import unittest
from cf_algorithms import cosine_sim, fast_cosine_sim, center_and_nan_to_zero, centered_cosine_sim, fast_centered_cosine_sim

# Rewriting the CF algorithm to use sparse matrices
def rate_all_items(utility_matrix, user_index, neighbourhood_size):
    print(f"\n>>> CF computation for UM w/ shape: "
          + f"{utility_matrix.shape}, user_index: {user_index}, neighborhood_size: {neighbourhood_size}\n")
    
    """ Compute the rating of all items not yet rated by the user"""
    # convert to sparse if not already done    
    if not sp.issparse(utility_matrix):
        utility_matrix = sp.csr_array(utility_matrix, dtype = 'float64')
    
    # replace nans
    utility_matrix.data = np.nan_to_num(utility_matrix.data)
        
    # select user 
    user_col = utility_matrix[:, [user_index]]
    
    orig_utility_matrix = utility_matrix.copy()
    
    # Compute the cosine similarity between the user and all other users
    similarities = np.array(fast_centered_cosine_sim(utility_matrix, user_col.T))
    
    def rate_one_item(item_index):
        # If the user has already rated the item, return the rating
        if (utility_matrix[item_index, user_index]) != 0:
            return orig_utility_matrix[item_index, user_index]

        # Find the indices of users who rated the item
        rated = utility_matrix.nonzero()
        users_who_rated = rated[1][np.where(rated[0] == item_index)[0]]
        # From those, get indices of users with the highest similarity (watch out: result indices are rel. to users_who_rated)
        best_among_who_rated = np.argsort(users_who_rated)
        # Select top neighborhood_size of them
        best_among_who_rated = best_among_who_rated[-neighbourhood_size:]
        # Convert the indices back to the original utility matrix indices
        best_among_who_rated = users_who_rated[best_among_who_rated]
        best_among_who_rated = best_among_who_rated[similarities[best_among_who_rated] != 0]     
        if best_among_who_rated.size > 0:
            sims_best = similarities[best_among_who_rated]
            rating_of_item = np.sum(sims_best * orig_utility_matrix[[item_index], best_among_who_rated]) / np.sum(np.abs(sims_best))           
        else:
            rating_of_item = np.nan
        print(f"item_idx: {item_index}, neighbors: {best_among_who_rated}, rating: {rating_of_item}")
        return rating_of_item

    num_items = orig_utility_matrix.shape[0]

    # Get all ratings
    ratings = list(map(rate_one_item, range(num_items)))
    return ratings