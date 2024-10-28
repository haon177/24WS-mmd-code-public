# Artur Andrzejak, October 2024
# Algorithms for collaborative filtering

import numpy as np
import scipy.sparse as sp
import unittest

def center_and_nan_to_zero(matrix, axis=0):
    """ Center the matrix and replace nan values with zeros"""
    # Compute along axis 'axis' the mean of non-nan values
    # E.g. axis=0: mean of each column, since op is along rows (axis=0)
    means = np.nanmean(matrix, axis=axis)
    # Subtract the mean from each axis
    matrix_centered = matrix - means
    return np.nan_to_num(matrix_centered)

def cosine_sim(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def fast_cosine_sim(utility_matrix, vector, axis=0):
    """ Compute the cosine similarity between the matrix and the vector"""
    # Compute the norms of each column
    norms = np.linalg.norm(utility_matrix, axis=axis)
    um_normalized = utility_matrix / norms
    # Compute the dot product of transposed normalized matrix and the vector
    dot = np.dot(um_normalized.T, vector)
    # Scale by the vector norm
    scaled = dot / np.linalg.norm(vector)
    return scaled

# Implement the CF from the lecture 1
def rate_all_items(orig_utility_matrix, user_index, neighbourhood_size):
    print(f"\n>>> CF computation for UM w/ shape: "
          + f"{orig_utility_matrix.shape}, user_index: {user_index}, neighborhood_size: {neighbourhood_size}\n")

    clean_utility_matrix = center_and_nan_to_zero(orig_utility_matrix)
    """ Compute the rating of all items not yet rated by the user"""
    user_col = clean_utility_matrix[:, user_index]
    # Compute the cosine similarity between the user and all other users
    similarities = fast_cosine_sim(clean_utility_matrix, user_col)

    def rate_one_item(item_index):
        # If the user has already rated the item, return the rating
        if not np.isnan(orig_utility_matrix[item_index, user_index]):
            return orig_utility_matrix[item_index, user_index]

        # Find the indices of users who rated the item
        users_who_rated = np.where(np.isnan(orig_utility_matrix[item_index, :]) == False)[0]
        # From those, get indices of users with the highest similarity (watch out: result indices are rel. to users_who_rated)
        best_among_who_rated = np.argsort(users_who_rated)
        # Select top neighborhood_size of them
        best_among_who_rated = best_among_who_rated[-neighbourhood_size:]
        # Convert the indices back to the original utility matrix indices
        best_among_who_rated = users_who_rated[best_among_who_rated]
        # Retain only those indices where the similarity is not nan
        best_among_who_rated = best_among_who_rated[np.isnan(similarities[best_among_who_rated]) == False]
        if best_among_who_rated.size > 0:
            # Compute the rating of the 
            sims_best = similarities[best_among_who_rated]
            rating_of_item = np.sum(sims_best * orig_utility_matrix[item_index, best_among_who_rated]) / np.sum(np.abs(sims_best))
        else:
            rating_of_item = np.nan
        print(f"item_idx: {item_index}, neighbors: {best_among_who_rated}, rating: {rating_of_item}")
        return rating_of_item

    num_items = orig_utility_matrix.shape[0]

    # Get all ratings
    ratings = list(map(rate_one_item, range(num_items)))
    return ratings

def centered_cosine_sim(u, v):
    """ Compute the centered cosine similarity between two sparse vectors"""
    v = np.nan_to_num(v)
    u = np.nan_to_num(u)
    
    if not sp.issparse(u):
        u = sp.csr_array(u, dtype = 'float64')
    if not sp.issparse(v):
        v = sp.csr_array(v, dtype = 'float64')    

    # somehow doesnt work here but works in the other function     
    # u.data = np.nan_to_num(u.data)
    # v.data = np.nan_to_num(v.data) 
    
    # Ensure there are non-zero elements to avoid division by zero
    if u.nnz == 0 or v.nnz == 0:
        return 0 
    
    # center
    u_mean = u.sum() / u.nnz
    v_mean = v.sum() / v.nnz
    u.data -= u_mean
    v.data -= v_mean
       
    # calculate cosine similarity
    dot = (u * v).sum()
    norm_u = sp.linalg.norm(u)
    norm_v = sp.linalg.norm(v)
    
    if norm_u == 0 or norm_v == 0:
        return 0
    
    return dot / (norm_u * norm_v)

def fast_centered_cosine_sim(utility_matrix, vector, axis=0):
    # Convert to sparse arrays
    if not sp.issparse(vector):
        vector = sp.csr_array(vector, dtype = 'float64')
    if not sp.issparse(utility_matrix):
        utility_matrix = sp.csr_array(utility_matrix, dtype = 'float64')
        
    # nan-> 0
    vector.data = np.nan_to_num(vector.data)
    means = np.nanmean(utility_matrix.todense(), axis = 0)
    
    nnz = vector.nnz
    # when vector empty, set mean to 0
    if nnz == 0:
        nnz = 1
    vmean = vector.sum() / nnz
    vector.data -= vmean

    # computing the means and subtracting from non zero entries 
    sums = utility_matrix.sum(axis = 0)
    nums = utility_matrix.nonzero()
    number_of_ratings = np.zeros(utility_matrix.shape[1])
    for n in nums[1]:
        number_of_ratings[n] +=1

    means = sums / number_of_ratings
    for i,j in zip(nums[0], nums[1]):
        utility_matrix[i,j] -= means[j]
    
    # normalize
    matrix_norms = sp.linalg.norm(utility_matrix, axis = 0)
    vector_norm = sp.linalg.norm(vector)
    utility_matrix = utility_matrix.multiply(1/matrix_norms)

    utility_matrix = utility_matrix.T.dot(vector.T)       
    utility_matrix = utility_matrix.multiply(1/vector_norm)
    return utility_matrix.todense().flatten()

# Testing the functions 
class TestCenteredCosineSimilarity(unittest.TestCase):
    
    def setUp(self):
        # Prepare test data as specified in the exercise sheet
        k = 100
        vector_x = [i+1 for i in range(k)]
        vector_y = [vector_x[k - j - 1] for j in range(k)]

        c_values = [c + x for c in [2,3,4,5,6] for x in np.arange(0,100,10)]

        vector_x2 = vector_x.copy()
        for c in c_values:
                vector_x2[c - 1] = np.nan
        vector_y2 = [vector_x2[k - j - 1] for j in range(k)]
        
        # vector to compare the fast similarity of the utility matrix since using x or y is trivial
        compare = np.random.rand((len(vector_x)))

        # stack vectors to Matrices and center them
        M = np.column_stack([vector_x, vector_y, compare])
        M_centered = center_and_nan_to_zero(M)
        M2 = np.column_stack([vector_x2, vector_y2, compare])
        M2_centered = center_and_nan_to_zero(M2)

        # Save test data as class members
        self.vector_x = vector_x
        self.vector_y = vector_y
        self.vector_x2 = vector_x2
        self.vector_y2 = vector_y2
        self.M = M
        self.M_centered = M_centered
        self.M2 = M2
        self.M2_centered = M2_centered
  
        self.vector_x_centered, self.vector_y_centered = M_centered[:,0], M_centered[:,1]
        self.vector_x2_centered, self.vector_y2_centered = M2_centered[:,0], M2_centered[:,1]


    def test_centered_cosine_sim(self):
        # compute cosine sim
        expected = cosine_sim(self.vector_x_centered, self.vector_y_centered)
        result = centered_cosine_sim(self.vector_x, self.vector_y)
        
        self.assertAlmostEqual(result, expected, 7, "centered_cosine_sim did not yield the correct result for non sparse vectors")
        
    def test_centered_cosine_sim_sparse(self):
        # compute cosine sim
        expected = cosine_sim(self.vector_x2_centered, self.vector_y2_centered)
        result = centered_cosine_sim(self.vector_x2, self.vector_y2)
        
        self.assertAlmostEqual(result, expected, 7, "centered_cosine_sim did not yield the correct result for sparse vectors")

    def test_fast_centered_cosine_sim(self):
        # Stack vectors into matrix to compute fast similarity against a vector
        expected = fast_cosine_sim(self.M_centered, self.M_centered[:,2])
        result = fast_centered_cosine_sim(self.M, self.M[:,2])   
        # Compare results pair wise since there is no Assertion with rounding errors for lists
        for i in range(len(expected)):
            self.assertAlmostEqual(result[i], expected[i], 7, "centered_cosine_sim did not yield the correct result for non sparse Matrices for entry %d"%i)

    def test_fast_centered_cosine_sim_sparse(self):
        # Stack vectors into matrix to compute fast similarity against a vector
        expected = fast_cosine_sim(self.M2_centered, self.M2_centered[:,2])
        result = fast_centered_cosine_sim(self.M2, self.M2[:,2])   
        # Compare results pair wise since there is no Assertion with rounding errors for lists
        for i in range(len(expected)):
            self.assertAlmostEqual(result[i], expected[i], 7, "centered_cosine_sim did not yield the correct result for sparse Matrices for entry %d"%i)

# Run tests
if __name__ == '__main__':
    unittest.main()