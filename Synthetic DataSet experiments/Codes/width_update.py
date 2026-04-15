import numpy as np
from scipy.spatial.distance import pdist, squareform

def update_widths(centers, beta):
    '''
    this function is to update the widths that will be used when calculating the basis vector
    
    parameters:
    - beta : RBF width parameter (user defined)
    - centers: the trained centers of the current batch

    returns:
    - an np.array containing all the widths
    '''
    #the number of widths equate to the number of centers that we have (i.e the number of nodes in the hidden layer)
    num_widths = len(centers)
    
    #initialising an array to store the updated widths
    updated_widths = np.zeros(num_widths) 

    distance_matrix = squareform(pdist(centers, metric='euclidean'))

    # For speed up if the H more than 100
    #X_sq = np.sum(centers**2, axis=1).reshape(-1, 1)
    #distance_matrix = np.sqrt(np.maximum(X_sq + X_sq.T - 2 * centers @ centers.T, 0))

    updated_widths = distance_matrix.sum(axis = 0)*(beta / (num_widths)) 
        
    #clip to prevent extreme cases
    updated_widths = updated_widths**2+ 1e-5 #implemented according to the Matlab implementation of OSSN
    
    return updated_widths
