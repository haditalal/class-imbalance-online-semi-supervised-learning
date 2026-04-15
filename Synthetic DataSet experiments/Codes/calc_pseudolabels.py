import numpy as np
from scipy.spatial import distance

def pseudolabels_calc(C, lab, c_unl, c_unl_pred, alpha, gamma):
    '''
    this function is to calculate the pseudolabels which will be used for updating the weights and calculating the loss
    we will require the labels of the labelled data, and the predicted labels of the unlabelled data.
    in order to calculate the predicted values, we will require to run the unlabelled data through the network

    parameters: 
    - C : centers
    - lab : labelled examples from the batch 
    - c_unl : centers + unlabelled examples from the batch
    - c_unl_pred : predicted values of the centers  and the unlabelled examples from the batch    
    - alpha : L2 regularisation term
    - gamma : RBFN width

    returns:
    - pseudolabels
    '''
    
    V_columns = np.vstack((lab,c_unl)) if lab.size else c_unl
    labels_c_unl_pred = np.concatenate((lab[:,-1],c_unl_pred)) if lab.size else c_unl_pred

    #calculate the similarity matrix
    S_matrix = similarity_matrix(c_unl, V_columns, gamma) #implemented according to the Matlab implementation of OSSN
   
    # New code speed up
    numerator = S_matrix @ labels_c_unl_pred
    denominator = S_matrix.sum(axis=1) + 1e-5
    pseudolabels = numerator / denominator


    return pseudolabels

def sigma_calc(matrix, gamma):
    '''
    a function to calculate the distance from each vertex to it's nearest neighbour

    parameters:
    - matrix : 2D np array of the vertices
    - gamma : RBFN width
    '''
    min_distances = matrix.min(axis=1)

    sigmas = min_distances * gamma

    #clip them
    sigmas = sigmas + 1e-5    
    return sigmas

def similarity_matrix(V_lines, V_columns, gamma):
    '''
    a function to generate the similarity matrix of a dataset V and some provided gamma

    parameters:
    - V_lines : lines of the matrix V
    - V_columns: columns of the matrix V
    - gamma : RBFN width
    '''

    similarity_matrix = distance.cdist(V_lines[:,:-1],V_columns[:,:-1],metric='euclidean')**2

    sigmas = sigma_calc(similarity_matrix, gamma)    

    similarity_matrix = np.exp(- similarity_matrix / (2 * (sigmas[:, None]**2)))


    return similarity_matrix