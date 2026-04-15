import numpy as np
from scipy.spatial import distance
from numba import njit, prange


def predict_multiple(samples, centers, widths, weights,bias):
    #remove label columns
    nAttrib = centers.shape[1]
    samples = samples[:,:nAttrib] 

    distances = distance.cdist(samples,centers,metric='euclidean')**2
    phi_vector = np.exp(- (distances/(2*(widths)) ));    

    z = phi_vector @ weights + bias

    prediction = 1 / (1 + np.exp(-z))
    return prediction,phi_vector



@njit(fastmath=True, parallel=True)
def predict(sample, centers, widths, weights,bias):

    #first ensure label column is removed
    nAttrib = centers.shape[1]
    sample = sample[:nAttrib]
    
    phi_vector = [] # initialise the phi vector (i.e hidden layer values)
    
    for i in range(centers.shape[0]): #for all centers
            phi = gaussian_basis(sample, centers[i], widths[i]) #calculate its phi value
            phi_vector.append(phi) #append

    phi_vector = np.array(phi_vector) #convert to numpy array
    z = np.sum(weights * phi_vector) + bias
    
    prediction = 1 / (1 + np.exp(-z))
                                      
    return prediction,phi_vector

@njit(fastmath=True)
def gaussian_basis(sample, center, width):

    numerator = -(np.sum((sample-center)**2)) 
    denominator = 2*(width)
    
    return np.exp(numerator/denominator)
