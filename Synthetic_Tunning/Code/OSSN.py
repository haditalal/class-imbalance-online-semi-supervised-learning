import numpy as np
from predict_function import *
from centers_training import *
from width_update import *
from calc_pseudolabels import *
from weight_update import *
from calc_CEL import *

def remove_unlabelled_samples(batch):
    '''
    This function removes the unlabelled datapoints to convert the model into supervised learning.
    Returns an empty array with the correct shape if there are no labeled samples.
    '''
    filtered_batch = [sample for sample in batch if sample[-1] != -1]
    
    if len(filtered_batch) == 0:
        #return an empty array with the correct number of columns 
        return np.empty((0, batch.shape[1]))
    
    return np.array(filtered_batch)

def OSNN(D, N, H, lam, alpha, beta, gamma, fadingFactor, type=0):
    '''
    The main body of the OSNN algorithm. This model assumes that the number of neurons is less than the chunk size (i.e, H < N)

    parameters:
    - D : dataset
    - N : chunk size
    - H : number of neurons in the network
    - lam : manifold regularisation term
    - alpha : L2 regularisation term
    - beta : RBF_width
    - gamma : RBFN_width
    - fadingFactor: fading use to compute the accuarcy
    - type : 1 for supervised, 0 for semisupervised
    
    returns:
    - the predictions made, this is given as a numpy array structured as [predicted_probability, predicted_class, true_label, assigned_label, accuracy]
    - the trained model at the final time step
        - The weights
        - The centers
        - The widths 
    '''

    #fix a seed
    np.random.seed(1)
    
    #initialise the algorithm
    t = 0
    nAttrib = len(D[0]) -2
    C = np.empty((0, nAttrib)) #the centers has nAttrib columns
    w = np.empty((0, 0)) #an empty list
    batchOrig = np.empty((0, nAttrib+1 )) #similar to C but one extra column for 'contains_bug' 
    bias = np.random.normal(0, 1.0)
    preqS = 0
    preqN = 0

    if type == 1: #if we are setting the model to be supervised only, lam = 0
        lam = 0
        
    #initialise a list to store predicted values and its true label
    predictions = np.empty((0,5))

   
    while t < len(D): #while there is data remaining in the dataset
        #if t%100 == 0: #to track the progress of the algorithm
           # print(t)

        if t % 1000 == 0:
            print(f"Processing sample {t} / {len(D)}")


        #set current batch
        if len(batchOrig) < N: #if the size of the batch is less than the chunk size, 
            batchOrig = np.vstack((batchOrig, D[t][:-1])) #append the most recent sample
        else:
            batchOrig = D[t-N+1:t+1,:-1] #otherwise set the batch to be the N most recent samples

        #if we are setting the model to be supervised only, we remove the unlabelled data in the batch
        if type == 1:
            batch = remove_unlabelled_samples(batchOrig)
        else:
            batch = batchOrig

        #if the batch is empty:
        if len(batch) == 0: 
            #if there are no centers or weights, then there's nothing to predict so move to the next time step
            if C.size == 0 or w.size == 0:
                t += 1
            #if there are centers and weights and the batch is empty, then just predict the next sample using the current weights, 
            #centers and widths and move to the next time step.
            else:
                if ((t+1) < len(D)):
                    prob,_ = predict(D[t+1], C, widths, w,bias)
                    s = 0 if prob < 0.5 else 1
                    #store probability, prediction, true label, 'contains bug' label, accuracy
                    testAcc = s == D[t+1,-1]
                    preqS = testAcc + fadingFactor*preqS
                    preqN = 1 + fadingFactor*preqN
                    preqAcc = preqS/preqN
                    predictions = np.vstack((predictions, [prob, s, D[t+1,-1], D[t+1,-2], preqAcc]))
                t += 1

        #if the batch is not empty, then see if we need to add centers, add them, else, train them, update, predict, and move to the next time step
        else:
            #separated labelled and unlabelled examples in the batch
            lab = np.array([sample for sample in batch if sample[-1] in [0,1]])
            unl = np.array([sample for sample in batch if sample[-1] == -1])
            
            if len(C) < H: 
                #while the number of centers is less than the number of neurons that we assigned
                #this is to ensure that we have enough centers for all the nodes of the network before we begin training
                
                #add that sample to the set of centers
                C = np.vstack((C, D[t][0:nAttrib])) #it ensures that only the attributes columns are added and the contains_bug and true label columns are excluded
                
                #initialise a new weight for that center 
                w = np.append(w, 0.05*np.random.normal(0, 1.0))                
            else:
                #train centers
                C = train_centers(C, batch, len(lab), len(unl))                
            
            #update the widths using the centers and the RBF width parameter, beta
            widths = update_widths(C, beta)
    
            #to calculate the pseudolabels and crossentropy loss, we require the predicted values of all samples in the batch, and the centers
            #lab_pred ,lab_phi = predict_multiple(lab, C, widths, w,bias) if len(lab) > 0 else ([],[])
            lab_pred ,lab_phi = predict_multiple(lab, C, widths, w,bias) if len(lab) > 0 else (np.empty((0,)), np.empty((0,)))

            
            #first we add a column to the end of the centers
            C_with_labels = np.hstack((C, -1 * np.ones((len(C), 1))))
            #concatenate centers and unlabeled --> #implemented according to the Matlab implementation of OSSN
            c_unl = np.vstack((C_with_labels, unl)) if unl.size else C_with_labels
            c_unl_pred,c_unl_phi = predict_multiple(c_unl, C, widths, w,bias)  
  
            #calc pseudolabels using unlabelled (centers+unalabelled of the batch) and labelled
            mu = pseudolabels_calc(C, lab, c_unl, c_unl_pred, alpha, gamma) 
            #set learning rate eta = 1 and a stopping point epsilon. (epsilon is a user parameter that'll require experimenting with)
            eta = 1
            epsilon = 0.0039 # =2^-8
            cont = 0
            while eta > epsilon: 
                cont+=1
                loss = cross_entropy_loss(lab, c_unl, c_unl_pred, lab_pred, mu, w, alpha, lam,bias) 
                #update the weights 
                w_new,bias_new = update_weights(w, lab, c_unl, c_unl_pred, lab_pred, mu, alpha, lam, C, widths, eta,bias,lab_phi,c_unl_phi )

                lab_pred_new, _ = predict_multiple(lab, C, widths, w_new,bias_new) if len(lab) > 0 else ([],[])
                c_unl_pred_new, _ = predict_multiple(c_unl, C, widths, w_new,bias_new)

                loss_new = cross_entropy_loss(lab, c_unl, c_unl_pred_new, lab_pred_new, mu, w_new, alpha, lam, bias_new)

                #if the loss of the new weights is less than the loss of the current weights, set the new weights as the current weights               
                if  loss_new < loss:
                    w = w_new
                    bias = bias_new
                    break #breaks out of the 'while eta > epsilon:' loop
                else:
                    eta = eta/2   
            if ((t+1) < len(D)): #if there is one more example in the dataset, then predict it
                prob,_ = predict(D[t+1], C, widths, w,bias)
                s = 0 if prob < 0.5 else 1
                testAcc = s == D[t+1,-1]
                preqS = testAcc + fadingFactor*preqS
                preqN = 1 + fadingFactor*preqN
                preqAcc = preqS/preqN
                #store probability, prediction, true label, 'contain's bug' label, and accuracy
                predictions = np.vstack((predictions, [prob, s, D[t+1,-1], D[t+1,-2], preqAcc]))
            t += 1
    
    return predictions # using C, widths, w
__all__ = ["OSNN"]
