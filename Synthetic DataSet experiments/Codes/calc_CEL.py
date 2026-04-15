import numpy as np


def cross_entropy_loss(lab, c_unl, c_unl_pred, lab_pred, mu, w, alpha, lam, bias):
    '''
    This function returns the cross entropy loss of the network

    parameters:
    - lab : labelled examples from the batch 
    - c_unl : centers + unlabelled examples from the batch
    - c_unl_pred : predicted values of the centers  and the unlabelled examples from the batch 
    - lab_pred: predicted values of the labelled examples from the batch
    - mu : pseudolabels
    - w : weights
    - alpha : L2 regularisation term
    - bias: bias

    returns:
    - loss
    '''

    m = 1/len(lab) + 1/len(c_unl) if len(lab) > 0 else 1/len(c_unl) #implemented according to the Matlab implementation of OSSN

    #contribution to loss of labelled batch
    class0_loss = 0
    class1_loss = 0
    unlabelled_loss = 0
    L = 0
    U = 0

    for i in range(len(lab)): #iterate over all labelled examples
        
        if lab[i][-1] == 1: #if it's labelled and of class 1, add to the class 1 running sum
            class1_loss += np.log(lab_pred[i])
            L += 1
            
        elif lab[i][-1] == 0: #if it's labelled and of class 0, add to the class 0 running sum
            #class0_loss += np.log(1-lab_pred[i])
            # امنع القيم من الوصول إلى 0 أو 1
            eps = 1e-12
            p = np.clip(lab_pred[i], eps, 1 - eps)
            # الآن log(1 - p) آمن
            class0_loss += np.log(1 - p)

            L += 1
    if (lam!=0):
        for i in range(len(c_unl)): ##iterate over all unlabelled examples
            #unlabelled_loss += mu[i]*np.log(c_unl_pred[i]) + (1 - mu[i])*np.log(1-c_unl_pred[i])
            eps = 1e-10  # قيمة صغيرة جداً لتفادي log(0)
            unlabelled_loss += mu[i]*np.log(c_unl_pred[i] + eps) + (1 - mu[i])*np.log(1 - c_unl_pred[i] + eps)

            U += 1

    #to avoid dividing by 0 errors:
    l = 0 if L == 0 else 1/L
    u = 0 if U == 0 else 1/U

    w_bias = np.append(w,bias)
    #l_2 regularisation loss
    l2_reg = (alpha*m /2) * (np.linalg.norm(w_bias)**2)

    loss = -l*(class1_loss + class0_loss) - (lam*u)*(unlabelled_loss) + l2_reg  
    return loss