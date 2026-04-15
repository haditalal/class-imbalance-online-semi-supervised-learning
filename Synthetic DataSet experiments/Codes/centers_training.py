import numpy as np

def train_centers(centers, batch,lblSize, unlSize):
    '''
    this function trains the centers using old centers and the current batch

    parameters:
    - centers : the old centers to be trained
    - batch: the current batch
    -labSize: number of labelled examples of the batch
    -unlSize: number of unlabelled examples of the batch

    outputs:
    - the trained centers
    '''
    num_centers = len(centers)
    #create an empty array to store the trained centers, the number of columns equal to the length of each center
    trained_centers = np.empty((0, len(centers[0])))

    #calculate the regions of each center:
    regions = region_split(centers, batch)

    #train centers
    trained_centers = np.array([train_center(centers[i], regions[i], lblSize, unlSize) for i in range(num_centers)])
    return trained_centers

def train_center(center, region, lblSize, unlSize):
    '''
    This function will train a particular center based on the region of its closest examples

    parameters:
    - center : the center we want to train, expects np.array
    - region : the region of samples which have this center as the nearest one, expects np.array

    returns:
    - trained_center
    '''
    
    #split samples into labelled and unlabelled batches
    labelled = np.array([sample for sample in region if sample[-1] in [0,1]])
    unlabelled = np.array([sample for sample in region if sample[-1] == -1])

    #split the labelled batch into majority and minority sets
    maj, min = split_labelled_batch(labelled, len(center))

    #remove the last column of the majority, minority, and unlabelled
    maj = maj[:,:len(center)] if len(maj) > 0 else maj
    min = min[:,:len(center)] if len(min) > 0 else min
    unlabelled = unlabelled[:,:len(center)] if len(unlabelled) != 0 else unlabelled
    maj = maj[:,:len(center)] if len(maj) > 0 else maj
       
    #handle cases if there are no samples in the batch
    L = 1/lblSize if lblSize > 0 else 0
    U = 1/unlSize if unlSize > 0 else 0

    #since centers are only impacted by samples in their region, if their region is empty, then don't update the center
    if L==0 and U==0:
        return center

    #calculated the updated center
    maj_sum = np.sum(maj, axis=0) if len(maj) > 0 else 0 #sum of elements in majority set, 0 if empty
    min_sum = np.sum(min, axis=0) if len(min) > 0 else 0 #sum of elements in minority set, 0 if empty
    unlabelled_sum = np.sum(unlabelled, axis=0) if U > 0 else 0 #sum of elements in unlabelled set, 0 if empty
    
    numerator = (L * (maj_sum - min_sum)) + (U * unlabelled_sum)
    denominator = (L * (len(maj) - len(min))) + (U * len(unlabelled))

    if denominator == 0:
        return center

    return numerator / denominator

def region_split(centers, batch):
    """
    نسخة سريعة من region_split بنفس منطق الكود الأصلي:
    - لا تغيّر أي خطوة حسابية
    - تعطي نفس نتائج توزيع العينات على المراكز
    - فقط استبدلنا الحلقات بحساب مصفوفة مسافات vectorized
    """

    if len(centers) == 0 or len(batch) == 0:
        return [np.empty((0, batch.shape[1])) for _ in range(len(centers))]

    # centers: (H, d)
    # batch: (N, d+1) ← العمود الأخير هو التسمية
    num_centers = centers.shape[0]
    num_samples = batch.shape[0]
    num_features = centers.shape[1]

    # استخراج الميزات فقط من batch (بدون العمود الأخير)
    X = batch[:, :num_features]  # (N, d)

    # حساب مصفوفة المسافات: ||x - c||²
    # باستخدام (x^2 + c^2 - 2 x·c)
    X_sq = np.sum(X**2, axis=1).reshape(-1, 1)        # (N, 1)
    C_sq = np.sum(centers**2, axis=1).reshape(1, -1)  # (1, H)
    #distances = np.sqrt(X_sq + C_sq - 2 * X @ centers.T)  # (N, H)

    # ✅ حساب مصفوفة المسافات بدون أخطاء عددية
    distances = X_sq + C_sq - 2 * X @ centers.T
    distances = np.clip(distances, 0, None)   # قص القيم السالبة الناتجة عن الخطأ العددي
    distances = np.sqrt(distances)


    # إيجاد أقرب مركز لكل عينة
    closest_idx = np.argmin(distances, axis=1)  # (N,)

    # تقسيم العينات حسب أقرب مركز
    regions = [[] for _ in range(num_centers)]
    for i in range(num_samples):
        regions[closest_idx[i]].append(batch[i])

    # تحويل كل قائمة إلى np.array كما في الكود الأصلي
    regions = [np.array(region) for region in regions]

    return regions
    
def split_labelled_batch(labelled_batch, numAttr):
    '''
    this function separates the samples in the labelled batch into majority and minority sets
    if there are no labelled samples, it should return two empty sets

    parameters:
    - labelled_batch: set of labelled samples
    '''
    L = len(labelled_batch)

    #initialise the sets
    minority = np.empty((0,numAttr)) 
    majority = np.empty((0,numAttr))
    
    #return 2 empty sets if there's no labelled samples
    if L == 0:
        return majority, minority 

    majority = np.array([sample for sample in labelled_batch if sample[-1]==0])
    minority = np.array([sample for sample in labelled_batch if sample[-1]==1])

    #ensure minority has fewer samples
    if len(minority) > len(majority):
        minority, majority = majority, minority
   
    return majority, minority