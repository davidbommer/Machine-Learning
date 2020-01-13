# # Kernel Regression
# 
# Given a training dataset {x_i, y_i\}, kernel regression approximates the unknown 
# nolinear relation between $x$ and $y$ with a function of form
# $$
# y approx equals f(x, w) = sum from {i=1} to n {w_i k(x, x_i)}
# $$
# where k(x, x') is a positive definite kernel specified by the users, and {w_i} is a
# set of weights. 
# We will use the simple Gaussian radius basis funciton (RBF) kernel,
# where h is a bandwith parameter. 
# 
# 
# ### Step 1. Simulate a 1-dimensional dataset
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
np.random.seed(100)

### Step 1: Simulate a simple 1D data ###
xTrain = np.expand_dims(np.linspace(-5, 5, 100), 1)  # 100*1
yTrain = np.sin(xTrain) + 0.5*np.random.uniform(-1, 1, size=xTrain.shape) ## 100 *1

print('xTrain shape', xTrain.shape, 'yTrain shape', yTrain.shape)
plt.plot(xTrain, yTrain, '*')
plt.show()


# Now we have a dataset with 100 training data points. Let us calculate the kernel function. 
# 
# ### Step 2. Kernel function

import math 
def rbf(xi, xj, h):
    return math.exp(-((xi-xj)**2)/(2*h*h))
    
"""
    calcuating kernel matrix between X and Xp
"""
def rbf_kernel(X, Xp, h):
    # X: n*1 matrix
    # Xp: m*1 matrix
    # h: scalar value 
    n = X.shape[0]
    m = Xp.shape[0]
    K = np.zeros((n, m))
    for row in range(n):
        for column in range(m):
            K[row, column] = rbf(X[row], Xp[column], h)
    
    return K #n*m

   
# ### Step 3. The median trick for bandwith
# The choice of the bandwidth h
# A common way to set the bandwith h in practice is the so called median trick,
# which sets h to be the median of the pairwise distance on the training data, that is
# 
# h_{med} = median(||x_i - x_j||: i!=j, i,j = 1,...,n).

from scipy.spatial import distance

def median_distance(X):
    # X: n*1 matrix
    dist = 0
    count = 0
    for i in range(X.shape[0]):
        for j in range(i, X.shape[0]):
            if (i != j):
                dist += math.sqrt((X[i]-X[j])**2)
                count += 1
    h = dist / count
    
    return h



# ### Step 4. Kernel regression
# The weights {w_i} are estimated by minimizing a regularized mean square error:
# where w is the column vector formed by w=[w_i] from i=1 to n and K is the kernel matrix.
# 

from numpy.linalg import inv
def kernel_regression_fitting(xTrain, yTrain, h, beta=1):
    # X: input data, numpy array, n*1
    # Y: input labels, numpy array, n*1
    #W=((K+BI)^-1)Y
    K = rbf_kernel(xTrain, xTrain, h)
    n = xTrain.shape[0]
    I = np.identity(n)
    B = beta
    K = np.add(K, I*B)
    K = inv(K)
    W = np.matmul(K,yTrain)
    return W



# ### Step 5. Evaluation and Cross Validation
# 
# We now need to evaluate the algorithm on the testing data and select the hyperparameters (bandwidth and regularization coefficient) using cross validation

def kernel_regression_fit_and_predict(xTrain, yTrain, xTest, h, beta):
    
    #fitting on the training data 
    W = kernel_regression_fitting(xTrain, yTrain, h, beta)
    
    # computing the kernel matrix between xTrain and xTest
    K_xTrain_xTest = rbf_kernel(xTrain, xTest, h)
   
    # predict the label of xTest
    yPred = np.dot( K_xTrain_xTest.T, W)
    return yPred

# generate random testing data
xTest = np.expand_dims(np.linspace(-6, 6, 200), 1) ## 200*1


beta = 1.
# calculating bandwith
h_med = median_distance(xTrain)  
yHatk = kernel_regression_fit_and_predict(xTrain, yTrain, xTest, h_med, beta)


# we also add linear regression for comparision
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(xTrain, yTrain)  
yHat = lr.predict(xTest) # prediction

# visulization
plt.plot(xTrain, yTrain, '*')
plt.plot(xTest, yHat, '*')
plt.plot(xTest, yHatk, '-k')
plt.show()


# ### Step 5.1. Impact of bandwith 
# Run the kernel regression with regularization coefficient beta=1 and 
# bandwidth h in {0.1*h_{med}, h_{med}, 10*h_{med}}.
# 
# Show the curve learned by different h. 

### fitting on the training data ###
beta = 1.

plt.figure(figsize=(12, 4))
for i, coff in enumerate([0.1, 1., 10]):
    plt.subplot(1, 3, i+1)
    
    #run kernel regression with bandwith h = coff * h_med. 
    yHatk_i =  kernel_regression_fit_and_predict(xTrain, yTrain, xTest, h_med*coff, beta)
    
    # visulization
    plt.plot(xTrain, yTrain, '*')
    plt.plot(xTest, yHat, '*')
    plt.plot(xTest, yHatk_i, '-k')
    plt.title('handwidth {} x h_med'.format(coff))
    
plt.show()
#Lower h leads to overfitting of the data 
#too high of an h leads to just linear regression 


# ### Step 5.2. Cross Validation (CV)
# Use 5-fold cross validation to find the optimal combination of 
# h and beta within h in {0.1*h_{med}, h_{med}, 10*h_{med}} and beta in {0.1, 1}.

best_beta, best_coff = 1., 1.
best_mse = 1e8
for beta in [0.1, 1]:
    for coff in [0.1, 1., 10.]:
        # 5-fold cross validation
        max_fold = 5
        mse = []
        stepSize = xTrain.shape[0] // max_fold
        for i in range(max_fold):
            
            ##TODO: calculate the index of the training/testing partition within 5 fold CV.
            # (hint: set trnIdx to be these index with idx%max_fold!=i, and testIdx with idx%max_fold==i)
            trnIdxBegin = i * stepSize
            trnIdxEnd = trnIdxBegin + stepSize
            if (i == 0):
                i_xTrain, i_yTrain = xTrain[trnIdxEnd:], yTrain[trnIdxEnd:]
                i_xValid, i_yValid = xTrain[:trnIdxEnd], yTrain[:trnIdxEnd]
            elif(i == 4):
                i_xTrain, i_yTrain = xTrain[:trnIdxBegin], yTrain[:trnIdxBegin]
                i_xValid, i_yValid = xTrain[trnIdxBegin:], yTrain[trnIdxBegin:]
            else:
                xTrainPre, yTrainPre = xTrain[:trnIdxBegin], yTrain[:trnIdxBegin]
                xTrainPost, yTrainPost = xTrain[trnIdxEnd:], yTrain[trnIdxEnd:]
                i_xTrain, i_yTrain = np.concatenate((xTrainPre, xTrainPost), axis=0), np.concatenate((yTrainPre, yTrainPost), axis=0)
                
                i_xValid, i_yValid = xTrain[trnIdxBegin:trnIdxEnd] , yTrain[trnIdxBegin:trnIdxEnd]
            
            ##TODO: run kernel regression on (i_xTrain, i_yTrain) and calculate the mean square error on (i_xValid, i_yValid)
            h = median_distance(i_xTrain)
            i_yPred = kernel_regression_fit_and_predict(i_xTrain, i_yTrain, i_xValid, h*coff, beta)     
            mse.append((i_yValid - i_yPred)**2)
        mse = np.mean(mse)
        # keep track of the combination with the best MSE
        if mse < best_mse:
            best_beta, best_coff = beta, coff
            best_mse = mse
        
print('Beta beta', best_beta, 'Best bandwith', '{}*h_med'.format(best_coff), 'mse', best_mse)

# bandwith
h = best_coff * median_distance(xTrain)  
yHatk_i = kernel_regression_fit_and_predict(xTrain, yTrain, xTest, h, best_beta)
    
# visulization
plt.plot(xTrain, yTrain, '*')
plt.plot(xTest, yHatk_i, '-k')
plt.title('beta {}, bandwidth {}h_med'.format(best_beta, best_coff))
plt.show()
