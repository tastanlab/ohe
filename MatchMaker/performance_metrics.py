from scipy import stats
from sklearn.metrics import mean_squared_error
import numpy as np


def pearson(y, pred):
    pear = stats.pearsonr(y, pred)
    pear_value = pear[0]
    pear_p_val = pear[1]
    print("Pearson correlation is {} and related p_value is {}".format(pear_value, pear_p_val))
    return pear_value

def spearman(y, pred):
    spear = stats.spearmanr(y, pred)
    spear_value = spear[0]
    spear_p_val = spear[1]
    print("Spearman correlation is {} and related p_value is {}".format(spear_value, spear_p_val))
    return spear_value

def mse(y, pred):
    err = mean_squared_error(y, pred)
    print("Mean squared error is {}".format(err))
    return err

def squared_error(y,pred):
    errs = []
    for i in range(y.shape[0]):
        err = (y[i]-pred[i]) * (y[i]-pred[i])
        errs.append(err)
    return np.asarray(errs)

# Function to calculate SE
def calculate_se(y, pred):
    # Calculate squared errors
    squared_errs = squared_error(y, pred)
    
    # Standard error of MSE
    mse_se = np.std(squared_errs, ddof=1) / np.sqrt(len(squared_errs))
    print("Standard Error (SE) of MSE is {}".format(mse_se))
    
    return mse_se