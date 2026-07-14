import numpy as np
def Amari_error(A_est,A_true):
    if A_true.shape[0]==A_true.shape[1]:
        G = np.dot(np.linalg.inv(A_est), A_true)
    else:
        G = np.dot(np.linalg.pinv(A_est), A_true)    
    P = np.abs(G)  
    n = P.shape[0]
    
    row_sum = np.sum(P / P.max(axis=1)[:, None], axis=1) - 1
    col_sum = np.sum(P / P.max(axis=0)[None, :], axis=0) - 1
    
    error = (np.sum(row_sum) + np.sum(col_sum)) / (2 * n)
    return error
