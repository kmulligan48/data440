import numpy as np
A = np.array([[1,4,-3], [2,-1,3]])
B = np.array([[-2,0,5], [0,-1,4]])
C = np.array([[1,0], [0,2]])
print(A.dot(B.T)+np.linalg.inv(C))