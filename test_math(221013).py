import numpy as np
joint = np.zeros((21, 3))


print(len(joint))
v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:]

print(len(v1))
print(len(v2))
v = v2 -v1
print(v)