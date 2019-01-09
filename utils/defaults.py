import numpy as np
from tf import transformations as tx

# default K (camera matrix)
K = np.reshape([
    499.114583, 0.000000, 325.589216,
    0.000000, 498.996093, 238.001597,
    0.000000, 0.000000, 1.000000], (3,3))#.astype(np.float32)
Ki = tx.inverse_matrix(K)

# default D (distortion)
D = np.float32([0.158661, -0.249478, -0.000564, 0.000157, 0.000000])

# camera extrinsic parameters
T_c2b = tx.compose_matrix(
                angles=[-np.pi/2-np.deg2rad(10),0.0,-np.pi/2],
                translate=[0.174,0,0.113])
T_b2c = tx.inverse_matrix(T_c2b)
