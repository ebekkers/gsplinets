# Some terminology
# G = N \rtimes H
# or: G = R^n \rtimes H
# g = (xx,h)
#
# xx is a vector in R^n
# xx= (x,y) in R^2 or xx = (x,y,z) in R^3
#
# n is the dimension of R^n
#
# N_k is the number of B-Splines
# N_x,N_y,N_z,N_h, the number of samples in the x,y,z axis and in the group
# N_h = length(grid_H)


# Usage:
# import gsplinets
# import gsplinets.group.se2 as se2

# glayers = gsplinets.Layers(se2)

# layer_1 = glayers.lifting_conv( tensor_in, x_max, grid_H , N_k)
# of 
# tensor_out, weights, centers_N = glayers.lifting_conv( tensor_in, x_max, grid_H , N_k)
# tensor_out, weights, centers_N, centers_H = glayers.group_conv( tensor_in, x_max, grid_H , N_k)

# TODO: check n-dimensional convolutions (currently conv2D is used everywhere..) and max-pooling
# TODO: second order B-splines
# TODO: Rn max-pooling
from gsplinets_tf.layers import layers