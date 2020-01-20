import tensorflow as tf
import numpy as np
import uuid
import numbers

from . import bsplines
# from .utils_TF import swapaxes, depthwise_conv_1D_in_ND

# TODO: Automatic option for h_basis_scale in ConvGG

# Start of (parent class) 
class layers:
    def __init__(self, group_name):
        import importlib
        group = importlib.import_module('gsplinets_tf.group.' + group_name)
        self.group = group
        self.G = group.G
        self.Rn = group.Rn
        self.H = group.H

    # Creates a lifting_layer object
    def ConvRnRn(
            self,
            # Required arguments
            inputs,                 # Should be a tensor of dimension self.Rn.n + 2 (Batch axis, Rn.n spatial axes, Feature channel axis)
            N_out,                  # Number of output channels
            kernel_size,            # Kernel size (integer)
            # Optional basis function related arguments
            xx_basis_size=None,     # Nr of spline basis functions in each axis
            xx_basis_scale=1,       # Scale of the splines (spacing between the splines)
            xx_basis_type = 'B_2',  # Type of B-Spline basis function
            xx_basis_mask = False,  # Whether or not to force the xx_centers to be within a disk mask
            # Optional generic arguments
            stride=1,               # Spatial stride in the convolution
            padding='VALID',        # Padding type
            name=None):             # Name of generated tensorflow variables

        with tf.compat.v1.name_scope('ConvRnRn'):
            return ConvRnRnLayer(self.group, inputs, N_out, kernel_size, xx_basis_size, xx_basis_scale, xx_basis_type, xx_basis_mask, stride, padding, name)

    # Creates a lifting_layer object
    def ConvRnG(
            self,
            # Required arguments
            inputs,                 # Should be a tensor of dimension self.Rn.n + 2 (Batch axis, Rn.n spatial axes, Feature channel axis)
            N_out,                  # Number of output channels
            kernel_size,            # Kernel size (integer)
            h_grid,                 # The grid of H on which to compute the output (see H.grid for more details)
            # Optional basis function related arguments
            xx_basis_size=None,     # Nr of spline basis functions in each axis
            xx_basis_scale=1,       # Scale of the splines (spacing between the splines)
            xx_basis_type = 'B_2',  # Type of B-Spline basis function
            xx_basis_mask = False,  # Whether or not to force the xx_centers to be within a disk mask
            # Optional generic arguments
            stride=1,               # Spatial stride in the convolution
            padding='VALID',        # Padding type
            name=None):             # Name of generated tensorflow variables

        with tf.compat.v1.name_scope('ConvRnG'):
            return ConvRnGLayer(self.group, inputs, N_out, kernel_size, h_grid, xx_basis_size, xx_basis_scale, xx_basis_type, xx_basis_mask, stride, padding, name)

    # Creates a group convolution layer object
    def ConvGG(
            self,
            # Required arguments
            inputs,                 # Should be a tensor of dimension self.Rn.n + 2 (Batch axis, Rn.n spatial axes, Feature channel axis)
            N_out,                  # Number of output channels
            kernel_size,            # Kernel size (integer)
            h_grid,                 # The grid of H on which to compute the output (see H.grid for more details)
            # Optional basis function related arguments
            xx_basis_size=None,     # Nr of spline basis functions in each axis
            xx_basis_scale=1,       # Scale of the splines (spacing between the splines)
            xx_basis_type = 'B_2',  # Type of B-Spline basis function
            xx_basis_mask = False,  # Whether or not to force the xx_centers to be within a disk mask
            h_basis_size=None,      # Nr of spline basis functions to cover H
            h_basis_scale=None,     # Scale of the splines (spacing between the splines), a list of scales (1 for each exp coordinate). If None then it uses the resolution of h_grid
            h_basis_type = 'B_2',   # Type of B-Spline basis function
            h_basis_local = False,  # Whether basis is uniformly covering the group (False) or is localized (True)
            # Optional grid re-sampling related arguments
            input_h_grid = None,    # In case the input grid is not the same as the intended output grid (see h_grid parameter)
            # Optional generic arguments
            stride=1,               # Spatial stride in the convolution
            padding='VALID',        # Padding type
            name=None):             # Name of generated tensorflow variables

        with tf.compat.v1.name_scope('ConvGG'):
            return ConvGGLayer(self.group, inputs, N_out, kernel_size, h_grid, xx_basis_size, xx_basis_scale, xx_basis_type, xx_basis_mask, h_basis_size, h_basis_scale, h_basis_type, h_basis_local, input_h_grid, stride, padding, name)

    def max_pooling_Rn(self, input, pool_size, stride , padding = 'SAME'):
        with tf.compat.v1.name_scope('MaxPoolRn'):
            # Check number of manifold dims
            ndims = input.get_shape().ndims - 2  # minus batch-dimension and channels-dimension
            if ndims == self.Rn.n:
                output = tf.compat.v1.layers.max_pooling2d( input, pool_size, stride, padding = padding)
            else:
                N_h = int(input.shape[-2])
                output = tf.stack( [tf.compat.v1.layers.max_pooling2d( input[...,i,:], pool_size, stride, padding = padding) for i in range(N_h)] , -2)
            return output

    def average_pooling_Rn(self, input, pool_size, stride , padding = 'SAME'):
        with tf.compat.v1.name_scope('MaxPoolRn'):
            # Check number of manifold dims
            ndims = input.get_shape().ndims - 2  # minus batch-dimension and channels-dimension
            if ndims == self.Rn.n:
                output = tf.compat.v1.layers.average_pooling2d( input, pool_size, stride, padding = padding)
            else:
                N_h = int(input.shape[-2])
                output = tf.stack( [tf.compat.v1.layers.average_pooling2d( input[...,i,:], pool_size, stride, padding = padding) for i in range(N_h)] , -2)
            return output



##########################################################################
############################## ConvRnRnLayer #############################
##########################################################################

# The core deformable B-Spline class on Rn
class ConvRnRnLayer:
    def __init__(
            self, 
            group,
            inputs,
            N_out,
            kernel_size,
            xx_basis_size,
            xx_basis_scale,
            xx_basis_type,
            xx_basis_mask,
            stride,
            padding,
            name
            ):
        
        ## Assert and set inputs
        self.kernel_type = 'Rn'
        self._assert_and_set_inputs(group, inputs, N_out, kernel_size, xx_basis_size, xx_basis_scale, xx_basis_type, xx_basis_mask, stride, padding, name)

        ## Construct the trainable weights
        self.xx_centers = self.variable_xx_centers_uniform(  constant = True, trainable = False, name = self.name)
        self.N_k = int(self.xx_centers.shape[1]) # Number of basis functions
        self.weights = self.variable_weights( name = self.name )

        ## Compute the output
        self.outputs = self.output()

########################### Assert and set inputs ########################

    ## Assert inputs
    def _assert_and_set_inputs(self, group, inputs, N_out, kernel_size, xx_basis_size, xx_basis_scale, xx_basis_type, xx_basis_mask, stride, padding, name):
        self._assert_and_set_inputs_RnRn(group, inputs, N_out, kernel_size, xx_basis_size, xx_basis_scale, xx_basis_type, xx_basis_mask, stride, padding, name)

    def _assert_and_set_inputs_RnRn(self, group, inputs, N_out, kernel_size, xx_basis_size, xx_basis_scale, xx_basis_type, xx_basis_mask, stride, padding, name):
        ## Check (and parse) all the inputs
        # Include the dictionary of the used parent class
        self.group = group
        self.G = group.G
        self.H = group.H
        self.Rn = group.Rn
        
        # Mandatory inputs
        self.inputs = self._assert_inputs(inputs)
        self.N_in = inputs.shape[-1]
        self.N_out = self._assert_N_out(N_out)
        self.kernel_size = self._assert_kernel_size(kernel_size)
        
        # Optional arguments (spatial B-Splines)
        self.xx_basis_size = self._assert_xx_basis_size(xx_basis_size)
        self.xx_basis_scale = self._assert_xx_basis_scale(xx_basis_scale)
        self.xx_basis_type = xx_basis_type
        self.xx_basis_mask = xx_basis_mask

        # Optional inputs (standard conv)
        self.padding=padding
        self.stride=stride
        if (name is None):  # Then set name as random unique string of length 6
            self.name = uuid.uuid4().hex[:6].upper()
        else:
            self.name = name

        ## Derived variables
        # Generate the spatial sampling grid
        self.xx_grid = tf.constant(self._xx_grid_np( self.kernel_size) )

        # B-spline specification
        self.xx_B_type, self.xx_B_degree = self._assert_basis_type( self.xx_basis_type )
        self.xx_B = bsplines.B(self.xx_B_degree, scale = xx_basis_scale) # Python function that takes as input a coordinate xx in R^n

    def _assert_inputs(self, inputs):
        # input_tensor
        ndims = inputs.get_shape().ndims - 2  # minus batch size and n-channels
        assert (ndims == self.Rn.n), "The manifold dimension of the input tensor is not {}.".format(self.Rn.n)
        return inputs

    def _assert_N_out(self, N_out):
        assert isinstance(N_out, int), "The specified argument \"N_out\" should be an integer."
        return N_out

    def _assert_kernel_size(self, kernel_size):
        assert isinstance(kernel_size, int), "The specified argument \"kernel_size\" should be an integer."
        return kernel_size

    def _assert_xx_basis_size(self, xx_basis_size_in):
        xx_basis_size = xx_basis_size_in
        if xx_basis_size == None:
            xx_basis_size = self.kernel_size
        assert isinstance(xx_basis_size, int), "The specified argument \"xx_basis_size\" should be an integer."
        return xx_basis_size

    def _assert_xx_basis_scale(self, xx_basis_scale):
        assert isinstance(xx_basis_scale, int) or isinstance(xx_basis_scale, float), "The specified argument \"xx_basis_scale\" should be a scalar (either int or float)"
        return float(xx_basis_scale)

    def _assert_basis_type(self, basis_type):
        B_type = basis_type.split('_')[0]
        B_degree = int(basis_type.split('_')[1])
        assert (B_type in ['B','B2']), 'Unkown option value \"{}\" for basis function B_type (\"B_n\" for normal splines, \"B2_n\" for second order derivative splines of order n (int)).'.format(basis_type)
        return B_type, B_degree

##################### Construct the trainable weights ####################

    def variable_xx_centers_uniform(self, constant=True, trainable = False, name = None):
        # Uniform:
        if self.kernel_type == "Rn":
            # If only spatial kernel then construct the grid along each spatial axis
            init = np.repeat(self._xx_grid_np( self.xx_basis_size , flatten = True, masked=self.xx_basis_mask, scale = self.xx_basis_scale)[np.newaxis,...],self.N_in,0)
        else:
            # If the kernel is on Rn x H then repeat the spatial grid for each h
            init = np.repeat(self._xx_grid_np( self.xx_basis_size , flatten = True, masked=self.xx_basis_mask, scale = self.xx_basis_scale)[np.newaxis,...],self.N_in,0)
            init = np.repeat(init,self.h_basis_size,1)
        
        # Variable name
        if not(name is None):
            nameTMP = name + '_xx'
        else:
            nameTMP = None

        # Construct centers
        if not(constant):# Then the variable can be made trainable or not
            shape = None
            xx_centers = tf.compat.v1.get_variable( nameTMP, shape, trainable=trainable, initializer=init, constraint=lambda t: tf.clip_by_value(t, -self.xx_i_max, self.xx_i_max), dtype=tf.float32)
        else:
            xx_centers = tf.constant( init , dtype = tf.float32 )

        # Return the centers
        return xx_centers

    def variable_weights(self, name = None):
        # For each input channel, for each basis function, for each output channel a weight
        # So this returns a 3D array
        # Variable name
        if not(name is None):
            nameTMP = name + '_weights'
        else:
            nameTMP = None
        # For each basis function a weight variable
        weight_variable = tf.compat.v1.get_variable(
            nameTMP,
            [self.N_in, self.N_k, self.N_out],
            initializer=self._xavier_initializer(int(self.N_in) * self.N_k, self.N_out))
        # Return the weight variable
        return weight_variable

############################ Compute the output ##########################

    ## Public functions
    def kernel(self, h = None):
        # The transformation to apply
        if h is None:
            h = self.H.e
        # The kernel as a function
        Kernel = self.spline_Rn( self.xx_centers, self.weights, self.xx_B )
        # Sample the kernel on the (transformed) grid
        return (1/self.H.det(h))*Kernel( self.H.left_action_on_Rn( self.H.inv(h), self.xx_grid ) )

    def output(self):
        return self.conv_Rn_Rn( self.kernel, self.inputs, self.padding , self.stride)

    def conv_Rn_Rn(self, kernel, input_tensor, padding = 'VALID', xx_stride = 1):
        output = tf.nn.conv2d(
                        input=input_tensor,
                        filter=kernel(self.H.e),
                        strides=[1, xx_stride, xx_stride, 1],
                        padding=padding)
        # Return the output
        return output

    def spline_Rn(self, xx_centers, weights, xx_B ):
        # This function
        def spline_Rn( xx ):
            n = self.Rn.n
            xx_B_sampler = self._spline_Rn_stack( xx_centers, xx_B )
            xx_B_sampled = xx_B_sampler( xx )

            # Reshape the weights
            weights_shape = [int(N_i) for N_i in weights.shape]
            _weights = tf.reshape(weights, [1]*(n) + weights_shape)
            xx_B_sampled = tf.expand_dims(xx_B_sampled,-1)
            xx_B_recon = tf.reduce_sum(input_tensor=xx_B_sampled*_weights,axis=-2) # Sum over the splines

            return xx_B_recon
        return spline_Rn

######################### Private helper functions #######################

    def _xx_grid_np(self, N_xx, flatten = False, scale = 1, masked = False):
        xx_max= (N_xx - 1)/2
        grid = np.moveaxis(np.mgrid[tuple([slice(-xx_max,xx_max+1)]*self.Rn.n)],0,-1).astype(np.float32)
        if flatten or masked:
            output_grid = scale*np.reshape(grid,[-1,self.Rn.n])
            if masked:
                toselect=[np.linalg.norm(np.array(vec))<=scale*xx_max + 0.5 for vec in output_grid]
                output_grid = output_grid[toselect]
            return output_grid
        else:
            return scale*grid

    def _xavier_initializer(self, n_in, n_out):
        # Xavier's/He-Rang-Zhen-Sun initialization for layers that are followed by a ReLu
        return tf.compat.v1.random_normal_initializer(mean=0.0, stddev=np.sqrt(2.0 / (n_in)))

    def _spline_Rn_stack(self, xx_centers, xx_B ):
        # Returns a function that for each provided coordinate returns the values of the splines in the shape of the xx_centers
        def _spline_Rn_stack( xx ):
            # n = int(xx.shape[-1])
            # weights_shape = [weights.shape[i] for i in range(len(weights.shape))]
            vectors_to_centers = self._Rn_vectors_to_centers(xx , xx_centers) # For each provided coordinate the distances
            xx_B_sampled = tf.reduce_prod( input_tensor=xx_B( vectors_to_centers ) , axis  = -3 )

            # Old method:
            # dist_to_centers = self._Rn_distance_to_centers(xx , xx_centers) # For each provided coordinate the distances
            # xx_B_sampled = xx_B( dist_to_centers )

            return xx_B_sampled
        return _spline_Rn_stack

    def _Rn_distance_to_centers(self, xx , xx_centers ):
        # xx is an array of arbitrary shape whose last axis is for the coordinates 
        # xx_centers is an array of arbtrary shape whose last axis is for the coordinates of the centers
        
        # Because the tensors are of arbitrary shape (up to the last axis constraint) we need the 
        # following to manipulate teh shapes
        xx_dim = len(xx.shape)
        xx_centers_dim = len(xx_centers.shape)

        # Move the coordinate axis to the front of xx_centers
        _xx_centers = tf.transpose(a=xx_centers,perm=np.roll(range(xx_centers_dim),1))
        # Add empty axes to to the array of coordinates xx
        _xx = tf.reshape(xx,[xx.shape[i] for i in range(xx_dim)] + [1]*(xx_centers_dim-1))

        # The next line generates distance vectors/arrays at grid point
        differences = _xx_centers - _xx
        # The next line computes the N_c distances at each voxel
        dists = tf.sqrt(tf.reduce_sum(input_tensor=differences**2,axis=xx_dim - 1)) # Reduce over the coordinate axis of xx
        # Return the distances
        return dists

    def _Rn_vectors_to_centers(self, xx , xx_centers ):
        # xx is an array of arbitrary shape whose last axis is for the coordinates 
        # xx_centers is an array of arbtrary shape whose last axis is for the coordinates of the centers
        
        # Because the tensors are of arbitrary shape (up to the last axis constraint) we need the 
        # following to manipulate teh shapes
        xx_dim = len(xx.shape)
        xx_centers_dim = len(xx_centers.shape)

        # Move the coordinate axis to the front of xx_centers
        _xx_centers = tf.transpose(a=xx_centers,perm=np.roll(range(xx_centers_dim),1))
        # Add empty axes to to the array of coordinates xx
        _xx = tf.reshape(xx,[xx.shape[i] for i in range(xx_dim)] + [1]*(xx_centers_dim-1))

        # The next line generates distance vectors/arrays at grid point
        differences = _xx_centers - _xx

        # Return the differences
        return differences
















##########################################################################
############################## ConvRnGLayer ##############################
##########################################################################

# Start of lifting_layer class
class ConvRnGLayer(ConvRnRnLayer):
    def __init__(
            self, 
            group,
            inputs,
            N_out,
            kernel_size,
            h_grid,
            xx_basis_size,
            xx_basis_scale,
            xx_basis_type,
            xx_basis_mask,
            stride,
            padding,
            name
            ):
        
        ## Assert and set inputs
        self.kernel_type = 'Rn'
        self._assert_and_set_inputs(group, inputs, N_out, kernel_size, h_grid, xx_basis_size, xx_basis_scale, xx_basis_type, xx_basis_mask, stride, padding, name)

        ## Construct the trainable weights
        self.xx_centers = self.variable_xx_centers_uniform(  constant = True, trainable = False, name = self.name)
        self.N_k = int(self.xx_centers.shape[1]) # Number of basis functions
        self.weights = self.variable_weights( name = self.name )

        ## Compute the output
        self.outputs = self.output()

########################### Assert and set inputs ########################

    # Method overriding:
    def _assert_and_set_inputs(self, group, inputs, N_out, kernel_size, h_grid, xx_basis_size, xx_basis_scale, xx_basis_type, xx_basis_mask, stride, padding, name):
        # Default Rn assertions
        self._assert_and_set_inputs_RnRn(group, inputs, N_out, kernel_size, xx_basis_size, xx_basis_scale, xx_basis_type, xx_basis_mask, stride, padding, name)
        # Specific initialization/assertion
        self.h_grid = self._assert_h_grid( h_grid )

    def _assert_h_grid(self, h_grid ):
        assert (h_grid.grid.get_shape().ndims == 2), "The \"h_grid\" option value should be a grid object with h_grid.grid a tensorflow tensor of dimension 2 (a list of group elements)."
        assert (h_grid.grid.shape[-1] == self.H.n), "The group element specification in \"h_grid\" is not correct. For the current group \"{}\" each group element should be a vector of length {}.".format(self.H.name,self.H.n)
        return h_grid

############################ Compute the output ##########################

    # Method overriding:
    def output(self):
        return self.conv_Rn_G( self.kernel, self.inputs, self.h_grid, self.padding, self.stride )

    def conv_Rn_G(self, kernel, input_tensor, h_grid, padding = 'VALID', xx_stride = 1):
        # Generate the full stack of convolution kernels (all transformed copies)
        N_h = int(h_grid.grid.shape[0])
        kernel_stack = tf.concat( [ kernel(h_grid.grid[i]) for i in range(N_h)], axis = -1) # [X,Y,N_in,N_out x N_h]
        # And apply them all at once
        output = tf.nn.conv2d(
                        input=input_tensor,
                        filter=kernel_stack,
                        strides=[1, xx_stride, xx_stride, 1],
                        padding=padding)
        # Reshape the last channel to create a vector valued RnxH feature map
        N_out = int(int(output.shape[-1])/N_h) # N_flat = N_h*N_out
        # output = tf.stack(tf.split(output,N_out,-1),-1)
        output = tf.stack(tf.split(output,N_h,-1),-2)
        # Return the output
        return output


















##########################################################################
############################### ConvGGLayer ##############################
##########################################################################


# Start of lifting_layer class
class ConvGGLayer(ConvRnGLayer):
    def __init__(
            self, 
            group,
            inputs,
            N_out,
            kernel_size,
            h_grid,
            xx_basis_size,
            xx_basis_scale,
            xx_basis_type,
            xx_basis_mask,
            h_basis_size,
            h_basis_scale,
            h_basis_type,
            h_basis_local,
            input_h_grid,
            stride,
            padding,
            name
            ):
        
        ## Assert and set inputs
        self.kernel_type = 'G'
        self._assert_and_set_inputs(group, inputs, N_out, kernel_size, h_grid, xx_basis_size, xx_basis_scale, xx_basis_type, xx_basis_mask, h_basis_size, h_basis_scale, h_basis_type, h_basis_local, input_h_grid, stride, padding, name)

        ## Construct the trainable weights
        self.xx_centers = self.variable_xx_centers_uniform( constant = True, trainable = False , name = self.name )
        self.N_k = int(self.xx_centers.shape[1]) # Number of basis functions
        self.h_centers = self.variable_h_centers_uniform( constant = True, trainable = False , name = self.name , local = self.h_basis_local)
        self.weights = self.variable_weights( name = self.name )

        ## Compute the output
        self.outputs = self.output()

########################### Assert and set inputs ########################

    # Method overriding:
    def _assert_and_set_inputs(self, group, inputs, N_out, kernel_size, h_grid, xx_basis_size, xx_basis_scale, xx_basis_type, xx_basis_mask, h_basis_size, h_basis_scale, h_basis_type, h_basis_local, input_h_grid, stride, padding, name):
        # Default Rn assertions
        self._assert_and_set_inputs_RnRn(group, inputs, N_out, kernel_size, xx_basis_size, xx_basis_scale, xx_basis_type, xx_basis_mask, stride, padding, name)
        
        # Specific initialization/assertion
        self.h_grid = self._assert_h_grid( h_grid )
        self.input_h_grid = self._assert_input_h_grid( input_h_grid)
        # B-spline specification
        self._assert_h_basis(h_basis_size, h_basis_scale, h_basis_local)
        self.h_basis_type = h_basis_type
        self.h_B_type, self.h_B_degree = self._assert_basis_type( self.h_basis_type )
        self.h_B = bsplines.B(self.h_B_degree, scale = self.h_basis_scale) # Python function that takes as input a coordinate xx in R^n

    # Method overriding:
    def _assert_inputs(self, inputs):
        # input_tensor
        ndims = inputs.get_shape().ndims - 2  # minus batch size and n-channels
        assert (ndims == self.G.n), "The manifold dimension of the input tensor is not {}.".format(self.G.n)
        return inputs

    def _assert_input_h_grid( self, input_h_grid ):
        if (input_h_grid is None):
            return self.h_grid
        else:
            assert (input_h_grid.grid.get_shape().ndims == 2), "The \"input_h_grid\" option value should be a grid object with input_h_grid.grid a tensorflow tensor of dimension 2 (a list of group elements)."
            assert (input_h_grid.grid.shape[-1] == self.H.n), "The group element specification in \"input_h_grid\" is not correct. For the current group \"{}\" each group element should be a vector of length {}.".format(self.H.name,self.H.n)
            return input_h_grid
    
    def _assert_h_basis(self, h_basis_size_in, h_basis_scale_in, h_basis_local_in):
        # Initialize values
        h_basis_size = h_basis_size_in      # Integer
        h_basis_scale = h_basis_scale_in    # List of scalars
        h_basis_local = h_basis_local_in    # True or False
        # Two cases: a local basis or a global basis (centers defined by group.H.grid_local and group.H.grid_global resp.)
        if h_basis_local:
            # LOCAL
            if h_basis_size == None:
                assert isinstance(h_basis_size, int), "When h_basis_local = True you need to specify the nr of basisfunctions with argument \"h_basis_size\" (it should be an integer), currently \"h_basis_size\" is set to {}.".format(h_basis_size)
            else:
                assert isinstance(h_basis_size, int), "When h_basis_local = True you need to specify the nr of basisfunctions with argument \"h_basis_size\" (it should be an integer), currently \"h_basis_size\" is set to {}.".format(h_basis_size)
                if h_basis_scale == None:
                    # LOCAL BASIS | GRID SCALE
                    h_basis_scale = self.input_h_grid.scale
                    h_basis_grid = self.H.grid_local( h_basis_size, h_basis_scale )
                else:
                    # LOCAL BASIS | CUSTOM SCALE
                    h_basis_grid = self.H.grid_local( h_basis_size, h_basis_scale )
        else:
            # GLOBAL
            if h_basis_size == None:
                if h_basis_scale == None:
                    # GLOBAL BASIS | SAME AS INPUT GRID
                    h_basis_scale = self.input_h_grid.scale
                    h_basis_size = self.input_h_grid.N
                    h_basis_grid = self.input_h_grid
                else:
                    raise ValueError('When h_basis_scale is specified, h_basis_size should be specified as well...')
            else:
                # GLOBAL BASIS | CUSTOM BASIS SIZE, CUSTOM SCALING
                # Using the same specs as the h_grid
                h_grid_args = self.h_grid.args.copy()
                h_grid_args.pop('N')
                h_basis_grid = self.H.grid_global( h_basis_size, **h_grid_args)
                if h_basis_scale == None:
                    # GLOBAL BASIS | CUSTOM BASIS SIZE, DENSE
                    h_basis_scale = h_basis_grid.scale
        # Set attributes
        self.h_basis_size = h_basis_size
        self.h_basis_scale = h_basis_scale
        self.h_basis_local = h_basis_local
        self.h_basis_grid = h_basis_grid

    def _assert_h_basis_size(self, h_basis_size_in):
        h_basis_size = h_basis_size_in
        if h_basis_size == None:
            h_basis_size = int(self.input_h_grid.grid.shape[0])
        assert isinstance(h_basis_size, int), "The specified argument \"h_basis_size\" should be an integer."
        return h_basis_size

    def _assert_h_basis_scale(self, h_basis_scale):
        # assert isinstance(h_basis_scale, int) or isinstance(h_basis_scale, float), "The specified argument \"h_basis_scale\" should be a scalar (either int or float)"
        if h_basis_scale == None:
            h_basis_scale = H.grid_scale(int(self.input_h_grid.grid.shape[0])) # Let the scale correspond to the spacing between the grid points
        return float(h_basis_scale)

##################### Construct the trainable weights ####################

    def variable_h_centers_uniform(self, constant = True, trainable = False, name = None , local = False):
        # # At each xx place a h_grid
        # if local:
        #     c_grid = self.h_basis_scale*tf.constant(np.array([np.linspace(-(self.h_basis_size-1)/2, (self.h_basis_size-1)/2,self.h_basis_size)]).transpose(),dtype=tf.float32)
        #     h_grid = self.H.exp(c_grid)
        # else:
        #     h_grid = self.H.grid_global(self.h_basis_size,self.h_basis_scale) # Is a tensorflow array
        h_grid = self.h_basis_grid.grid
        init = tf.expand_dims(h_grid,0) # Now dim = [1,N_h,H.n]
        # repeat every h for every spatial center
        # init = tf.tile(init,[self.xx_basis_size**self.Rn.n,1,1]) # new dim is [N_xx,N_h,H,n]
        init = tf.tile(init,[int(self.N_k/self.h_basis_size),1,1]) # new dim is [N_xx,N_h,H,n]
        init = tf.reshape(init,[-1,self.H.n])
        # Repeat for each input
        init = tf.tile(tf.expand_dims(init,0),[self.N_in,1,1])

        # Variable name
        if not(name is None):
            nameTMP = name + '_h'
        else:
            nameTMP = None

        # Construct centers
        if not(constant):
            shape = None
            h_centers = tf.compat.v1.get_variable( nameTMP, shape, trainable=trainable, initializer=init )
        else:
            h_centers = init # In this case init is already a constant tensor..

        # Return the centers
        return h_centers

############################ Compute the output ##########################

    # Method overriding:
    def output(self):
        return self.conv_G_G( self.kernel, self.inputs, self.h_grid, self.input_h_grid, self.padding, self.stride )
    
    def output_sparse(self):
        return self.conv_G_G_sparse( self.kernel, self.inputs, self.h_grid, self.input_h_grid, self.padding, self.stride )
    
    # Method overriding:
    def kernel(self, h = None):
        # The transformation to apply
        if h is None:
            h = self.H.e
        # The kernel as a function
        Kernel = self.spline_RnxH( self.xx_centers, self.h_centers, self.weights, self.xx_B, self.h_B )
        # Sample the kernel on the (transformed) grid
        h_inv = self.H.inv(h)
        return (1/self.H.det(h))*Kernel( self.H.left_action_on_Rn( h_inv, self.xx_grid ), self.H.prod(h_inv, self.input_h_grid.grid) )

    def conv_G_G(self, kernel, input_tensor, h_grid, input_h_grid, padding = 'VALID', xx_stride = 1):
        # First determine the dimensions
        dims = list(map( int, kernel().shape))
        N_h = int(h_grid.grid.shape[0]) # Target sampling
        N_h_in = int(input_h_grid.grid.shape[0]) # Input sampling
        # Generate the full stack of convolution kernels (all transformed copies)
        kernel_stack = tf.concat( [ kernel(h_grid.grid[i]) for i in range(N_h)], axis = -1) # [X,Y,N_h_in,N_in,N_out x N_h]
        # Reshape input tensor and kernel as if they were Rn tensors
        kernel_stack_as_if_Rn = tf.reshape( kernel_stack, dims[0:self.Rn.n] + [dims[self.Rn.n]*dims[self.Rn.n+1]] + [dims[self.Rn.n+2]*N_h] )
        input_tensor_as_if_Rn = tf.reshape( input_tensor, [(tf.shape(input=input_tensor)[i] if (input_tensor.shape[i] >0 )==None else int(input_tensor.shape[i])) for i in range(0,self.Rn.n+1)] + [-1] )

        # And apply them all at once
        output = tf.nn.conv2d(
                        input=input_tensor_as_if_Rn,
                        filter=kernel_stack_as_if_Rn,
                        strides=[1, xx_stride, xx_stride, 1],
                        padding=padding)
        # Reshape the last channel to create a vector valued RnxH feature map
        N_out = int(int(output.shape[-1])/N_h) # N_flat = N_h*N_out
        output = tf.stack(tf.split(output,N_h,-1),-2)
        # The above includes integration over S1, take discretization into account
        output = (2*np.pi/N_h)*output # TODO: change this to a (Haar) measure or something, this is specific for SE(2)
        # # Return the output
        return output

    def conv_G_G_sparse(self, kernel, input_tensor, h_grid, input_h_grid, padding = 'VALID', xx_stride = 1):
        # This code is still not faster (even though a lot of zero multiplications are avoided)
        # There is a lot of overhead in repeating conv2d
        # First determine the dimensions
        dims = list(map( int, kernel().shape))
        N_h = int(h_grid.grid.shape[0]) # Target sampling
        N_h_in = int(input_h_grid.grid.shape[0]) # Input sampling

        # Reshape the input tensor (merge H and channel axis)
        input_tensor_as_if_Rn = tf.reshape( input_tensor, [tf.shape(input=input_tensor)[0]] + [tf.shape(input=input_tensor)[i] for i in range(1,self.Rn.n+1)] + [-1] ) # [B,X,Y,H*Cin]

        # Apply the kernels for each transformation h
        results_list = [None]*N_h
        for i in range(N_h):
            kernel_h = kernel(h_grid.grid[i]) #[X,Y,H,Cin,Cout]
            kernel_h_as_if_Rn = tf.reshape( kernel_h, dims[0:self.Rn.n] + [-1] + [dims[-1]] ) #[X,Y,H*Cin,Cout]

            # Reduce (throw away the zero h_indices, which can be there due to dilated or localized bases)
            keep_indices = tf.reshape(tf.compat.v1.where(tf.greater(tf.reduce_sum(input_tensor=tf.abs(kernel_h_as_if_Rn),axis=list(range(self.Rn.n))+[-1]),0.)),[-1])
            kernel_h_as_if_Rn_red = tf.gather(kernel_h_as_if_Rn,keep_indices,axis=self.Rn.n)
            input_tensor_as_if_Rn_red = tf.gather(input_tensor_as_if_Rn,keep_indices,axis=self.Rn.n+1)

            # Apply the 2D convolution
            result = tf.nn.conv2d( # [B,X,Y,Cout]
                            input=input_tensor_as_if_Rn_red,
                            filter=kernel_h_as_if_Rn_red,
                            strides=[1, xx_stride, xx_stride, 1],
                            padding=padding)
            results_list[i] = tf.expand_dims(result,-2) # [B,X,Y,1,C]
        output = tf.concat( results_list, axis = -2) # [B,X,Y,H,C]

        # Return the output
        return output

    def spline_H( self, h_centers, weights, h_B):
        def spline_H( h ):
            n = H.n
            h_B_sampler = self._spline_H_stack( h_centers, h_B )
            h_B_sampled = h_B_sampler( h )

            # Reshape the weights
            weights_shape = [int(N_i) for N_i in weights.shape]
            _weights = tf.reshape(weights, [1]*(n) + weights_shape)
            h_B_sampled = tf.expand_dims(h_B_sampled,-1)
            h_B_recon = tf.reduce_sum(input_tensor=h_B_sampled*_weights,axis=-2) # Sum over the splines

            return h_B_recon
        return spline_H

    def spline_RnxH( self,  xx_centers, h_centers, weights, xx_B, h_B):
        def spline_RnxH( xx, h ):
            n = self.Rn.n + self.H.n
            g_B_sampler = self._spline_RnxH_stack( xx_centers, h_centers, xx_B, h_B)
            g_B_sampled = g_B_sampler( xx, h )

            # Reshape the weights
            weights_shape = [int(N_i) for N_i in weights.shape]
            _weights = tf.reshape(weights, [1]*(n) + weights_shape)
            g_B_sampled = tf.expand_dims(g_B_sampled,-1)
            g_B_recon = tf.reduce_sum(input_tensor=g_B_sampled*_weights,axis=-2) # Sum over the splines

            return g_B_recon
        return spline_RnxH

######################### Private helper functions #######################

    def _spline_H_stack( self, h_centers, h_B ):
        # Returns a function that for each provided coordinate returns the values of the splines in the shape of the xx_centers
        def _spline_H_stack( h ):
            dist_to_centers = self._H_distance_to_centers(h , h_centers ) # For each provided coordinate the distances
            h_B_sampled = h_B( dist_to_centers )
            return h_B_sampled
        return _spline_H_stack

    def _spline_RnxH_stack( self, xx_centers, h_centers, xx_B, h_B ):
        def _spline_RnxH_stack( xx, h ):

            xx_B_sampler = self._spline_Rn_stack( xx_centers, xx_B )
            h_B_sampler = self._spline_H_stack( h_centers, h_B )

            # xx, h = G.xx_h(g)
            xx_B_sampled = xx_B_sampler( xx )
            h_B_sampled = h_B_sampler( h )

            # Outer product
            xx_shape = [int(N_i) for N_i in xx.shape[0:-1]]
            h_shape = [int(N_i) for N_i in h.shape[0:-1]]
            centers_shape = [int(N_i) for N_i in xx_centers.shape[0:-1]]
            
            # B
            xx_B_sampled = tf.reshape(xx_B_sampled,xx_shape+ [1]*self.H.n + centers_shape)
            h_B_sampled = tf.reshape(h_B_sampled, [1]*self.Rn.n + h_shape + centers_shape)
            g_B_sampled = xx_B_sampled*h_B_sampled
            return g_B_sampled
        return _spline_RnxH_stack

    def _H_distance_to_centers( self, h, h_centers ):
        h_shape = [int(N_i) for N_i in h.shape]
        h_shape_flat_len = np.prod(h_shape)
        h_shape_flat = [h_shape_flat_len,1] # TODO: Danger!! This one is for number of group elements which is 1 for SO(2), but could be different for other groups

        h_centers_shape = [int(N_i) for N_i in h_centers.shape]

        h_flat = tf.reshape(h, h_shape_flat)
        h_distances = tf.stack( [self.H.dist( h_flat[i] , h_centers ) for i in range(h_shape_flat_len) ])
        h_distances = tf.reshape( h_distances, h_shape[0:-1] + h_centers_shape[0:-1])
        return h_distances

