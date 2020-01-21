# Model classes that are compatible with the training.py should contain the following:
# # Place holders:
# 	model.inputs_ph
# 	model.labels_ph
# 	model.is_training_ph
# # Metrics
# 	model.loss
# 	model.loss_l2
# 	model.success_rate
#	model.number_of_errors


# Core libraries
import numpy as np
import tensorflow as tf

# G-splinet specific
import importlib
import gsplinets_tf as gsplinets
layers = gsplinets.layers('SE2')


class PCAM_SE2_norotaugm:

    def __init__(self, args = {'N_h':1,'N_k_h':1,'N_c':32,'h_kernel_type':'dense'}):

        ## Check arguments
        N_h = int(args['N_h'])			# 1,	4,	8,	16
        N_k_h = int(args['N_k_h'])
        N_c = int(args['N_c'])			# 36,	18,	13,	19
        h_kernel_type = args['h_kernel_type'] # 'dense', 'atrous', 'localized'

    	## Some attributes
        self.all_weights={}
        self.all_kernels={}
        self.all_inputs={}
        self.all_outputs={}
        self.all_bn_betas={}
        self.all_bn_gammas={}
        self.all_biases={}

        ## Setings
        # Base settings
        batch_normalization = True
        padding = 'VALID'
        activation = tf.nn.relu
        
        # Grid settings on H
        h_grid = layers.H.grid_global(N_h)
        
        # Basis settings on H (use defaults except for the spline scale)
        # N_k_h = 1 # Nr of basis functions to use
        if h_kernel_type == 'localized':
            local = True
            h_basis_scale = 2*np.pi/N_h
            print('local')
        elif h_kernel_type == 'atrous':
            local = False
            h_basis_scale = 2*np.pi/N_h
            print('atrous')
        elif h_kernel_type == 'dense':
            local = False
            h_basis_scale = 2*np.pi/N_k_h
            print('dense')

        
        ## Place holders
        self.inputs_ph = tf.compat.v1.placeholder( dtype = tf.float32, shape = [None,96,96,3] ) # CIFAR RGB input images (3 channel feature maps)
        self.labels_ph = tf.compat.v1.placeholder( dtype = tf.int32, shape = [None,] )
        self.is_training_ph = tf.compat.v1.placeholder( dtype = tf.bool, shape = None)


        ## The layers
        tensor_out = self.inputs_ph[:,4:-4,4:-4,:]/255. # Start with 88 x 88 patches

        # tensor_out_augm = tf.map_fn(lambda img: self.augment(img), tensor_out) # Move to inside tf.cond(...)
        # tensor_out_augm = self.augment( tensor_out )
        tensor_out = tf.cond( pred=self.is_training_ph, true_fn=lambda: tf.map_fn(lambda img: self.augment(img), tensor_out), false_fn=lambda: tensor_out )

        # ----------------------------
        # 1: 88 -> 84
        tensor_out = self.layer_ConvRnG( tensor_out, N_c, 5, h_grid, name = 'layer_1', padding = padding, batch_normalization = batch_normalization, is_training = self.is_training_ph, activation = activation)
        # Max-pool: 84 -> 42
        tensor_out = layers.max_pooling_Rn(  tensor_out, 2, 2,  padding = 'SAME' )

        # ----------------------------
        # 2: 42 -> 38
        tensor_out = self.layer_ConvGG( tensor_out, N_c, 5, h_grid, h_basis_size = N_k_h, h_basis_scale = h_basis_scale, name = 'layer_2', padding = padding, batch_normalization = batch_normalization, is_training = self.is_training_ph, activation = activation, h_basis_local = local)
        # Max-pool: 38 -> 19
        tensor_out = layers.max_pooling_Rn(  tensor_out, 2, 2,  padding = 'SAME' )

        # ----------------------------
        # 3: 19 -> 15
        tensor_out = self.layer_ConvGG( tensor_out, N_c, 5, h_grid, h_basis_size = N_k_h, h_basis_scale = h_basis_scale, name = 'layer_3', padding = padding, batch_normalization = batch_normalization, is_training = self.is_training_ph, activation = activation, h_basis_local = local)
        # Max-pool: 15 -> 5
        tensor_out = layers.max_pooling_Rn(  tensor_out, 3, 3,  padding = 'SAME' )

        # ----------------------------
        # 4: 5 -> 1
        tensor_out = self.layer_ConvGG( tensor_out, N_c, 5, h_grid, h_basis_size = N_k_h, h_basis_scale = h_basis_scale, name = 'layer_4', padding = padding, batch_normalization = batch_normalization, is_training = self.is_training_ph, activation = activation, h_basis_local = local)
        # 5: 1 -> 1
        tensor_out = self.layer_ConvGG( tensor_out, 64, 1, h_grid, h_basis_size = N_k_h, h_basis_scale = h_basis_scale, name = 'layer_5', padding = padding, batch_normalization = batch_normalization, is_training = self.is_training_ph, activation = activation, h_basis_local = local)
        # Project to 2D plane
        tensor_out = tf.reduce_max(input_tensor=tensor_out,axis=3)

        # ----------------------------
        # 6: 1 -> 1
        tensor_out = self.layer_ConvRnRn( tensor_out, 16, 1, name = 'layer_6', padding = padding, batch_normalization = batch_normalization, is_training = self.is_training_ph, activation = activation)

        # ----------------------------
        # 7: 1 -> 1
        tensor_out = self.layer_ConvRnRn( tensor_out, 2, 1, name = 'layer_7', padding = padding, batch_normalization = False, is_training = self.is_training_ph, activation = tf.identity)

        print('output layer: ',tensor_out)

        # The output layer
        self.logits = tensor_out[:,0,0,:] # tensor_out at this point has shape [B,1,1,2]
        self.predictions = tf.argmax(input=self.logits, axis=1, output_type=tf.int32)
        self.probabilities = tf.nn.softmax(self.logits)
        
        # Losses
        # Use reduction="weighted_mean" to make the loss batch size independent
        self.loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=self.labels_ph, logits=self.logits,reduction="weighted_mean")
        # Only compute the l2 loss over the encoding part
        # Normalized/averaged l2 losses (l2 norm normalized by area: so average of squared weights instead of sum/integral)
        # Sum over the weights, average over the domain
        # list_of_l2_losses = [ tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(self.all_kernels[key]**2,axis=-1),-1)) for key in ['layer_1','layer_2','layer_3','layer_4','layer_5']]
        # Average over the weights and domain:
        # list_of_l2_losses = [ tf.reduce_mean(self.all_kernels[key]**2) for key in ['layer_1','layer_2','layer_3','layer_4','layer_5']]
        # self.loss_l2 = tf.add_n(list_of_l2_losses)/len(list_of_l2_losses)
        # Sum over the weights and domain (standard):
        list_of_l2_losses = [ tf.reduce_sum(input_tensor=self.all_kernels[key]**2) for key in ['layer_1','layer_2','layer_3','layer_4','layer_5']]
        self.loss_l2 = tf.add_n(list_of_l2_losses)

        # # Other metrics
        correct_predictions = tf.equal( self.predictions, self.labels_ph )
        self.success_rate = tf.reduce_mean(input_tensor=tf.cast(correct_predictions, tf.float32))
        self.number_of_errors = tf.reduce_sum(input_tensor=1 - tf.cast(correct_predictions, tf.float32))


    def augment( self, input_tensor ):
        output_tensor = input_tensor
        # Geometric:
        # output_tensor = tf.image.rot90(output_tensor,k=tf.random_uniform(dtype=tf.int32, minval=0, maxval=4, shape=()))
        output_tensor = tf.image.random_flip_left_right(output_tensor)
        # Color
        output_tensor = tf.image.random_brightness(output_tensor, 64/255)
        output_tensor = tf.image.random_saturation(output_tensor, 0.75, 1.25)
        output_tensor = tf.image.random_hue(output_tensor,0.04)
        output_tensor = tf.image.random_contrast(output_tensor,0.5,1.5)
        output_tensor = tf.clip_by_value(output_tensor, 0.0, 1.0)
        return output_tensor





    def layer_ConvRnRn( # The same arguments as the standard layers from gsplinets, except for additional options s.a. batch_normalization and activation function
    	    self,
            # Required arguments
            inputs,                 # Should be a tensor of dimension self.Rn.n + 2 (Batch axis, Rn.n spatial axes, Feature channel axis)
            N_out,                  # Number of output channels
            kernel_size,            # Kernel size (integer)
            # Optional basis function related arguments
            xx_basis_size=None,     # Nr of spline basis functions in each axis
            xx_basis_scale=1,       # Scale of the splines (spacing between the splines)
            xx_basis_type = 'B_2',  # Type of B-Spline basis function
            xx_basis_mask = True,
            # Optional generic arguments
            stride=1,               # Spatial stride in the convolution
            padding='VALID',        # Padding type
            name=None,              # Name of generated tensorflow variables
            # Options specific for network construction
            batch_normalization = False, # If False then a trainable bias is added after the convolution
            activation = tf.nn.relu,	 # Activation function
            is_training = False			 # Whether or not in training mode (used in moment optimization in batch_normalization)
            ):             

        # Basic settings
        if (name is None):  # Then set name as random unique string of length 6
            self.name = uuid.uuid4().hex[:6].upper()
        else:
            self.name = name

        # The main layer
        with tf.compat.v1.name_scope(self.name):
            l = layers.ConvRnRn( inputs, N_out, kernel_size, xx_basis_size, xx_basis_scale, xx_basis_type, xx_basis_mask, stride, padding, self.name)

            # Add some of the attributes to the model collections
            self.all_kernels[name]=l.kernel()
            self.all_inputs[name] = l.inputs
            self.all_weights[name] = l.weights

            # The actual layer
            outputs = l.outputs # Perfoms separable convolutions (is much slower though..)
            # Batch normalization
            if batch_normalization:
                bn_layer = tf.compat.v1.layers.BatchNormalization(momentum = 0.9,trainable=True)
                outputs = bn_layer(outputs,training=is_training)
                self.all_bn_betas[name] = bn_layer.beta
                self.all_bn_gammas[name] = bn_layer.gamma
            else:
                self.all_biases[name] = tf.compat.v1.get_variable( 
                                name+'_biases',
                                [1, 1, 1, N_out], 
                                initializer=tf.compat.v1.constant_initializer(value=0.01))
                outputs = outputs + self.all_biases[name]
            # Activation
            outputs = activation(outputs)

            # Add to collection
            self.all_outputs[name] = outputs

            # Return outputs
            return outputs

    def layer_ConvRnG( # The same arguments as the standard layers from gsplinets, except for additional options s.a. batch_normalization and activation function
    	    self,
            # Required arguments
            inputs,                 # Should be a tensor of dimension self.Rn.n + 2 (Batch axis, Rn.n spatial axes, Feature channel axis)
            N_out,                  # Number of output channels
            kernel_size,            # Kernel size (integer)
            h_grid,                 # The grid of H on which to compute the output
            # Optional basis function related arguments
            xx_basis_size=None,     # Nr of spline basis functions in each axis
            xx_basis_scale=1,       # Scale of the splines (spacing between the splines)
            xx_basis_type = 'B_2',  # Type of B-Spline basis function
            xx_basis_mask = True,
            # Optional generic arguments
            stride=1,               # Spatial stride in the convolution
            padding='VALID',        # Padding type
            name=None,              # Name of generated tensorflow variables
            # Options specific for network construction
            batch_normalization = False, # If False then a trainable bias is added after the convolution
            activation = tf.nn.relu,	 # Activation function
            is_training = False			 # Whether or not in training mode (used in moment optimization in batch_normalization)
            ):    

        # Basic settings
        if (name is None):  # Then set name as random unique string of length 6
            self.name = uuid.uuid4().hex[:6].upper()
        else:
            self.name = name

        # The main layer
        with tf.compat.v1.name_scope(self.name):
            l = layers.ConvRnG( inputs, N_out, kernel_size, h_grid, xx_basis_size, xx_basis_scale, xx_basis_type, xx_basis_mask, stride, padding, self.name)

            # Add some of the attributes to the model collections
            self.all_kernels[name]=l.kernel()
            self.all_inputs[name] = l.inputs
            self.all_weights[name] = l.weights

           # The actual layer
            outputs = l.outputs # Perfoms separable convolutions (is much slower though..)
            # Batch normalization
            if batch_normalization:
                bn_layer = tf.compat.v1.layers.BatchNormalization(momentum = 0.9,trainable=True)
                outputs = bn_layer(outputs,training=is_training)
                self.all_bn_betas[name] = bn_layer.beta
                self.all_bn_gammas[name] = bn_layer.gamma
            else:
                self.all_biases[name] = tf.compat.v1.get_variable( 
                                name+'_biases',
                                [1, 1, 1, 1, N_out], 
                                initializer=tf.compat.v1.constant_initializer(value=0.01))
                outputs = outputs + self.all_biases[name]
            # Activation
            outputs = activation(outputs)

            # Add to collection
            self.all_outputs[name] = outputs

            # Return outputs
            return outputs


    def layer_ConvGG( # The same arguments as the standard layers from gsplinets, except for additional options s.a. batch_normalization and activation function
    	    self,
            # Required arguments
            inputs,                 # Should be a tensor of dimension self.Rn.n + 2 (Batch axis, Rn.n spatial axes, Feature channel axis)
            N_out,                  # Number of output channels
            kernel_size,            # Kernel size (integer)
            h_grid,                 # The grid of H on which to compute the output
            # Optional basis function related arguments
            xx_basis_size=None,     # Nr of spline basis functions in each axis
            xx_basis_scale=1,       # Scale of the splines (spacing between the splines)
            xx_basis_type = 'B_2',  # Type of B-Spline basis function
            xx_basis_mask = True,
            h_basis_size=None,      # Nr of spline basis functions to cover H
            h_basis_scale=1,        # Scale of the splines (spacing between the splines)
            h_basis_type = 'B_2',   # Type of B-Spline basis function
            h_basis_local = True,  # Whether basis is uniformly covering the group (False) or is localized (True)
            # Optional grid re-sampling related arguments
            input_h_grid = None,    # In case the input grid is not the same as the intended output grid (see h_grid parameter)
            # Optional generic arguments
            stride=1,               # Spatial stride in the convolution
            padding='VALID',        # Padding type
            name=None,              # Name of generated tensorflow variables
            # Options specific for network construction
            batch_normalization = False, # If False then a trainable bias is added after the convolution
            activation = tf.nn.relu,	 # Activation function
            is_training = False			 # Whether or not in training mode (used in moment optimization in batch_normalization)
            ):    

        # Basic settings
        if (name is None):  # Then set name as random unique string of length 6
            self.name = uuid.uuid4().hex[:6].upper()
        else:
            self.name = name

        # The main layer
        with tf.compat.v1.name_scope(self.name):
            l = layers.ConvGG( inputs, N_out, kernel_size, h_grid, xx_basis_size, xx_basis_scale, xx_basis_type, xx_basis_mask, h_basis_size, h_basis_scale, h_basis_type, h_basis_local, input_h_grid, stride, padding, self.name)

            # Add some of the attributes to the model collections
            self.all_kernels[name]=l.kernel()
            self.all_inputs[name] = l.inputs
            self.all_weights[name] = l.weights

           # The actual layer
            outputs = l.outputs # Perfoms separable convolutions (is much slower though..)
            # Batch normalization
            if batch_normalization:
                bn_layer = tf.compat.v1.layers.BatchNormalization(momentum = 0.9,trainable=True)
                outputs = bn_layer(outputs,training=is_training)
                self.all_bn_betas[name] = bn_layer.beta
                self.all_bn_gammas[name] = bn_layer.gamma
            else:
                self.all_biases[name] = tf.compat.v1.get_variable( 
                                name+'_biases',
                                [1, 1, 1, 1, N_out], 
                                initializer=tf.compat.v1.constant_initializer(value=0.01))
                outputs = outputs + self.all_biases[name]
            # Activation
            outputs = activation(outputs)

            # Add to collection
            self.all_outputs[name] = outputs

            # Return outputs
            return outputs
