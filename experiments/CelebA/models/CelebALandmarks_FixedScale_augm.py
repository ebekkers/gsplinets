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
layers = gsplinets.layers('R2R+')


class CelebALandmarks_FixedScale_augm:

    def __init__(self, args = {'N_h':1,'N_k_h':1,'N_c':32,'h_range':2,'h_origin_at':'center','h_kernel_type':'dense','scale_augm_range':1.2}):

        ## Check arguments
        N_h = int(args['N_h'])			# 1,	4,	8,	16
        N_k_h = int(args['N_k_h'])
        N_c = int(args['N_c'])			# 36,	18,	13,	19
        h_kernel_type = args['h_kernel_type'] # 'dense', 'atrous', 'localized'
        h_range = float(args['h_range'])
        self.scale_augm_range = float(args['scale_augm_range'])
        h_origin_at = args['h_origin_at']  # 'min','max','center'

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
        padding = 'SAME'
        activation = tf.nn.relu
        
        # Grid settings on H
        h_grid = layers.H.grid_global(N_h,h_range)
        if h_origin_at == 'min':
            xx_basis_scale = 1
        elif h_origin_at == 'max':
            xx_basis_scale = 1/h_range
        elif h_origin_at == 'center':
            xx_basis_scale = 1/np.exp(0.5*np.log(h_range))
        h_grid_min = 1
        h_grid_max = h_range

        # Kernel size (based on maximum scaling)
        # xx_basis_scale = 1/h_grid_min # Minimal scale (should always be one)
        if N_h == 1:
            xx_basis_scale = h_range
            kernel_size1 = int( ((xx_basis_scale*(1+1))//2)*2 + 1 )
            kernel_size3 = int( ((xx_basis_scale*(3+1))//2)*2 + 1 )
            kernel_size5 = int( ((xx_basis_scale*(5+1))//2)*2 + 1 )
        else:
            kernel_size1 = int( ((h_grid_max*xx_basis_scale*(1+1))//2)*2 + 1 )
            kernel_size3 = int( ((h_grid_max*xx_basis_scale*(3+1))//2)*2 + 1 )
            kernel_size5 = int( ((h_grid_max*xx_basis_scale*(5+1))//2)*2 + 1 )

        # Basis settings on H (use defaults except for the spline scale)
        # N_k_h = 1 # Nr of basis functions to use
        if h_kernel_type == 'localized':
            local = True
            if N_h == 1:
                h_basis_scale = 1
            else:
                h_basis_scale = np.log(h_range)/(N_h)
        elif h_kernel_type == 'atrous':
            local = False
            if N_h == 1:
                h_basis_scale = 1
            else:
                h_basis_scale = np.log(h_range)/(N_h)
        elif h_kernel_type == 'dense':
            local = False
            if N_k_h == 1:
                h_basis_scale = 1
            else:
                h_basis_scale = np.log(h_range)/(N_k_h)
        
        ## Place holders
        self.x_ph = tf.compat.v1.placeholder( dtype = tf.float32, shape = [None,128,128,3] ) # CIFAR RGB input images (3 channel feature maps)
        self.y_ph = tf.compat.v1.placeholder( dtype = tf.float32, shape = [None,128,128,5] )
        self.is_training_ph = tf.compat.v1.placeholder( dtype = tf.bool, shape = None)

        # The inputs in the right range (0 to 1)
        x = self.x_ph/255.
        y = self.y_ph/255.

        # Data augmentation (y_target should be augmented accordingly...)
        xy = tf.concat([x,y],-1)
        xy = tf.cond( pred=self.is_training_ph, true_fn=lambda: tf.map_fn(lambda img: self.augment(img), xy), false_fn=lambda: xy )
        x = xy[...,0:3] # RGB input
        y = xy[...,3:]  # The remaing heat maps

        self.x = x
        self.y = y


		## The layers
        # Input
        tensor_out = x # Start with 128 x 128 patches

        # ----------------------------
        # Block 1: 128->128
        tensor_out = self.layer_ConvRnG( tensor_out, N_c, kernel_size5, h_grid, xx_basis_size=5, xx_basis_scale = xx_basis_scale, name = 'layer_1', padding = padding, batch_normalization = batch_normalization, is_training = self.is_training_ph, activation = activation)
        tensor_out = self.layer_ConvGG( tensor_out, N_c, kernel_size5, h_grid, xx_basis_size=5, xx_basis_scale = xx_basis_scale, h_basis_size = N_k_h, h_basis_scale = h_basis_scale, name = 'layer_2', padding = padding, batch_normalization = batch_normalization, is_training = self.is_training_ph, activation = activation, h_basis_local = local)
        tensor_out = self.layer_ConvGG( tensor_out, N_c, kernel_size5, h_grid, xx_basis_size=5, xx_basis_scale = xx_basis_scale, h_basis_size = N_k_h, h_basis_scale = h_basis_scale, name = 'layer_3', padding = padding, batch_normalization = batch_normalization, is_training = self.is_training_ph, activation = activation, h_basis_local = local)
        tensor_out = tf.reduce_mean(input_tensor=tensor_out,axis=3)
        tensor_out_block_1 = tensor_out
        tensor_out = layers.max_pooling_Rn(  tensor_out, 2, 2,  padding = 'SAME' )

        # ----------------------------
        # Block 2: 64->64
        tensor_out = self.layer_ConvRnG( tensor_out, N_c, kernel_size5, h_grid, xx_basis_size=5, xx_basis_scale = xx_basis_scale, name = 'layer_4', padding = padding, batch_normalization = batch_normalization, is_training = self.is_training_ph, activation = activation)
        tensor_out = self.layer_ConvGG( tensor_out, N_c, kernel_size5, h_grid, xx_basis_size=5, xx_basis_scale = xx_basis_scale, h_basis_size = N_k_h, h_basis_scale = h_basis_scale, name = 'layer_5', padding = padding, batch_normalization = batch_normalization, is_training = self.is_training_ph, activation = activation, h_basis_local = local)
        tensor_out = self.layer_ConvGG( tensor_out, N_c, kernel_size5, h_grid, xx_basis_size=5, xx_basis_scale = xx_basis_scale, h_basis_size = N_k_h, h_basis_scale = h_basis_scale, name = 'layer_6', padding = padding, batch_normalization = batch_normalization, is_training = self.is_training_ph, activation = activation, h_basis_local = local)
        tensor_out = tf.reduce_mean(input_tensor=tensor_out,axis=3)
        tensor_out = tf.image.resize(tensor_out,[128,128])
        tensor_out_block_2 = tensor_out

        # ----------------------------
        # Block 3: 128->128 (to output)
        tensor_out = tf.concat([tensor_out_block_1,tensor_out_block_2],axis=-1)
        tensor_out = self.layer_ConvRnRn( tensor_out, 32, 3, name = 'layer_7', padding = padding, batch_normalization = batch_normalization, is_training = self.is_training_ph, activation = activation)
        tensor_out = self.layer_ConvRnRn( tensor_out, 32, 3, name = 'layer_8', padding = padding, batch_normalization = batch_normalization, is_training = self.is_training_ph, activation = activation)
        tensor_out = self.layer_ConvRnRn( tensor_out, 64, 1, name = 'layer_9', padding = padding, batch_normalization = batch_normalization, is_training = self.is_training_ph, activation = activation)
        tensor_out = self.layer_ConvRnRn( tensor_out, 5, 1, name = 'layer_10', padding = padding, batch_normalization = False, is_training = self.is_training_ph, activation = tf.identity)

        print('output layer: ',tensor_out)

        # The output layer
        self.logits = tensor_out # new shape is [B,128,128,5]
        self.yy = tf.nn.sigmoid(self.logits)
        self.yy_B = blur2d(self.yy, 1.)
        self.yy_ij = argmax2d(self.yy_B)
        
        # Losses
        # Use reduction="weighted_mean" to make the loss batch size independent
        loss_per_pixel = tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=self.logits)
        self.loss = tf.reduce_mean(input_tensor=loss_per_pixel*heatmap2weightmap(y))
        # self.loss = tf.losses.softmax_cross_entropy(onehot_labels = y, logits = self.logits)
        # Only compute the l2 loss over the encoding part
        list_of_l2_losses = [ tf.reduce_sum(input_tensor=self.all_kernels[key]**2) for key in ['layer_1','layer_2','layer_3','layer_4','layer_5','layer_6','layer_7','layer_8','layer_9']]
        self.loss_l2 = tf.add_n(list_of_l2_losses)

        # # Other metrics
        # correct_predictions = tf.equal( self.predictions, self.labels_ph )
        # self.success_rate = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        # self.number_of_errors = tf.reduce_sum(1 - tf.cast(correct_predictions, tf.float32))


    def augment( self, x):
        x_augm = x
        # Geometric:
        # k = tf.random_uniform(dtype=tf.int32, minval=0, maxval=4, shape=())
        # x_augm = tf.image.rot90(x_augm,k=k)
        x_augm = self.random_zoom( x_augm )

        return x_augm
    
    def random_zoom( self, x: tf.Tensor) -> tf.Tensor:
        """Zoom augmentation

        Args:
            x: Image

        Returns:
            Augmented image
        """

        # Generate 20 crop settings, ranging from a 1% to 20% crop.
        # scales = list(np.arange(1/self.h_range - 0.01, 1.0, 0.01))
        scale_range = self.scale_augm_range
        scales = list(np.arange(1/scale_range - 0.01, 1.0, 0.01))
        boxes = np.zeros((len(scales), 4))

        for i, scale in enumerate(scales):
            x1 = y1 = 0.5 - (0.5 * scale)
            x2 = y2 = 0.5 + (0.5 * scale)
            boxes[i] = [x1, y1, x2, y2]

        def random_crop(img):
            # Create different crops for an image
            crops = tf.image.crop_and_resize([img], boxes=boxes, box_indices=np.zeros(len(scales)), crop_size=(128, 128))
            # Return a random crop
            return crops[tf.random.uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]


        choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

        # Only apply cropping 50% of the time
        return tf.cond(pred=choice < 0.5, true_fn=lambda: x, false_fn=lambda: random_crop(x))





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
            xx_basis_mask = False,
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
            xx_basis_mask = False,
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
            xx_basis_mask = False,
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

def argmax2d(tensor): # For every image and every channel return the argmax, outputshape [B,D,2]

    # input format: BxHxWxD
    assert len(tensor.get_shape()) == 4

    # flatten the Tensor along the height and width axes
    flat_tensor = tf.reshape(tensor, (tf.shape(input=tensor)[0], -1, tf.shape(input=tensor)[3]))

    # argmax of the flat tensor
    argmax = tf.cast(tf.argmax(input=flat_tensor, axis=1), tf.int32)

    # convert indexes into 2D coordinates
    argmax_x = argmax // tf.shape(input=tensor)[2]
    argmax_y = argmax % tf.shape(input=tensor)[2]

    # stack and return 2D coordinates
    argmax_xy = tf.stack((argmax_x, argmax_y), axis=1)
    argmax_xy = tf.transpose(a=argmax_xy,perm=[0,2,1])
    return argmax_xy

def gaussian_kernel(size: int,
                    mean: float,
                    std: float,
                   ):
    """Makes 2D gaussian Kernel for convolution."""
    d = tf.compat.v1.distributions.Normal(mean, std)
    vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))
    gauss_kernel = tf.einsum('i,j->ij', vals, vals)
    return gauss_kernel / tf.reduce_sum(input_tensor=gauss_kernel)

def blur2d(scalar_image_stack, std):
    # Make Gaussian Kernel with desired specs.
    gauss_kernel = gaussian_kernel( int(round(2*2*std)), 0., std )
    
    # Expand dimensions of `gauss_kernel` for `tf.nn.conv2d` signature.
    gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]
    gauss_kernel = tf.concat([gauss_kernel]*int(scalar_image_stack.shape[-1]),axis = -2)

    # Convolve.
    result = tf.nn.depthwise_conv2d(input=scalar_image_stack, filter=gauss_kernel, strides=[1, 1, 1, 1], padding="SAME")
    return result

def heatmap2weightmap(heatmap): #Input shape [B,W,H,C]
    numpos = tf.reduce_sum(input_tensor=tf.reduce_sum(input_tensor=heatmap,axis=1),axis=1) # [B,C]
    numneg = tf.reduce_sum(input_tensor=tf.reduce_sum(input_tensor=1 - heatmap,axis=1),axis=1) # [B,C]
    eps = 1e-6
    pfrac = numpos/(numpos + numneg) # [B,C]
    nfrac = numneg/(numpos + numneg) # [B,C]
    weightmap = nfrac[:,tf.newaxis,tf.newaxis,:]*heatmap + pfrac[:,tf.newaxis,tf.newaxis,:]*(1-heatmap) # [B,W,H,C]
    correctionfactor = 1/(eps + tf.reduce_mean(input_tensor=tf.reduce_mean(input_tensor=weightmap,axis=1),axis=1)) # [B,C]
    weightmap = weightmap*correctionfactor[:,tf.newaxis,tf.newaxis,:] # [B,W,H,C]
    return weightmap
