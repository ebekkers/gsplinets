# Some basic packeges
import numpy as np
import math as m
import time
import logging
import argparse
import ast
import shutil
import sklearn.metrics
from importlib.machinery import SourceFileLoader

# Add the core library to the system path
import os,sys
splinets_source =  os.path.join(os.getcwd(),'..','..')
if splinets_source not in sys.path:
    sys.path.append(splinets_source)

# Tensorflow
import tensorflow as tf

# Import cifar10 functions
from pcam import get_pcam_data

from tensorflow.python.client import device_lib
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def train_arg_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str,
                        default='./data')
    parser.add_argument('--resultdir_root', type=str,
                        default='./results/')
    parser.add_argument('--resultdir', type=str,
                        default=None)

    parser.add_argument('--modelfn', type=str,
                        default='./models/PCAM_SE2.py')
    parser.add_argument('--trainfn', type=str,
                        default='train_all.npz')
    parser.add_argument('--valfn', type=str,
                        default='test.npz')

    parser.add_argument('--epochs', type=int,
                        default=100)
    parser.add_argument('--batchsize', type=int,
                        default=32)
    parser.add_argument('--steps_per_epoch', type=int,
                        default=2048)

    parser.add_argument('--net_kwargs', type=ast.literal_eval,
                        default={'N_h':8,'N_k_h':8,'N_c':14,'h_kernel_type':'dense'})

    parser.add_argument('--weight_decay', type=float,
                        default=0.0005)
    parser.add_argument('--l2_loss', type=float,
                        default=0.0) 

    parser.add_argument('--lr', type=float,
                        default=0.01)
    parser.add_argument('--lr_decay_schedule', type=str,
                        default='12-24-36-48-60-72-84-96-108-110-112-114-116-118')
    parser.add_argument('--lr_decay_factor', type=float,
                        default=0.5)  # default 0 means no learning rate decay

    return parser

def create_result_dir(resultdir, modelfn):
    result_dir = os.path.join(resultdir, os.path.basename(modelfn).split('.')[0], time.strftime('r%Y_%m_%d_%H_%M_%S'))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Create init file so we can import the model module
    # f = open(os.path.join(result_dir, '__init__.py'), 'wb')
    # f.close()

    # Create directories for exporting the model
    os.makedirs(os.path.join(result_dir,'tf_model_first'))
    os.makedirs(os.path.join(result_dir,'tf_model_last'))
    os.makedirs(os.path.join(result_dir,'tf_model_best'))

    return result_dir

def get_model(result_dir, modelfn, net_kwargs):
    model_fn = os.path.basename(modelfn)
    model_name = model_fn.split('.')[0]
    module = SourceFileLoader(model_name, modelfn).load_module()
    net = getattr(module, model_name)

    # Copy model definition and this train script to the result dir
    logging.info('copying..')
    dst = '%s/%s' % (result_dir, model_fn)
    if not os.path.exists(dst):
        shutil.copy(modelfn, dst)
    # Also copy to tf exported model directories
    dst = '%s/%s/%s' % (result_dir, 'tf_model_first', model_fn)
    if not os.path.exists(dst):
        shutil.copy(modelfn, dst)
    dst = '%s/%s/%s' % (result_dir, 'tf_model_last', model_fn)
    if not os.path.exists(dst):
        shutil.copy(modelfn, dst)
    dst = '%s/%s/%s' % (result_dir, 'tf_model_best', model_fn)
    if not os.path.exists(dst):
        shutil.copy(modelfn, dst)
    # Copy current trainig file
    dst = '%s/%s' % (result_dir, os.path.basename(__file__))
    if not os.path.exists(dst):
        shutil.copy(__file__, dst)

    # Create model
    model = net(net_kwargs) # possibly add arguments here (see the initializer of the model)
    
    return model

def validate( session, model, inputs, labels , batch_size = 256, logging = None, epoch_nr = -1, name = 'Val'):
    # Initialize
    epoch_size = m.ceil(len(inputs)/batch_size) - 1#number of iterations per epoch
    n_samples = labels.shape[0]
    probs = np.zeros(labels.shape)
    average_loss = 0
    t0=time.time()
    # Loop over the whole dataset
    for i in range(epoch_size+1):
        i_start = i*batch_size
        i_end = min(n_samples,(i+1)*batch_size)
        # The data to feed
        feed_dict = { 
                model.inputs_ph: inputs[i_start:i_end],
                model.labels_ph: labels[i_start:i_end],
                model.is_training_ph: False }
        # Do the operations
        batch_probs, batch_loss = session.run( [model.probabilities, model.loss], feed_dict)
        probs[i_start:i_end] = batch_probs[:,1]
        average_loss += batch_loss*(i_end-i_start)
    average_loss = average_loss/n_samples
    tElapsed = time.time() - t0

    # Some scores:
    auc_score = sklearn.metrics.roc_auc_score(labels, probs)
    acc_score = sklearn.metrics.accuracy_score(labels, np.round(probs))

    # Log the results
    msg = 'Epoch:{:03d}\t{}\t|\tav_loss={}\tACC={} \tAUC={} \tTime={}'.format(epoch_nr, name, round(average_loss,4), round(acc_score,4), round(auc_score,4), round(tElapsed,4))
    if not(logging==None):
        logging.info(msg)
    else:
        print(msg)
    
    return average_loss, acc_score, auc_score

def train_one_epoch( session, model, inputs, labels , train_op, batch_size = 64, logging = None, epoch_nr = -1, name = 'Train', steps_per_epoch = None):
    # Initialize
    if steps_per_epoch == None:
    	steps_per_epoch = m.ceil(len(inputs)/batch_size) #number of iterations per epoch

    n_samples = m.ceil(labels.shape[0])
    probs = np.zeros(labels.shape)
    average_loss = 0
    t0=time.time()
    # Create reandom batches
    samples=np.random.permutation(len(inputs))
    # Loop over the whole dataset
    for i in range(steps_per_epoch):
        i_start = i*batch_size
        i_end = min(n_samples,(i+1)*batch_size)

        # The data to feed
        feed_dict = { 
                model.inputs_ph: inputs[samples[i_start:i_end]],
                model.labels_ph: labels[samples[i_start:i_end]],
                model.is_training_ph: True }
        # Do the operations
        _, batch_probs, batch_loss = session.run( [train_op, model.probabilities, model.loss], feed_dict)
        probs[i_start:i_end] = batch_probs[:,1]
        average_loss += batch_loss*(i_end-i_start)
    average_loss = average_loss/n_samples
    tElapsed = time.time() - t0

    # Some scores:
    auc_score = sklearn.metrics.roc_auc_score(labels[samples[:i_end]], probs[:i_end])
    acc_score = sklearn.metrics.accuracy_score(labels[samples[:i_end]], np.round(probs[:i_end]))

    # Log the results
    msg = 'Epoch:{:03d}\t{}\t|\tav_loss={}\tACC={} \tAUC={} \tTime={}'.format(epoch_nr, name, round(average_loss,4), round(acc_score,4), round(auc_score,4), round(tElapsed,4))
    if not(logging==None):
        logging.info(msg)
    else:
        print(msg)
    
    return average_loss, acc_score, auc_score

def train(
    datadir, # Directory where to find the data
    resultdir_root, resultdir, # Directory where to export the results to
    modelfn, trainfn, valfn, # Model file name, training data fn, validation data fn
    epochs, batchsize, steps_per_epoch,# Nr of epochs, batch size
    # opt, opt_kwargs, # Optimizer, and it's arguments
    net_kwargs, # Arguments to pass to the network
    # transformations, # Transformations in the data augmentation
    weight_decay, # Weight decay factor
    l2_loss, # L2 loss constant (do not use at the same time with weight decay)
    lr, # The initial learning rage
    lr_decay_schedule, # Learning rate decay schedule in the for 'N1-N2-N3'
    lr_decay_factor, # Learning rate decay factor, decrease learning by multiplying with this factor every Ni epochs
    # gpu, # ?
    # seed, # ?
    # silent=False,
    logme=None): # Some extra logging info

    ## Load an pre-process the data
    # train_data, train_labels, val_data, val_labels = get_cifar10_data(datadir, trainfn, valfn)
    (train_data,train_labels,_), (test_data,test_labels,_), (val_data,val_labels,_) = get_pcam_data(datadir)

    # For quick testing:
    # train_data = train_data[0:10000]
    # train_labels = train_labels[0:10000]

    ## Create result dir
    if resultdir==None:
        resultdir = create_result_dir(resultdir_root, modelfn)
    
    ## Logger
    log_fn = '%s/log.txt' % resultdir
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(message)s',
        filename=log_fn, level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(logme)
    
    # The location where the model will be stored
    modelfn_base = os.path.basename(modelfn).split('.')[0]
    model_file_base_first = os.path.join( resultdir , 'tf_model_first', modelfn_base)
    model_file_first      = model_file_base_first+'.meta'
    model_file_base_last = os.path.join( resultdir , 'tf_model_last', modelfn_base)
    model_file_last      = model_file_base_last+'.meta'
    model_file_base_best = os.path.join( resultdir , 'tf_model_best', modelfn_base)
    model_file_best      = model_file_base_best+'.meta'
    model_file_base = os.path.join( resultdir , 'last', modelfn_base)
    model_file      = model_file_base+'.meta'

    ## Init graph
    graph = tf.Graph()
    graph.as_default()
    tf.compat.v1.reset_default_graph()

    ## Create model
    model = get_model(resultdir, modelfn, net_kwargs)

    ## Extract useful variables
    # Place holders:
    inputs_ph = model.inputs_ph
    labels_ph = model.labels_ph
    is_training_ph = model.is_training_ph
    # Metrics
    loss_op = model.loss
    loss_l2_op = model.loss_l2
    if l2_loss==0.:
    	loss_total_op = loss_op
    else:
    	loss_total_op = loss_op + l2_loss*loss_l2_op
    success_rate_op = model.success_rate
    number_of_errors_op = model.number_of_errors

    ## Batch settings
    if steps_per_epoch == None:
    	steps_per_epoch = m.ceil(len(train_data)/batchsize) #number of iterations per epoch'

    ## Optimizer
    # Set up the learning reate decay scheme
    global_step = tf.Variable(0, trainable=False)
    boundaries = [int(istr)*steps_per_epoch - 1 for istr in lr_decay_schedule.split('-')] # The boundaries of the intervals
    # boundaries = list(np.cumsum(lr_decay_schedule)*steps_per_epoch-1) # Accumulate (determine the boundaries of the intervals)
    values = [lr] + [ lr*pow(lr_decay_factor,n+1) for n in range(len(boundaries))] # Decay after each transition/boundary
    learning_rate = tf.compat.v1.train.piecewise_constant(global_step, boundaries, values)
    # Define the optimizer
    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    if weight_decay==0.: # Without weight decay
        # The optimizer
        optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9)
        # The update opteration (one optimization step)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss=loss_total_op,global_step=global_step)
    else: # With weight decay
        # The optimizer
        optimizer_wd = tf.contrib.opt.MomentumWOptimizer
        optimizer =optimizer_wd(learning_rate=learning_rate,momentum=0.9,weight_decay=learning_rate*weight_decay)
        # The variables to decay
        decay_var_list = [model.all_weights[key] for key in model.all_weights]
        # The update opteration (one optimization step)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss=loss_total_op,global_step=global_step,decay_var_list=decay_var_list)

    ## Start the (GPU) session
    initializer = tf.compat.v1.global_variables_initializer()
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_frac)
    gpu_options = tf.compat.v1.GPUOptions(allow_growth = True)
    session = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(),config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) #-- Session created
    session.run(initializer)

    ## Define the saver
    saver = tf.compat.v1.train.Saver()
    # Restore previously saved model
    if os.path.exists(model_file_last): # Restore model weights from previously saved model
        saver.restore(session, model_file_base_last)
        print("Model restored from file: %s" % model_file_last + '\n')
    else:
    	# Export initialized model
    	saver.save(session, model_file_base_first)

    ## Do the optimization
    # last_learning_rate = session.run(learning_rate)
    best_validation_success_rate = 0
    start_epoch = int(round(session.run(global_step)/steps_per_epoch))
    for epoch_nr in range(start_epoch,epochs):
        logging.info('')
        ## Training
        average_loss_train, acc_score_train, auc_score_train = train_one_epoch( session, model, train_data, train_labels , train_op, batch_size = batchsize, logging = logging, epoch_nr = epoch_nr, name = 'Train', steps_per_epoch = steps_per_epoch)

        # Validation
        average_loss_val, acc_score_val, auc_score_val = validate( session, model, val_data, val_labels , batch_size = batchsize, logging = logging, epoch_nr = epoch_nr, name = 'Val')

        # Testing
        average_loss_test, acc_score_test, auc_score_test = validate( session, model, test_data, test_labels , batch_size = batchsize, logging = logging, epoch_nr = epoch_nr, name = 'Test')

        ## Store the model after every epoch
        saver.save(session, model_file_base_last)
        if acc_score_val >= best_validation_success_rate:
        	best_validation_success_rate = acc_score_val
        	saver.save(session, model_file_base_best)
 
    return 0


if __name__ == '__main__':
    
    parser = train_arg_parser()
    args = parser.parse_args()
    vargs = vars(args)

    # Check available gpus
    print('available gpus: ', get_available_gpus())
    
    val_error = train(logme=vargs, **vargs)

    print( 'Finished training' )
    print( 'Final validation error:', val_error )
    print( 'Saving model...' )
    # import chainer.serializers as sl
    # sl.save_hdf5('./my.model', model)
