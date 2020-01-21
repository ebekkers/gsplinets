import os
import numpy as np


def get_celebanoses_data(datadir):

    assert os.path.exists(datadir), 'Datadir does not exist: %s' % datadir

    train_dataset = _load_dataset(datadir,'train')
    test_dataset = _load_dataset(datadir,'test')
    val_dataset = _load_dataset(datadir,'val')
    
    return (train_dataset['x'], train_dataset['y'], train_dataset['y_ij']), (test_dataset['x'], test_dataset['y'], test_dataset['y_ij']), (val_dataset['x'], val_dataset['y'], val_dataset['y_ij'])

def _load_dataset(datadir,setname):
    # The images:
    filename_x = os.path.join(datadir,setname+'_x.h5')
    x = _load_H5(filename_x)
    # The target heatmap
    filename_y = os.path.join(datadir,setname+'_y.h5')
    y = _load_H5(filename_y)
    # The noses
    filename_meta = os.path.join(datadir,setname+'_meta.h5')
    y_ij = np.transpose(np.array([_load_H5(filename_meta,key) for key in ['lefteye','righteye','nose','leftmouth','rightmouth']]),[1,0,2])
    # Store all data in a python library
    dataset = {'x':x,'y':y,'y_ij':y_ij}
    return dataset


def _load_H5(file, key = None):
    import h5py
    with h5py.File(file, 'r') as f:
        if key == None:
            data = f.get(list(f.keys())[0])[()]
        else:
            data = f.get(key)[()]
    return data