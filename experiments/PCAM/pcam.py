import os
import numpy as np


def get_pcam_data(datadir):

    assert os.path.exists(datadir), 'Datadir does not exist: %s' % datadir

    train_dataset = _load_dataset(datadir,'train')
    test_dataset = _load_dataset(datadir,'test')
    val_dataset = _load_dataset(datadir,'valid')
    
    return (train_dataset['x'], train_dataset['y'].flatten(), train_dataset), (test_dataset['x'], test_dataset['y'].flatten(), test_dataset), (val_dataset['x'], val_dataset['y'].flatten(), val_dataset)

def _load_dataset(datadir,setname):
    # The images:
    filename_x = os.path.join(datadir,'camelyonpatch_level_2_split_'+setname+'_x.h5')
    x = _load_PCAM_H5(filename_x)
    # The labels (whether it is a centered tumor patch or not)
    filename_y = os.path.join(datadir,'camelyonpatch_level_2_split_'+setname+'_y.h5')
    y = _load_PCAM_H5(filename_y)
    # The meta data
    filename_meta = os.path.join(datadir,'camelyonpatch_level_2_split_'+setname+'_meta.csv')
    tumor_patch, center_tumor_patch, wsi = _load_PCAM_CSV(filename_meta)
    # Store all data in a python library
    dataset = {'x':x,'y':y,'tumor_patch':tumor_patch,'center_tumor_patch':center_tumor_patch,'wsi':wsi}
    return dataset

def _load_PCAM_H5(file, key = None):
    import h5py
    with h5py.File(file, 'r') as f:
        if key == None:
            data = f.get(list(f.keys())[0])[()]
        else:
            data = f.get(key)[()]
    return data

def _load_PCAM_CSV(file):
    import csv
    tumor_patch = []
    center_tumor_patch = []
    wsi = []
    with open(file, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            tumor_patch+=[row["tumor_patch"]=='True']
            center_tumor_patch+=[row["center_tumor_patch"]=='True']
            wsi+=[row["wsi"]]
    return tumor_patch,center_tumor_patch,wsi