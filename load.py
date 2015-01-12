import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
import numpy as np

def start(train_dataName,train_labelName,test_dataName,test_labelName):
    training_data = read(range(10),train_dataName,train_labelName, "training")
    test_data = read(range(10),test_dataName,test_labelName, "testing")
    return training_data, test_data
def read(digits,dataName,labelName, dataset = "training", path = "."):
    if dataset is "training":
        fname_img = os.path.join(path, dataName)
        fname_lbl = os.path.join(path, labelName)
    elif dataset is "testing":
        fname_img = os.path.join(path, dataName)
        fname_lbl = os.path.join(path, labelName)
    else:
        raise ValueError, "dataset must be 'testing' or 'training'"

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in xrange(size) if lbl[k] in digits ]
    N = size

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    
    for i in xrange(size):
        images[i] = array(img[ i*rows*cols : (i+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[i]
    images /= 255.0
    if dataset is "training":
		inputs = [np.reshape(x, (784, 1)) for x in images]
		results = [vectorized_result(y) for y in labels]
		data = zip(inputs, results)
    elif dataset is "testing":
		inputs = [np.reshape(x, (784, 1)) for x in images]
		data = zip(inputs, labels)
			
    return data


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

