import numpy as np
from keras.utils import multi_gpu_model
from keras.optimizers import Adam
import matplotlib.pyplot as plt

from model_SVAD import *
from load_feature import *

class Options(object):
    def __init__(self):
        self.threshold = 0.5
        self.stride = 5


def SVAD(FILE_NAME,options):

    model = SVAD_CONV_MultiLayer()
    weight_name = './weights/SVAD_CNN_ML.hdf5'
    opt = Adam(lr=0.05, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.load_weights(weight_name)

    ''' load test data '''
    feature = featureExtract(FILE_NAME)
    x_test = makingTensor(feature, stride= options.stride)

    ''' prediction'''
    y_predict = model.predict(x_test, verbose=1)
    y_predict = (y_predict > options.threshold)
    y_predict = y_predict.astype('int8')

    return y_predict


if __name__ == '__main__':
    options = Options()

    FILE_NAME = './data/example2.wav'
    y_predict = SVAD(FILE_NAME,options)

    PATH_SAVE = './results/result1.txt'
    if not os.path.exists(os.path.dirname(PATH_SAVE)):
	    os.makedirs(os.path.dirname(PATH_SVAE))
	
    with open(PATH_SAVE, 'w',) as f:
        for j in range(len(y_predict)):
            est = "%.2f\t%.2f\n" % (0.01 * options.stride * j, y_predict[j])
            f.write(est)
