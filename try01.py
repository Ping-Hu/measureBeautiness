# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 23:55:02 2015

@author: Peggy
"""

import os
import glob
import string
import random
import scipy.misc
from scipy.stats import pearsonr
from PIL import ImageEnhance
from nolearn.lasagne import BatchIterator
from sklearn.utils import shuffle
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import pickle
from datetime import datetime
import sys
from matplotlib import pyplot
import numpy as np
import theano



#   from PIL import Image

FTRAIN0 = 'training.csv'
FTEST0 = 'test.csv'
#FTRAIN = '/Users/Peggy/Documents/MATLAB/FinalProject/dataset_face/preprocess/nolearnTrain/target_matrix.csv'
FTRAIN = '/Users/Peggy/Documents/MATLAB/FinalProject/dataset_face/preprocess/nolearn/train'
FTEST = '/Users/Peggy/Documents/MATLAB/FinalProject/dataset_face/preprocess/nolearn/test'
start_train = False
TrainOrTest = ''
PlotLoss = False
PickNet = '6'
max_ep = 2
which_ep = 4
ComputePearson = False
PlotPearson = True
ComputeMSE = False

def float32(k):
    return np.cast['float32'](k)


class FlipBatchIterator(BatchIterator):
    
    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)

        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices, :, :, :] = Xb[indices, :, :, ::-1];
            
        if( (PickNet == '6')):
            # adjust contrast  
            indices = np.random.choice(bs, bs / 2, replace=False)
            Xb = np.array(Xb)
            Xb = float32(Xb)
            mean = np.mean(Xb)  
            Xb[indices, :, :, :] = Xb[indices, :, :, :] - mean;    
            Xb[indices, :, :, :] = Xb[indices, :, :, :]*1.5 + mean*0.7
            big = np.max(Xb)
            Xb[indices, :, :, :] = Xb[indices, :, :, :]/big
        
            
        return Xb, yb
        
        
#class AdjustVariable(object):
#    def __init__(self, name, start=0.03, stop=0.001):
#        self.name = name
#        self.start, self.stop = start, stop
#        self.ls = None
#
#    def __call__(self, nn, train_history):
#        if self.ls is None:
#            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)
#
#        epoch = train_history[-1]['epoch']
#        new_value = np.cast['float32'](self.ls[epoch - 1])
#        getattr(nn, self.name).set_value(new_value)

    
def load2d(test=False, cols=None):
    
    X120 = np.zeros((1500,3,120,90))
    X120 = float32(X120)
    
    y120 = np.zeros((1500,1))
    y120 = float32(y120)
    
    X110 = np.zeros((1500,3,110,90))
    X110 = float32(X110)
    
    y110 = np.zeros((1500,1))
    y110 = float32(y110)
    
    Xtest120 = np.zeros((500,3,120,90))
    Xtest120 = float32(Xtest120)
    
    Xtest110 = np.zeros((500,3,110,90))
    Xtest110 = float32(Xtest110)
    
    testlabel = np.zeros((500,1))
    testlabel = float32(testlabel)    
    
    tempX = np.zeros((1500,3,120,90))
    tempX = float32(tempX)
    
    tempy = np.zeros((1500,1))
    tempy = float32(tempy)
    
    tempX_test = np.zeros((500,3,120,90))
    tempX_test = float32(tempX_test)
    
    # prepare training set    
    order = np.arange(1500)
    random.shuffle(order)
    
    allfiles = []
    ylabels = []
    inpath = FTRAIN
    files = os.listdir(FTRAIN);
    for f in files:
            if(os.path.isdir(inpath + '/' + f)): 
                if(f[0] == '.'):
                    pass
                else:
                    ID = int(f)
                    #print(ID)
                    deeper_path = inpath + '/' + f
                    deeper_files = os.listdir(deeper_path)
                    for ff in deeper_files:
                        if( not (os.path.isdir(deeper_path + '/' + ff))):
                            if( not (ff[0] == '.')):
                                allfiles.append(deeper_path + '/' + ff)
                                ylabels.append(ID)
    
    count = 0 
    for i in range(1500):
        #im = allfiles[sele[i]]
        im = allfiles[i]
        temp_im = scipy.misc.imread(im) 
        temp_imR = temp_im[:,:,0]
        temp_imG = temp_im[:,:,1]
        temp_imB = temp_im[:,:,2]
        tempX[count,0,:,:] = temp_imR[:,:]/255
        tempX[count,1,:,:] = temp_imG[:,:]/255
        tempX[count,2,:,:] = temp_imB[:,:]/255
        #y[count,0] = (ylabels[sele[i]]-10)/20
        tempy[count,0] = (ylabels[i]-10)/20
        count = count+1   
      
    for i in range(1500):
        X120[i,:,:,:] = tempX[order[i],:,:,:]
        y120[i,0] = tempy[order[i],0]                     
                            
                            
    h_pos = np.random.randint(0, 10)
    for i in range(1500):
        X110[i,:,:,:] = tempX[order[i], :, h_pos:h_pos+110,:]
        y110[i,0] = tempy[order[i],0]
        
        
    #prepare testing set:      
    allfiles_test = []
    inpath_test = FTEST
    files_test = os.listdir(FTEST);
    c = 0;
    for f in files_test:
            if(os.path.isdir(inpath_test + '/' + f)): 
                if(f[0] == '.'):
                    pass
                else:
                    ID = int(f)
                    deeper_path_test = inpath_test + '/' + f
                    deeper_files_test = os.listdir(deeper_path_test)
                    for ff in deeper_files_test:
                        if(not (os.path.isdir(deeper_path_test + '/' + ff))):
                            if( not (ff[0] == '.')):
                                allfiles_test.append(deeper_path_test + '/' + ff)
                                testlabel[c,0] = ID
                                c = c+1
    
    count = 0 
    for i in range(500):
        #im = allfiles[sele[i]]
        im = allfiles_test[i]
        temp_imtest = scipy.misc.imread(im) 
        temp_testR = temp_imtest[:,:,0]
        temp_testG = temp_imtest[:,:,1]
        temp_testB = temp_imtest[:,:,2]
        tempX_test[count,0,:,:] = temp_testR[:,:]/255
        tempX_test[count,1,:,:] = temp_testG[:,:]/255
        tempX_test[count,2,:,:] = temp_testB[:,:]/255
        testlabel[count,0] = (testlabel[count,0]-10)/20
        #y[count,0] = (ylabels[sele[i]]-10)/20
        count = count+1
        
        
    Xtest120[:,:,:,:] = tempX_test[:,:,:,:]
        
    h_pos = np.random.randint(0, 10);
    Xtest110[:,:,:,:] = tempX_test[:, :, h_pos:h_pos+110,:];         
            
        
    if(test == False):
        if(PickNet == '3') or (PickNet == '4'):
            return X110, y110
        else:
            return X120, y120
    elif(test == True):
        if(PickNet == '3') or (PickNet == '4'):
            return Xtest110, testlabel#, allfiles_test
        else:
            return Xtest120, testlabel#, allfiles_test
    

if(PickNet == '1'):
    print('picknet == 1')
    net = NeuralNet(
        layers=[  # three layers: one hidden layer            
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),            
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),            
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),            
            ('hidden4', layers.DenseLayer),
            ('hidden5', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        # layer parameters:
        input_shape=(None, 3, 120, 90),
        conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
        conv2_num_filters=64, conv2_filter_size=(2, 3), pool2_pool_size=(2, 2),
        conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
        hidden4_num_units=200,
        hidden5_num_units=200,
        output_num_units=1, output_nonlinearity=None,
    
        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=theano.shared(float32(0.03)),
        update_momentum=theano.shared(float(0.9)),
    
        regression=True,
        batch_iterator_train=FlipBatchIterator(batch_size=32),
    #    on_epoch_finished=[
    #        AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
    #        AdjustVariable('update_momentum', start=0.9, stop=0.999),
    #        ],
        max_epochs= max_ep,
        verbose=1,
    
        )
elif(PickNet == '2'):
    net = NeuralNet(
        layers=[  # three layers: one hidden layer
            
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('dropout1', layers.DropoutLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('dropout2', layers.DropoutLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('dropout3', layers.DropoutLayer),
            ('hidden4', layers.DenseLayer),
            ('dropout4', layers.DropoutLayer),
            ('hidden5', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        # layer parameters:
        input_shape=(None, 3, 120, 90),
        conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
        dropout1_p=0.1,
        conv2_num_filters=128, conv2_filter_size=(2, 3), pool2_pool_size=(2, 2),
        dropout2_p=0.2,
        conv3_num_filters=256, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
        dropout3_p=0.3,
        hidden4_num_units=200,
        dropout4_p=0.5,
        hidden5_num_units=200,
        output_num_units=1, output_nonlinearity=None,
    
        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=theano.shared(float32(0.03)),
        update_momentum=theano.shared(float(0.9)),
    
        regression=True,
        batch_iterator_train=FlipBatchIterator(batch_size=32),
#        on_epoch_finished=[
#            AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
#            AdjustVariable('update_momentum', start=0.9, stop=0.999),
#            ],
        max_epochs= max_ep,
        verbose=1,
    
        )
        
elif(PickNet == '3'):
    print('picknet == 3')
    net = NeuralNet(
        layers=[  # three layers: one hidden layer
            
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('dropout1', layers.DropoutLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('dropout2', layers.DropoutLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('dropout3', layers.DropoutLayer),
            ('hidden4', layers.DenseLayer),
            ('dropout4', layers.DropoutLayer),
            ('hidden5', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        # layer parameters:
        input_shape=(None, 3, 110, 90),
        conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
        dropout1_p=0.1,
        conv2_num_filters=64, conv2_filter_size=(3, 3), pool2_pool_size=(2, 2),
        dropout2_p=0.2,
        conv3_num_filters=128, conv3_filter_size=(3, 2), pool3_pool_size=(2, 2),
        dropout3_p=0.3,
        hidden4_num_units=200,
        dropout4_p=0.5,
        hidden5_num_units=200,
        output_num_units=1, output_nonlinearity=None,
    
        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=theano.shared(float32(0.03)),
        update_momentum=theano.shared(float(0.9)),
    
        regression=True,
        batch_iterator_train=FlipBatchIterator(batch_size=32),
    #    on_epoch_finished=[
    #        AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
    #        AdjustVariable('update_momentum', start=0.9, stop=0.999),
    #        ],
        max_epochs= max_ep,
        verbose=1,
    
        )        
elif(PickNet == '4'):
    print('picknet == 4')
    net = NeuralNet(
        layers=[  # three layers: one hidden layer
            
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('dropout1', layers.DropoutLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('dropout2', layers.DropoutLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('dropout3', layers.DropoutLayer),
            ('hidden4', layers.DenseLayer),
            ('dropout4', layers.DropoutLayer),
            ('hidden5', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        # layer parameters:
        input_shape=(None, 3, 110, 90),
        conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
        dropout1_p=0.1,
        conv2_num_filters=64, conv2_filter_size=(3, 3), pool2_pool_size=(2, 2),
        dropout2_p=0.2,
        conv3_num_filters=128, conv3_filter_size=(3, 2), pool3_pool_size=(2, 2),
        dropout3_p=0.3,
        hidden4_num_units=200,
        dropout4_p=0.5,
        hidden5_num_units=200,
        output_num_units=1, output_nonlinearity=None,
    
        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=theano.shared(float32(0.03)),
        update_momentum=theano.shared(float(0.9)),
    
        regression=True,
        batch_iterator_train=FlipBatchIterator(batch_size=32),
    #    on_epoch_finished=[
    #        AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
    #        AdjustVariable('update_momentum', start=0.9, stop=0.999),
    #        ],
        max_epochs= max_ep,
        verbose=1,
    
        )             

elif(PickNet == '5'):
    print('picknet == 5')
    net = NeuralNet(
        layers=[  # three layers: one hidden layer            
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),            
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),            
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),            
            ('hidden4', layers.DenseLayer),
            ('hidden5', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        # layer parameters:
        input_shape=(None, 3, 120, 90),
        conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
        conv2_num_filters=64, conv2_filter_size=(2, 3), pool2_pool_size=(2, 2),
        conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
        hidden4_num_units=200,
        hidden5_num_units=200,
        output_num_units=1, output_nonlinearity=None,
    
        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=theano.shared(float32(0.02)),
        update_momentum=theano.shared(float(0.9)),
    
        regression=True,
        batch_iterator_train=FlipBatchIterator(batch_size=32),
    #    on_epoch_finished=[
    #        AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
    #        AdjustVariable('update_momentum', start=0.9, stop=0.999),
    #        ],
        max_epochs= max_ep,
        verbose=1,
    
        )

elif(PickNet == '6'):
    print('picknet == 6')
    net = NeuralNet(
        layers=[  # three layers: one hidden layer            
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),            
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),            
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),            
            ('hidden4', layers.DenseLayer),
            ('hidden5', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        # layer parameters:
        input_shape=(None, 3, 120, 90),
        conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
        conv2_num_filters=64, conv2_filter_size=(2, 3), pool2_pool_size=(2, 2),
        conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
        hidden4_num_units=200,
        hidden5_num_units=200,
        output_num_units=1, output_nonlinearity=None,
    
        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=theano.shared(float32(0.03)),
        update_momentum=theano.shared(float(0.9)),
    
        regression=True,
        batch_iterator_train=FlipBatchIterator(batch_size=32),
    #    on_epoch_finished=[
    #        AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
    #        AdjustVariable('update_momentum', start=0.9, stop=0.999),
    #        ],
        max_epochs= max_ep,
        verbose=1,
    
        )
elif(PickNet == '7'):
    print('picknet == 6')
    net = NeuralNet(
        layers=[  # three layers: one hidden layer
            
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('dropout1', layers.DropoutLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('dropout2', layers.DropoutLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('dropout3', layers.DropoutLayer),
            ('hidden4', layers.DenseLayer),
            ('dropout4', layers.DropoutLayer),
            ('hidden5', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        # layer parameters:
        input_shape=(None, 3, 120, 90),
        conv1_num_filters=64, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
        dropout1_p=0.2,
        conv2_num_filters=128, conv2_filter_size=(2, 3), pool2_pool_size=(2, 2),
        dropout2_p=0.3,
        conv3_num_filters=256, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
        dropout3_p=0.4,
        hidden4_num_units=200,
        dropout4_p=0.5,
        hidden5_num_units=100,
        output_num_units=1, output_nonlinearity=None,
    
        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=theano.shared(float32(0.03)),
        update_momentum=theano.shared(float(0.9)),
    
        regression=True,
        batch_iterator_train=FlipBatchIterator(batch_size=32),
#        on_epoch_finished=[
#            AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
#            AdjustVariable('update_momentum', start=0.9, stop=0.999),
#            ],
        max_epochs= max_ep,
        verbose=1,
    
        )


""" model 1 parameters: max-epochs = 1;range = 10; random-flipping
    model 2 parameters: max-epochs = 10;range = 10; random-flipping
    model 3 random-flipping + soft-window      
"""

if(PlotLoss == False):
    if(TrainOrTest == 'train'):
        X, y = load2d()
        Xtest,TestLabels = load2d(test = True)
        for i in range(30):
            if(start_train == True):
                epoch_name = str(max_ep*i)
                constant_name = 'cnn'+ PickNet + '.pkl'
                current_net_name = 'cnn'+ PickNet + '-' + epoch_name + '.pkl'
                
                print('load net...')
                net = pickle.load(open(constant_name, 'rb'))
                print('start training the %s-th time'%(i+1))
                net.fit(X, y)
                print('one epoch is finished.')
                pickle.dump(net, open(constant_name, 'wb'))
                pickle.dump(net, open(current_net_name, 'wb'))
                print('finish training the %s-th time'%(i+1))
#                print('start loading last pearson correlation data...')
#                pear_name = 'Net' + PickNet + 'pearson_epoch' + '.pkl'
#                res = np.zeros((i+1,1))
#                for j in range(i):
#                    res[j,0] = pickle.load(open(pear_name, 'rb'))[j]
#                print('start making prediciton')
#                ytest = net.predict(Xtest)
#                s = 0
#                for j in range(500):
#                    s = s + (ytest[j] - TestLabels)*(ytest[j] - TestLabels) 
#                mse = s/500;
#                res[i,0] = pearsonr(ytest, TestLabels)[0][0];
#                print('the prediction MSE is: %s, Pearson is: %s'%(mse, res[i,0]))
#                pickle.dump(res, open(pear_name, 'wb'))
                
                #pyplot.plot(res, linewidth=3, label="PearsonCorrelation")
                #pyplot.grid()
                #pyplot.legend()
                #pyplot.xlabel("epoch")
                #pyplot.ylabel("loss")
                #pyplot.ylim(-1, 1)
                #pyplot.yscale("log")
                #pyplot.show()
            else:
                
                epoch_name = str(max_ep*i)
                constant_name = 'cnn'+ PickNet + '.pkl'
                current_net_name = 'cnn'+ PickNet + '-' + epoch_name + '.pkl'
                print('net %s start training..'%(PickNet))
                net.fit(X, y)
                print('one epoch is finished.')
                pickle.dump(net, open(constant_name, 'wb'))
                pickle.dump(net, open(current_net_name, 'wb'))
                start_train = True
                
#                pear_name = 'pearson_epoch.pkl'
#                
#                ytest = net.predict(Xtest)
#                res = pearsonr(ytest, TestLabels)[0][0];
#                pickle.dump(res, open(pear_name, 'wb'))
                
    elif(TrainOrTest == 'test'):
            for j in range(1):
                net_name = 'cnn'+ PickNet + str(which_ep) + '.pkl'
                test_result_name = 'ytest' + PickNet + '-' + str(which_ep) + '.pkl'
                Xtest,TestLabels = load2d(test = True)
                #net = pickle.load(open('cnn2.pkl','rb'))
                net = pickle.load(open(net_name,'rb'))
                print('Start testing...\n')
                ytest = net.predict(Xtest)
                print('Testing is finished.\n')
                pickle.dump(ytest, open(test_result_name,'wb'))
                print('real label of image1 is %f and prediction is %f'%(TestLabels[0],ytest[0]))
                print(ytest[0])
                print(ytest[499])
           # pearsonr(ytest, TestLabels)[0][0]
elif(PlotLoss == True):
    #net_name = 'cnn'+ PickNet + '-' + str(which_ep) + '.pkl'
    net_name = 'cnn7.pkl'
    net = pickle.load(open(net_name, 'rb'))   
    train_loss = np.array([i["train_loss"] for i in net.train_history_])
    valid_loss = np.array([i["valid_loss"] for i in net.train_history_])
#    print(train_loss[48])
#    print(valid_loss[48])
    pyplot.plot(train_loss, linewidth=3, label="train")
    pyplot.plot(valid_loss, linewidth=3, label="valid")
    pyplot.grid()
    pyplot.legend()
    pyplot.xlabel("epoch")
    pyplot.ylabel("loss")
    pyplot.ylim(0.01, 0.2)
    pyplot.yscale("log")
    pyplot.show()

if(ComputePearson == True):
    #test_result_name = 'ytest'+PickNet + '-' + str(which_ep) +'.pkl'
    test_result_name = 'ytest'+PickNet + '.pkl'
    predict = pickle.load(open(test_result_name,'rb'))
    Xtest,TestLabels = load2d(test = True)
    res = pearsonr(predict, TestLabels)[0][0];
    print(res)

if(ComputeMSE == True):
    print('loading testing set..')
    Xtest,TestLabels,files = load2d(test = True)
    print('loading trained net..')
    net = pickle.load(open('cnn6.pkl','rb'))
    print('making prediction..')
    ytest = net.predict(Xtest)
    print('computing MSE..')
    s = 0
    for j in range(500):
        s = s + (ytest[j] - TestLabels[j,0])*(ytest[j] - TestLabels[j,0]) 
    mse = s/500;
    print('MSE of cnn %s is:'%(PickNet))
    print(mse)
    print('See some images..')
    diff = np.zeros((500))
    diff = float32(diff)
    for j in range(500):
        diff[j] = abs(ytest[j] - TestLabels[j,0])
    print('the top 10 smallest/biggest differences are of the indices: ')
    print(np.argsort(diff)[0:10])
    print('the corresponding image predictions are: ')
    print(ytest[np.argsort(diff)[0:10]] * 20 +10)
    print('these images real scores are: ')
    print(TestLabels[np.argsort(diff)[0:10]] * 20 + 10)
    print('the corresponding image files are: ')
    for k in range(10):    
        na = files[np.argsort(diff)[k]]
        print(na)
    


if(PlotPearson == True):
    Xtest,TestLabels = load2d(test = True)
    net = pickle.load(open('cnn6.pkl','rb'))
    ytest = net.predict(Xtest)
    res = pearsonr(ytest, TestLabels)[0][0];
    print(res)
    xcomponent = np.array(ytest);
    ycomponent = np.array(TestLabels);
    pyplot.plot(xcomponent, ycomponent, 'ro')
    #pyplot.plot(ycomponent, linewidth=3, label="groundtruth")
    pyplot.grid()
    pyplot.legend()
    pyplot.xlabel("net 6 prediction")
    pyplot.ylabel("groundtruth")
    pyplot.ylim(-1, 1)
    #pyplot.yscale("log")
    pyplot.show()


