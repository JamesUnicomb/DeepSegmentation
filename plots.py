import os, shutil
import numpy as np
from skimage import io

import matplotlib.pyplot as plt

train = True

trX = []
teX = []

trY = []
teY = []

try:
    if train:
        shutil.rmtree('plot_results/train/result')
        shutil.rmtree('plot_results/train/gt')
        shutil.rmtree('plot_results/train/images')
    else:
        shutil.rmtree('plot_results/test/result')
        shutil.rmtree('plot_results/test/gt')
        shutil.rmtree('plot_results/test/images')
        shutil.rmtree('plot_results/test/errors')
        shutil.rmtree('plot_results/test/mask')
except OSError:
    pass


try:
    if train:
        os.makedirs('plot_results/train/result')
        os.makedirs('plot_results/train/gt')
        os.makedirs('plot_results/train/images')
    else:
        os.makedirs('plot_results/test/result')
        os.makedirs('plot_results/test/gt')
        os.makedirs('plot_results/test/images')
        os.makedirs('plot_results/test/errors')
        os.makedirs('plot_results/test/mask')
except OSError:
    pass


if train:
    for p in os.listdir('leftImg8bit/train'):
        print os.path.join('leftImg8bit/train',p)

        for f in os.listdir(os.path.join('leftImg8bit/train',p)):
            img_file = os.path.join('leftImg8bit/train',p,f)
            trX.append(io.imread(img_file)[::8,::8,:].astype(np.uint8))

            img_str_splt = img_file.split('leftImg8bit')
            gt_str = 'gtFine' + img_str_splt[1] + 'gtFine_labelIds' + img_str_splt[2]
            img = io.imread(gt_str)
            trY.append(np.logical_or(np.logical_or(np.logical_or(img == 7, img == 8), img == 9), img == 10)[::8,::8].astype(np.uint8))

else:
    for p in os.listdir('leftImg8bit/val'):
        print os.path.join('leftImg8bit/val',p)

        for f in os.listdir(os.path.join('leftImg8bit/val',p)):
            img_file = os.path.join('leftImg8bit/val',p,f)
            teX.append(io.imread(img_file)[::8,::8,:].astype(np.uint8))

            img_str_splt = img_file.split('leftImg8bit')
            gt_str = 'gtFine' + img_str_splt[1] + 'gtFine_labelIds' + img_str_splt[2]
            img = io.imread(gt_str)
            teY.append(np.logical_or(np.logical_or(np.logical_or(img == 7, img == 8), img == 9), img == 10)[::8,::8].astype(np.uint8))

trX = np.array(trX)
teX = np.array(teX)

trY = np.array(trY)
teY = np.array(teY)


import theano
import theano.tensor as T

import lasagne
from lasagne.updates import adam, total_norm_constraint
from lasagne.objectives import squared_error, binary_crossentropy
from lasagne.layers import DenseLayer, InputLayer, ConcatLayer, Conv2DLayer, MaxPool2DLayer, DilatedConv2DLayer, \
                           Pool2DLayer, Upscale2DLayer, PadLayer, ElemwiseSumLayer, ConcatLayer, SpatialPyramidPoolingLayer, \
                           LocallyConnected2DLayer, TransposedConv2DLayer, ReshapeLayer, \
                           FlattenLayer, DimshuffleLayer, NonlinearityLayer, \
                           BatchNormLayer, batch_norm, DropoutLayer, GaussianNoiseLayer, \
                           get_output, get_all_params, get_all_param_values, set_all_param_values
from lasagne.nonlinearities import rectify, sigmoid, softmax, tanh, linear


def output_layer_nonlinearity(x):
    return T.clip(sigmoid(x),1e-5,1.0-1e-4)


def network(image, p):
    input_image = InputLayer(input_var = image,
                             shape     = (None, 128, 256, 3))

    input_image = DimshuffleLayer(input_image,
                                  pattern = (0,3,1,2))

    conv1        = batch_norm(Conv2DLayer(input_image,
                                          num_filters  = 16,
                                          filter_size  = (3,3),
                                          stride       = (1,1),
                                          nonlinearity = rectify,
                                          pad          = 'same'))

    conv1        = batch_norm(Conv2DLayer(conv1,
                                          num_filters  = 16,
                                          filter_size  = (3,3),
                                          stride       = (1,1),
                                          nonlinearity = rectify,
                                          pad          = 'same'))

    conv1        = DropoutLayer(conv1, p=p)

    conv1        = ConcatLayer([input_image,
                                conv1], axis = 1)

    conv2        = batch_norm(Conv2DLayer(conv1,
                                          num_filters  = 32,
                                          filter_size  = (3,3),
                                          stride       = (1,1),
                                          nonlinearity = rectify,
                                          pad          = 'same'))

    conv2        = batch_norm(Conv2DLayer(conv2,
                                          num_filters  = 32,
                                          filter_size  = (3,3),
                                          stride       = (1,1),
                                          nonlinearity = rectify,
                                          pad          = 'same'))

    conv2        = DropoutLayer(conv2, p=p)

    conv2        = batch_norm(ConcatLayer([conv2,
                                           conv1], axis = 1))

    atr1         = DilatedConv2DLayer(PadLayer(conv2, width = 1),
                                      num_filters  = 16,
                                      filter_size  = (3,3),
                                      dilation     = (1,1),
                                      pad          = 0,
                                      nonlinearity = rectify)

    atr2         = DilatedConv2DLayer(PadLayer(conv2, width = 2),
                                      num_filters  = 16,
                                      filter_size  = (3,3),
                                      dilation     = (2,2),
                                      pad          = 0,
                                      nonlinearity = rectify)

    atr4         = DilatedConv2DLayer(PadLayer(conv2, width = 4),
                                      num_filters  = 16,
                                      filter_size  = (3,3),
                                      dilation     = (4,4),
                                      pad          = 0,
                                      nonlinearity = rectify)

    atr8         = DilatedConv2DLayer(PadLayer(conv2, width = 8),
                                      num_filters  = 16,
                                      filter_size  = (3,3),
                                      dilation     = (8,8),
                                      pad          = 0,
                                      nonlinearity = rectify)

    sumblock    = ConcatLayer([conv2,atr1,atr2,atr4,atr8], axis = 1)

    crp         = MaxPool2DLayer(PadLayer(sumblock, width = 1),
                                 pool_size     = (3,3),
                                 stride        = (1,1),
                                 ignore_border = False)

    crp         = batch_norm(Conv2DLayer(crp,
                                         num_filters  = 115,
                                         filter_size  = (3,3),
                                         stride       = (1,1),
                                         nonlinearity = rectify,
                                         pad          = 'same'))

    sumblock    = ElemwiseSumLayer([sumblock,
                                    crp])

    ground      = batch_norm(Conv2DLayer(sumblock,
                                         num_filters  = 1,
                                         filter_size  = (3,3),
                                         stride       = (1,1),
                                         nonlinearity = output_layer_nonlinearity,
                                         pad          = 'same'))

    ground        = ReshapeLayer(ground,
                                 shape = ([0],128,256))

    return ground


X  = T.ftensor4()
Y  = T.ftensor3()
P  = T.scalar()
lr = T.scalar()

RoadSegment = network(X,P)
rs          = get_output(RoadSegment)
frs         = get_output(RoadSegment, deterministic=True)

def load_weights():
    model_name = 'model_weights.npz'
    with np.load(model_name) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    set_all_param_values([RoadSegment], param_values)

def save_weights():
    np.savez('model_weights.npz', *get_all_param_values([RoadSegment]))

road_segment  = theano.function(inputs               = [X],
                                outputs              = frs,
                                allow_input_downcast = True)

loss = binary_crossentropy(rs,Y)
loss = loss.mean()

IoU = T.and_(T.ge(rs, 0.5),T.ge(Y,0.5)).sum(axis=(1,2)) * \
      T.minimum(T.inv(T.or_(T.ge(rs, 0.5),T.ge(Y,0.5)).sum(axis=(1,2))), 128.0 * 256.0)
IoU = IoU.mean()

test_loss = binary_crossentropy(frs,Y)
test_loss = test_loss.mean()

test_IoU = T.and_(T.ge(frs, 0.5),T.ge(Y,0.5)).sum(axis=(1,2)) * \
           T.minimum(T.inv(T.or_(T.ge(frs, 0.5),T.ge(Y,0.5)).sum(axis=(1,2))), 128.0 * 256.0)
test_IoU = test_IoU.mean()

loss_function = theano.function(inputs               = [X,Y,P],
                                outputs              = [loss,IoU],
                                allow_input_downcast = True)

test_loss_function = theano.function(inputs               = [X,Y],
                                     outputs              = [test_loss,test_IoU],
                                     allow_input_downcast = True)

params = get_all_params(RoadSegment, trainable=True)

updates = adam(loss,
               params,
               learning_rate = lr)

train_network = theano.function(inputs               = [X,Y,P,lr],
                                outputs              = [loss,IoU],
                                updates              = updates,
                                allow_input_downcast = True)

all_errs = []

load_weights()
k=0

if train:
    dataset = zip(*[trX, trY])
else:
    dataset = zip(*[teX, teY])

for tx, ty in dataset:
    k += 1

    bce_, iou_ = test_loss_function(tx.reshape(1,128,256,3),
                                    ty.reshape(1,128,256))

    all_errs.append((bce_,iou_))

    print 'IoU: %f, BCE: %f' % (iou_, bce_)

    if train:
        plt.imsave('plot_results/train/images/x_%05d.png'%(k), tx.reshape(128,256,3))
        plt.imsave('plot_results/train/gt/y_%05d.png'%(k), ty.astype(np.float32).reshape(128,256))
        plt.imsave('plot_results/train/result/im_%05d.png'%(k), road_segment(tx.reshape(1,128,256,3)).reshape(128,256))
    else:
        plt.imsave('plot_results/test/images/x_%05d.png'%(k), tx.reshape(128,256,3))
        plt.imsave('plot_results/test/gt/y_%05d.png'%(k), ty.astype(np.float32).reshape(128,256))
        plt.imsave('plot_results/test/result/im_%05d.png'%(k), road_segment(tx.reshape(1,128,256,3)).reshape(128,256))

        error = np.concatenate([ty.astype(np.float32).reshape(128,256,1),
                                1.0 * (road_segment(tx.reshape(1,128,256,3)).reshape(128,256,1) > 0.5),
                                np.zeros((128,256,1))], axis=2)

        mask = 0.5 * (tx.astype(np.float32) / 255) + 0.5 * np.repeat(road_segment(tx.reshape(1,128,256,3)).reshape(128,256,1) > 0.5,3,axis=2)

        plt.imsave('plot_results/test/errors/er_%05d.png'%(k),error)
        plt.imsave('plot_results/test/mask/ma_%05d.png'%(k),mask)

all_errs = np.array(all_errs)

print 'all errors:'
print '    iou:%f' % (np.mean(all_errs[all_errs[:,1] > 0.1], axis=0)[1])
print '    bce:%f' % (np.mean(all_errs[all_errs[:,1] > 0.1], axis=0)[0])
