"""
Example script for training a HyperMorph model to tune the
regularization weight hyperparameter.

If you use this code, please cite the following:

    A Hoopes, M Hoffmann, B Fischl, J Guttag, AV Dalca. 
    HyperMorph: Amortized Hyperparameter Learning for Image Registration
    IPMI: Information Processing in Medical Imaging. 2021. https://arxiv.org/abs/2101.01035

Copyright 2020 Andrew Hoopes

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""

import os
import random
import argparse
import numpy as np
import tensorflow as tf
import voxelmorph as vxm
from tensorflow.keras import backend as K
from datetime import datetime


#from tqdm.keras import TqdmCallback
#tf.compat.v1.disable_eager_execution()

from tensorflow.python.framework.ops import disable_eager_execution
#disable_eager_execution()
# tf.executing_eagerly()
# tf.eagerly()
# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
parser.add_argument('--img-list', required=True, help='line-seperated list of training files')
parser.add_argument('--img-prefix', help='optional input image file prefix')
parser.add_argument('--img-suffix', help='optional input image file suffix')
parser.add_argument('--hyper_gen', help='atlas filename')
parser.add_argument('--model-dir', default='models',
                    help='model output directory (default: models)')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')
parser.add_argument('--test-reg', nargs=3,
                    help='example registration pair and result (moving fixed moved) to test')

# training parameters
parser.add_argument('--gpu', default='2', help='GPU ID numbers (default: 0)')
parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of training epochs (default: 6000)')
parser.add_argument('--steps-per-epoch', type=int, default=500,
                    help='steps per epoch (default: 100)')
parser.add_argument('--load-weights', help='optional weights file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=0,
                    help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-6, help='learning rate (default: 1e-4)')

# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+',
                    help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+',
                    help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
parser.add_argument('--int-steps', type=int, default=7,
                    help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=2,
                    help='flow downsample factor for integration (default: 2)')

# loss hyperparameters
parser.add_argument('--activ', default='sigmoid')
parser.add_argument('--type', type=int, default=1)
parser.add_argument('--hyper_num', type=int, default=3)


parser.add_argument('--image-loss', default='dice',
                    help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--image-sigma', type=float, default=0.05,
                    help='estimated image noise for mse image scaling (default: 0.05)')
parser.add_argument('--oversample-rate', type=float, default=1,
                    help='hyperparameter end-point over-sample rate (default 0.2)')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
gpu_avilable = tf.config.experimental.list_physical_devices('GPU')
print(gpu_avilable)

logdir = args.model_dir+"/logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# no need to append an extra feature axis if data is multichannel
add_feat_axis = not args.multichannel
    # scan-to-scan generator
base_generator = vxm.generators.single_mods_gen(
        args.img_list,phase='train', batch_size=args.batch_size, add_feat_axis=add_feat_axis, type=args.type)

base_generator_valid = vxm.generators.single_mods_gen(
        args.img_list,phase='valid', batch_size=args.batch_size, add_feat_axis=add_feat_axis, type=args.type)

# random hyperparameter generator

hyperps = np.load('hyperp.npy')





validation_steps=100
#validation_steps=np.ceil(len(valid_files)/args.batch_size)

# extract shape and number of features from sampled input
sample_shape = next(base_generator)[0][0].shape
inshape = sample_shape[1:-1]
nfeats = sample_shape[-1]

# prepare model folder
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)

# unet architecture
enc_nf = args.enc if args.enc else [16, 32, 32, 32]
dec_nf = args.dec if args.dec else [32, 32, 32, 32, 16]

# prepare model checkpoint save path
save_filename = os.path.join(model_dir, '{epoch:04d}.h5')

# tensorflow device handling
device, nb_devices = vxm.tf.utils.setup_device(args.gpu)


def test(model_test):
    for i, data in enumerate(base_generator_valid):
        hyper_val = random_hyperparam(args.hyper_num)
        hyp = np.array([hyper_val for _ in range(args.batch_size)])
        inputs, outputs = data
        #inputs = (*inputs, hyp)

        layer_model = tf.keras.Model(inputs=model_test.input, outputs=model.layers[93].output)
        feature1 = layer_model.predict(inputs)
        layer_model = tf.keras.Model(inputs=model_test.input, outputs=model.layers[94].output)
        feature2 = layer_model.predict(inputs)
        predicted = model_test.predict(inputs)
        predicted=tf.keras.activations.sigmoid(predicted)
        import matplotlib.pyplot as plt
        plt.subplot(2,2,1)
        plt.imshow(feature1[0,:,:,48,0])
        plt.subplot(2, 2, 2)
        plt.imshow(feature2[0, :, :, 48, 0])
        plt.subplot(2, 2, 3)
        plt.imshow(outputs[0][0, :, :, 48, 0])
        plt.subplot(2, 2, 4)
        plt.imshow(predicted[0, :, :, 48, 0])
        plt.show()
        #vxm.py.utils.save_volfile(predicted, 'example.nii')



with tf.device(device):

#    model = vxm.networks.UnetSingle(
#        inshape=inshape,
#        nb_unet_features=[enc_nf, dec_nf],
#        src_feats=nfeats,
#        trg_feats=nfeats,
#        unet_half_res=False,
#        activate=args.activ)

    model = vxm.nnUnet.NNUnet(in_shape=inshape, n_channels=1, n_classes=1, pocket=False, deep_supervision=False)
    #print(model.summary())
    #load initial weights (if provided)
    if args.load_weights:
        model.load_weights(os.path.join(model_dir, '{:04d}.h5'.format(int(args.load_weights))))
        print('loading weights from {:04d}.h5'.format(int(args.load_weights)))

    #test(model)
    # prepare image loss
    #hyper_val=model.references.hyper_val
    if args.image_loss == 'dice':
        #image_loss_func = vxm.losses.HyperBinaryDiceLoss(hyper_val, args.mod).loss
        image_loss_func = vxm.losses.BinaryDiceLoss().loss
        ce_loss = tf.keras.losses.BinaryCrossentropy(
            from_logits=True, label_smoothing=0, name='binary_crossentropy'
                )
    elif args.image_loss == 'mse':
        scaling = 1.0 / (args.image_sigma ** 2)
        image_loss_func = lambda x1, x2: scaling * K.mean(K.batch_flatten(K.square(x1 - x2)), -1)
    else:
        raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

    # prepare loss functions and compile model

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr), loss=[image_loss_func])

    save_callback = tf.keras.callbacks.ModelCheckpoint(save_filename, save_freq='epoch', save_best_only=True)
    logger = tf.keras.callbacks.CSVLogger(
        os.path.join(model_dir,'LOGGER.TXT'), separator=',', append=False
    )
    training_history = model.fit(base_generator,initial_epoch=args.initial_epoch,
                        epochs=args.epochs,
                        steps_per_epoch=args.steps_per_epoch,
                        callbacks=[save_callback, logger, tensorboard_callback], verbose=1,
                        validation_steps=validation_steps,
                        validation_data=base_generator_valid)

    # save final weights
    model.save(save_filename.format(epoch=args.epochs))
    print("Average test loss: ", np.average(training_history.history['loss']))
    
    # save an example registration across lambda values

