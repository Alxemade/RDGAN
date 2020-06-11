from tensorlayer.layers import *
from utils import *
from spectral_norm import *
import tensorflow as tf
import tensorlayer as tl
import time
"""
func: discriminator
args:
    input_images: this is the input image 
    is_train: we set True when we on the training time and set False when on test time 
    reuse: set False when we use on the training time and set True when on the test time  
    outputs: net_ho: the probability of image whether it is fake or true range from 0 to 1 
    outputs: logits: the output images 

"""
"""
patchGan: The 70 × 70 discriminator architecture is:
    C64-C128-C256-C512-C1024-C2048-C1024-[C512 + Res128-Res128-Res512]-DesNets
After the last layer, a convolution is applied to map   
After the last layer, a convolution is applied to map to a 1
dimensional output, followed by a Sigmoid function. As an
exception to the above notation, BatchNorm is not applied
to the first C64 layer. All ReLUs are leaky, with slope 0.2.


Attention: The discriminator does
not use batch normalization because that would conflict with the gradient penalty,
meaning instance normalization or layer normalization can still be used without issue.
The authors suggest that layer normalization is used if any. 

"""


def discriminator(input_images, lossfunciton, is_train=True, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None
    gamma_init = tf.random_normal_initializer(1., 0.02)
    df_dim = 64

    with tf.variable_scope("discriminator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net_in = InputLayer(input_images,
                            name='input')  # net_in: [batch, 256, 256, 1]

        net_h0 = Conv2d(net_in, df_dim, (4, 4), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),
                        padding='SAME', W_init=w_init, name='h0/conv2d')  # net_h0: [batch, 128, 128, 64]

        net_h1 = Conv2d(net_h0, df_dim * 2, (4, 4), (2, 2), act=None,
                        padding='SAME', W_init=w_init, b_init=b_init, name='h1/conv2d') # net_h1: [batch, 64, 64, 128]
        net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='h1/batchnorm') # net_h1: [bt, 64, 64, 128]

        net_h2 = Conv2d(net_h1, df_dim * 4, (4, 4), (2, 2), act=None,
                        padding='SAME', W_init=w_init, b_init=b_init, name='h2/conv2d')  # net_h2: [bt, 32, 32, 256]
        net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='h2/batchnorm') # net_h2: [bt, 32, 32, 256]

        net_h3 = Conv2d(net_h2, df_dim * 8, (4, 4), (2, 2), act=None,
                        padding='SAME', W_init=w_init, b_init=b_init, name='h3/conv2d') # net_h3: [bt, 16, 16, 512]
        net_h3 = BatchNormLayer(net_h3, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='h3/batchnorm')  # net_h3: [bt, 16, 16, 512]

        net_h4 = Conv2d(net_h3, df_dim * 16, (4, 4), (2, 2), act=None,
                        padding='SAME', W_init=w_init, b_init=b_init, name='h4/conv2d')  # net_h4: [bt, 8, 8, 1024]
        net_h4 = BatchNormLayer(net_h4, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='h4/batchnorm') # net_h4: [bt, 8, 8, 1014]

        net_h5 = Conv2d(net_h4, df_dim * 32, (4, 4), (2, 2), act=None,
                        padding='SAME', W_init=w_init, b_init=b_init, name='h5/conv2d')  # net_h5: [bt, 4, 4, 2048]
        net_h5 = BatchNormLayer(net_h5, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='h5/batchnorm')  # net_h5: [bt, 4, 4, 2048]

        net_h6 = Conv2d(net_h5, df_dim * 16, (1, 1), (1, 1), act=None,
                        padding='SAME', W_init=w_init, b_init=b_init, name='h6/conv2d')  # net_h6: [bt, 4, 4, 1024]
        net_h6 = BatchNormLayer(net_h6, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='h6/batchnorm') # net_h6: [bt, 4, 4, 1024]

        net_h7 = Conv2d(net_h6, df_dim * 8, (1, 1), (1, 1), act=None,
                        padding='SAME', W_init=w_init, b_init=b_init, name='h7/conv2d') # net_h7: [bt, 4, 4, 512]
        net_h7 = BatchNormLayer(net_h7, is_train=is_train, gamma_init=gamma_init, name='h7/batchnorm') # net_h7: [bt, 4, 4, 512]

        net = AtrousConv2dLayer(net_h7, n_filter=df_dim*2, filter_size=(1, 1), rate=2, act=None, padding='SAME',
                                         W_init=w_init, b_init=b_init, name='h7_res/conv2d')
        net = BatchNormLayer(net, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train,
                                      gamma_init=gamma_init, name='h7_res/batchnorm')
        net = AtrousConv2dLayer(net, n_filter=df_dim * 2, filter_size=(3, 3), rate=4, act=None,
                                         padding='SAME', W_init=w_init, b_init=b_init, name='h7_res/conv2d2')
        net = BatchNormLayer(net, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train,
                                      gamma_init=gamma_init, name='h7_res/batchnorm2')
        net = AtrousConv2dLayer(net, n_filter=df_dim * 8, filter_size=(3, 3), rate=8, act=None,
                                         padding='SAME', W_init=w_init, b_init=b_init, name='h7_res/conv2d3')
        net = BatchNormLayer(net, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train,
                                      gamma_init=gamma_init, name='h7_res/batchnorm3')

        net_ho = FlattenLayer(net, name='output/flatten')  # net_ho: [bt*4*4*512]
        net_ho = DenseLayer(net_ho, n_units=1, act=tf.identity, W_init=w_init, name='output/dense') # net_h0: 1
        logits = net_ho.outputs
        if (lossfunciton == 'gan_loss') or (lossfunciton == 'lsgan_loss'):
            net_ho.outputs = tf.nn.sigmoid(net_ho.outputs)
        elif lossfunciton == 'wgan_loss':  # Wasserstein GAN doesn't need the sigmoid output
            pass
        else:
            raise Exception("Unknow lossfunction")
    return net_ho, logits


def u_net_bn(x, is_train=False, reuse=False, is_refine=False):
    """image to image translation via conditional adversarial learning"""
    w_init = tf.truncated_normal_initializer(stddev=0.01)
    # w_init = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)  # xavier init
    # w_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False, seed=None,
    #                                                         dtype=tf.float32)   #  MSRA init
    # tf.glorot_uniform_initializer()
    b_init = tf.constant_initializer(value=0.0)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    df_dim = 64
    with tf.variable_scope("u_net", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        """input layer"""
        inputs = InputLayer(x, name='input')  # inputs: [batch, 256, 256, 1]
        """
        Let Ck denote a Convolution-BatchNorm-ReLU layer with k filters.
        encoder layer: C64-C128-C256-C512-C512-C512-C512-C512 
        Norm is not applied to the first C64 layer in the encoder.All ReLUs in the encoder are leaky, with slope 0.2
        Convolutions in the encoder, and in the discriminator, downsample by a factor of 2,
        All convolutions are 4 × 4 spatial filters applied with stride 2 instead of the origin u-net 3 x 3 with stride 1
        At the same time because of using this trick we don't need any pooling layer, we use fully convolutions networks.
        """
        conv1 = Conv2d(inputs, df_dim, (4, 4), (2, 2), act=None, padding='SAME',
                       W_init=w_init, b_init=b_init, name='conv1')  # conv1: [batch, 128, 128, 64]不使用BN,避免模型震荡
        conv2 = Conv2d(conv1, df_dim*2, (4, 4), (2, 2), act=None, padding='SAME',
                       W_init=w_init, b_init=b_init, name='conv2')   # conv2: [batch, 64, 64, 128]
        conv2 = BatchNormLayer(conv2, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='bn2')  # conv2: [batch, 64, 64, 128]

        conv3 = Conv2d(conv2, df_dim*4, (4, 4), (2, 2), act=None, padding='SAME',
                       W_init=w_init, b_init=b_init, name='conv3')  # conv3: [batch, 32, 32, 256]
        conv3 = BatchNormLayer(conv3, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='bn3')  # conv3: [batch, 32, 32, 256]

        conv4 = Conv2d(conv3, df_dim*8, (4, 4), (2, 2), act=None, padding='SAME',
                       W_init=w_init, b_init=b_init, name='conv4')  # conv4: [batch, 16, 16, 512]
        conv4 = BatchNormLayer(conv4, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='bn4')  # conv4: [batch, 16, 16, 512]

        conv5 = Conv2d(conv4, df_dim*8, (4, 4), (2, 2), act=None, padding='SAME',
                       W_init=w_init, b_init=b_init, name='conv5')  # conv5: [batch, 8, 8, 512]
        conv5 = BatchNormLayer(conv5, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='bn5')  # conv5: [batch, 8, 8, 512]

        conv6 = Conv2d(conv5, df_dim*8, (4, 4), (2, 2), act=None, padding='SAME',
                       W_init=w_init, b_init=b_init, name='conv6')  # conv6: [batch, 4, 4, 512]
        conv6 = BatchNormLayer(conv6, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='bn6')  # conv6: [batch, 4, 4, 512]

        conv7 = Conv2d(conv6, df_dim*8, (4, 4), (2, 2), act=None, padding='SAME',
                       W_init=w_init, b_init=b_init, name='conv7')  # conv7: [batch, 2, 2, 512]
        conv7 = BatchNormLayer(conv7, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='bn7')  # conv7: [batch, 2, 2, 512]

        conv8 = Conv2d(conv7, df_dim*8, (4, 4), (2, 2), act=None, padding='SAME',
                       W_init=w_init, b_init=b_init, name='conv8')   # conv8: [batch, 1, 1, 512] without BN

        conv8 = BatchNormLayer(conv8, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='bn8')
        # Dilated conv from here
        dilate_conv1 = AtrousConv2dLayer(conv8, n_filter=df_dim*8, filter_size=(3, 3), rate=2, act=None, padding='SAME',
                                         W_init=w_init, b_init=b_init, name='atrous_2d1')
        dilate_conv1 = BatchNormLayer(dilate_conv1, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train,
                                      gamma_init=gamma_init, name='atrous_bn1')
        dilate_conv2 = AtrousConv2dLayer(dilate_conv1, n_filter=df_dim * 8, filter_size=(3, 3), rate=4, act=None,
                                         padding='SAME', W_init=w_init, b_init=b_init, name='atrous_2d2')
        dilate_conv2 = BatchNormLayer(dilate_conv2, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train,
                                      gamma_init=gamma_init, name='atrous_bn2')
        dilate_conv3 = AtrousConv2dLayer(dilate_conv2, n_filter=df_dim * 8, filter_size=(3, 3), rate=8, act=None,
                                         padding='SAME', W_init=w_init, b_init=b_init, name='atrous_2d3')
        dilate_conv3 = BatchNormLayer(dilate_conv3, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train,
                                      gamma_init=gamma_init, name='atrous_bn3')
        #  Dilated conv end here
        conv9 = ElementwiseLayer(layers=[conv8, dilate_conv3], combine_fn=tf.add, name='h9/add')  # net_h8: [bt, 4, 4, 512]
        conv9.outputs = tl.act.lrelu(conv9.outputs, 0.2)
        conv9.outputs = tl.act.ramp(conv9.outputs, v_min=-1, v_max=1)
        """ 
        CDk denotes a a DeConvolution-BN-ReLU layer
        in the decoder they upsample by a factor of 2.
        U-Net decoder:C512-CD1024-CD1024-CD1024-CD512-CD256-CD128-CD64-CD64-C1
        ReLUs in the decoder are not leaky and the last activation is tanh to scale the outputs to [-1,1]
        """
        
        up7 = DeConv2d(conv9, df_dim*8, (4, 4), out_size=(2, 2), strides=(2, 2), padding='SAME',
                       act=None, W_init=w_init, b_init=b_init, name='deconv7')  # up7: [batch, 2, 2, 512]
        up7 = BatchNormLayer(up7, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn7')  # up7: [batch, 2, 2, 512]

        up6 = ConcatLayer([up7, conv7], concat_dim=3, name='concat6')  # up6: [batch, 2, 2, 512+512]
        up6 = DeConv2d(up6, df_dim*16, (4, 4), out_size=(4, 4), strides=(2, 2), padding='SAME',
                       act=None, W_init=w_init, b_init=b_init, name='deconv6')  # up6: [batch, 4, 4, 1024]
        up6 = BatchNormLayer(up6, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn6')  # up6: [batch, 4, 4, 1024]

        up5 = ConcatLayer([up6, conv6], concat_dim=3, name='concat5')  # up5: [batch, 4, 4, 512+1024]
        up5 = DeConv2d(up5, df_dim*16, (4, 4), out_size=(8, 8), strides=(2, 2), padding='SAME',
                       act=None,  b_init=b_init, name='deconv5')  # up5: [batch, 8, 8, 1024]
        up5 = BatchNormLayer(up5, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn5')  # up5: [batch, 8, 8, 1024]

        up4 = ConcatLayer([up5, conv5], concat_dim=3, name='concat4')  # up4: [batch, 8, 8, 1024+512]
        up4 = DeConv2d(up4, df_dim*16, (4, 4), out_size=(16, 16), strides=(2, 2), padding='SAME',
                       act=None, W_init=w_init, b_init=b_init, name='deconv4')  # up4: [batch, 16, 16, 1024]
        up4 = BatchNormLayer(up4, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn4')   # up4: [batch, 16, 16, 1024]

        up3 = ConcatLayer([up4, conv4], concat_dim=3, name='concat3')   # up3： [batch, 16, 16, 1024+512]
        up3 = DeConv2d(up3, df_dim*4, (4, 4), out_size=(32, 32), strides=(2, 2), padding='SAME',
                       act=None, W_init=w_init, b_init=b_init, name='deconv3')  # up3: [batch, 32, 32, 256]
        up3 = BatchNormLayer(up3, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn3')  # up3: [batch, 32, 32, 256]

        up2 = ConcatLayer([up3, conv3], concat_dim=3, name='concat2')  # up2: [batch, 32, 32, 256+256]
        up2 = DeConv2d(up2, df_dim*2, (4, 4), out_size=(64, 64), strides=(2, 2), padding='SAME',
                       act=None, W_init=w_init, b_init=b_init, name='deconv2')  # up2: [batch, 64, 64, 128]
        up2 = BatchNormLayer(up2, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn2')  # up2: [batch, 64, 64, 128]

        up1 = ConcatLayer([up2, conv2], concat_dim=3, name='concat1')  # up1: [batch, 64, 64, 128 + 128]
        up1 = DeConv2d(up1, df_dim, (4, 4), out_size=(128, 128), strides=(2, 2), padding='SAME',
                       act=None, W_init=w_init, b_init=b_init, name='deconv1')  # up1: [batch, 128, 128, 64]
        up1 = BatchNormLayer(up1, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn1')  # up1: [batch, 128, 128, 64]

        up0 = ConcatLayer([up1, conv1], concat_dim=3, name='concat0')  # up0: [batch, 128, 128, 64+64]
        up0 = DeConv2d(up0, df_dim, (4, 4), out_size=(256, 256), strides=(2, 2), padding='SAME',
                       act=None, W_init=w_init, b_init=b_init, name='deconv0')  # up0: [batch, 256, 256, 64]
        up0 = BatchNormLayer(up0, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn0')  # up0: [batch, 256, 256, 64]

        if is_refine:
            out = Conv2d(up0, 1, (1, 1), act=tf.nn.tanh, name='out')
            out = ElementwiseLayer(layers=[out, inputs], combine_fn=tf.add, name='add_for_refine')
            out.outputs = tl.act.ramp(out.outputs, v_min=-1, v_max=1)
        else:
            out = Conv2d(up0, 1, (1, 1), act=tf.nn.tanh, name='out')

    return out

"""
func: vgg16_cnn_emb
args:
    t_image: the input image range from -1 to 1
    reuse: set False when we use on the training time and set True when on the test time 
    outs: network
"""


def vgg19_simple_api(rgb, reuse=False):
    VGG_MEAN = [103.939, 116.779, 123.68]
    with tf.variable_scope("VGG19", reuse=reuse) as vs:
        start_time = time.time()
        print("build model started")
        rgb_scaled = (rgb + 1) * 127.5
        # Convert RGB to BGR
        if tf.__version__ <= '0.11':
            red, green, blue = tf.split(3, 3, rgb_scaled)
        else:  # TF 1.0
            # print(rgb_scaled)
            red, green, blue = tf.split(rgb_scaled, 3, 3)

        print("red shape", red.get_shape().as_list()[1:])
        print("green shape", green.get_shape().as_list()[1:])
        print("blue shape", blue.get_shape().as_list()[1:])
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        if tf.__version__ <= '0.11':
            bgr = tf.concat(3, [
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ])
        else:
            bgr = tf.concat(
                [
                    blue - VGG_MEAN[0],
                    green - VGG_MEAN[1],
                    red - VGG_MEAN[2],
                ], axis=3)
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
        """ input layer """
        net_in = InputLayer(bgr, name='input')
        """ conv1 """
        network = Conv2d(net_in, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv1_1')
        network = Conv2d(network, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv1_2')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool1')
        """ conv2 """
        network = Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv2_1')
        network = Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv2_2')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool2')
        """ conv3 """
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv3_1')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv3_2')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv3_3')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv3_4')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool3')
        """ conv4 """
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv4_1')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv4_2')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv4_3')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv4_4')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME',
                            name='pool4')  # (batch_size, 14, 14, 512)
        conv = network
        """ conv5 """
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv5_1')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv5_2')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv5_3')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv5_4')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME',
                            name='pool5')  # (batch_size, 7, 7, 512)
        """ fc 6~8 """
        network = FlattenLayer(network, name='flatten')
        network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc6')
        network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc7')
        network = DenseLayer(network, n_units=1000, act=tf.identity, name='fc8')
        print("build model finished: %fs" % (time.time() - start_time))
        return conv, network


if __name__ == "__main__":
    pass

