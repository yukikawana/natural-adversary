import os, sys, time
sys.path.append(os.getcwd())

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.size'] = 12
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')

import numpy as np
import tensorflow as tf
import argparse

import tflib
#import tflib.mnist
import tflib.plot
import tflib.save_images
import tflib.ops.batchnorm
import tflib.ops.conv2d
import tflib.ops.deconv2d
import tflib.ops.linear
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout
from tensorflow.python.keras.layers import Activation, BatchNormalization, add, Reshape
from tensorflow.python.keras.layers import DepthwiseConv2D
slim = tf.contrib.slim
def relu6(x):
    return tf.keras.backend.relu(x, maxval=6)
from tensorflow.python.keras import backend as K


class MnistWganInv(object):
    def __init__(self, x_dim=784, z_dim=64, latent_dim=64, batch_size=80,
                 c_gp_x=10., lamda=0.1, output_path='./',training=True):
        with tf.variable_scope('wgan'):
            self.bn_params = {
                "decay": 0.99,
                "epsilon": 1e-5,
                "scale": True,
                "is_training": training
            }
            self.x_dim = x_dim
            self.z_dim = z_dim
            self.latent_dim = latent_dim
            self.batch_size = batch_size
            self.c_gp_x = c_gp_x
            self.lamda = lamda
            self.output_path = output_path

            self.gen_params = self.dis_params = self.inv_params = None

            self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
            self.x_p = self.generate(self.z)
            print("xp",self.x_p.shape)

            #self.x = tf.placeholder(tf.float32, shape=[None, self.x_dim])
            self.x = tf.placeholder(tf.float32, shape=[None, 512,31,10])
            self.z_p = self.invert(self.x)

            self.dis_x = self.discriminate(self.x)
            self.dis_x_p = self.discriminate(self.x_p,reuse=True)
            self.rec_x = self.generate(self.z_p,reuse=True)
            self.rec_z = self.invert(self.x_p,reuse=True)

            self.gen_cost = -tf.reduce_mean(self.dis_x_p)

            self.inv_cost = tf.reduce_mean(tf.square(self.x - self.rec_x))
            self.inv_cost += self.lamda * tf.reduce_mean(tf.square(self.z - self.rec_z))

            self.dis_cost = tf.reduce_mean(self.dis_x_p) - tf.reduce_mean(self.dis_x)

            alpha = tf.random_uniform(shape=[self.batch_size, 1], minval=0., maxval=1.)
            difference = self.x_p - self.x
            interpolate = self.x + alpha * difference
            gradient = tf.gradients(self.discriminate(interpolate), [interpolate])[0]
            slope = tf.sqrt(tf.reduce_sum(tf.square(gradient), axis=1))
            gradient_penalty = tf.reduce_mean((slope - 1.) ** 2)
            self.dis_cost += self.c_gp_x * gradient_penalty

            self.gen_train_op = tf.train.AdamOptimizer(
                learning_rate=1e-4, beta1=0.9, beta2=0.999).minimize(
                self.gen_cost, var_list=self.gen_params)
            self.inv_train_op = tf.train.AdamOptimizer(
                learning_rate=1e-4, beta1=0.9, beta2=0.999).minimize(
                self.inv_cost, var_list=self.inv_params)
            self.dis_train_op = tf.train.AdamOptimizer(
                learning_rate=1e-4, beta1=0.9, beta2=0.999).minimize(
                self.dis_cost, var_list=self.dis_params)



    def _conv_block(self,inputs, filters, kernel, strides,act=relu6):
        """Convolution Block
        This function defines a 2D convolution operation with BN and relu6.

        # Arguments
            inputs: Tensor, input tensor of conv layer.
            filters: Integer, the dimensionality of the output space.
            kernel: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window.
            strides: An integer or tuple/list of 2 integers,
                specifying the strides of the convolution along the width and height.
                Can be a single integer to specify the same value for
                all spatial dimensions.

        # Returns
            Output tensor.
        """

        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

        x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
        x = BatchNormalization(axis=channel_axis)(x)
        return Activation(act)(x)


    def _bottleneck(self,inputs, filters, kernel, t, s, r=False,act=relu6):
        """Bottleneck
        This function defines a basic bottleneck structure.

        # Arguments
            inputs: Tensor, input tensor of conv layer.
            filters: Integer, the dimensionality of the output space.
            kernel: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window.
            t: Integer, expansion factor.
                t is always applied to the input size.
            s: An integer or tuple/list of 2 integers,specifying the strides
                of the convolution along the width and height.Can be a single
                integer to specify the same value for all spatial dimensions.
            r: Boolean, Whether to use the residuals.

        # Returns
            Output tensor.
        """

        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        tchannel = K.int_shape(inputs)[channel_axis] * t

        x = self._conv_block(inputs, tchannel, (1, 1), (1, 1),act=act)

        x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation(act)(x)

        x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)
        x = BatchNormalization(axis=channel_axis)(x)

        if r:
            x = add([x, inputs])
        return x

    def _inverted_residual_block(self,inputs, filters, kernel, t, strides, n, upsample=False,act=relu6):
        """Inverted Residual Block
        This function defines a sequence of 1 or more identical layers.

        # Arguments
            inputs: Tensor, input tensor of conv layer.
            filters: Integer, the dimensionality of the output space.
            kernel: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window.
            t: Integer, expansion factor.
                t is always applied to the input size.
            s: An integer or tuple/list of 2 integers,specifying the strides
                of the convolution along the width and height.Can be a single
                integer to specify the same value for all spatial dimensions.
            n: Integer, layer repeat times.
        # Returns
            Output tensor.
        """

        input_shape = inputs.shape
        nf_input = input_shape[-1]
        if upsample:
            upsample_shape = tuple([int(x)*2 for x in  input_shape[1:3]])
            print(upsample_shape)
            shortcut = tf.image.resize_nearest_neighbor(inputs, upsample_shape) 

        x = self._bottleneck(inputs, filters, kernel, t, strides,act=act)

        for i in range(1, n):
            x = self._bottleneck(x, filters, kernel, t, 1, True,act=act)

        return x
    def generate(self, z, reuse=False):
        with tf.variable_scope('generate', reuse=reuse):
            net = slim.fully_connected(z, 32*2*160, activation_fn=None) # 4x4x512
            net = tf.reshape(net, [-1, 32, 2, 160])
            net = self._inverted_residual_block(net, 96, (3, 3), t=6, strides=1, n=1,upsample=True,act=tf.nn.relu)
            net = self._inverted_residual_block(net, 64, (3, 3), t=6, strides=1, n=1,upsample=True,act=tf.nn.relu)
            net = self._inverted_residual_block(net, 32, (3, 3), t=6, strides=1, n=1,upsample=True,act=tf.nn.relu)
            net = self._inverted_residual_block(net, 24, (3, 3), t=6, strides=1, n=1,upsample=True,act=tf.nn.relu)
            net = slim.batch_norm(net, activation_fn=tf.nn.relu, **self.bn_params)
            net = slim.conv2d(net, 3, kernel_size=[3,3], activation_fn=None)
            net = net[:,:,:-1,:]

            return net

    def discriminate(self, X, reuse=False):
        with tf.variable_scope('discriminate', reuse=reuse):
            print(X.shape)
            net = slim.conv2d(X, 32, [3,3], activation_fn=None) # 64x64x64
            net = self._inverted_residual_block(net, 24, (3, 3), t=6, strides=2, n=1,act=tf.nn.leaky_relu)
            net = self._inverted_residual_block(net, 32, (3, 3), t=6, strides=2, n=1,act=tf.nn.leaky_relu)
            net = self._inverted_residual_block(net, 64, (3, 3), t=6, strides=2, n=1,act=tf.nn.leaky_relu)
            net = self._inverted_residual_block(net, 96, (3, 3), t=6, strides=2, n=1,act=tf.nn.leaky_relu)
            net = slim.flatten(net)
            net = slim.fully_connected(net, 1, activation_fn=None)
            return net
 
    def invert(self, x,reuse=False):
        with tf.variable_scope('invert', reuse=reuse):
            net = slim.conv2d(x, 32, [3,3], activation_fn=None) # 64x64x64
            net = self._inverted_residual_block(net, 24, (3, 3), t=6, strides=2, n=1,act=tf.nn.leaky_relu)
            net = self._inverted_residual_block(net, 32, (3, 3), t=6, strides=2, n=1,act=tf.nn.leaky_relu)
            net = self._inverted_residual_block(net, 64, (3, 3), t=6, strides=2, n=1,act=tf.nn.leaky_relu)
            net = self._inverted_residual_block(net, 96, (3, 3), t=6, strides=2, n=1,act=tf.nn.leaky_relu)
            net = slim.flatten(net)
            net = slim.fully_connected(net, 130, activation_fn=None)
            net= tf.nn.leaky_relu(net)
            net = slim.fully_connected(net, self.z_dim, activation_fn=None)

            return net
 


    def train_gen(self, sess, x, z):
        _gen_cost, _ = sess.run([self.gen_cost, self.gen_train_op],
                                feed_dict={self.x: x, self.z: z})
        return _gen_cost

    def train_dis(self, sess, x, z):
        _dis_cost, _ = sess.run([self.dis_cost, self.dis_train_op],
                                feed_dict={self.x: x, self.z: z})
        return _dis_cost

    def train_inv(self, sess, x, z):
        _inv_cost, _ = sess.run([self.inv_cost, self.inv_train_op],
                                feed_dict={self.x: x, self.z: z})
        return _inv_cost

    def generate_from_noise(self, sess, noise, frame):
        samples = sess.run(self.x_p, feed_dict={self.z: noise})
        tflib.save_images.save_images(
            samples.reshape((-1, 28, 28)),
            os.path.join(self.output_path, 'examples/samples_{}.png'.format(frame)))
        return samples

    def reconstruct_images(self, sess, images, frame):
        reconstructions = sess.run(self.rec_x, feed_dict={self.x: images})
        comparison = np.zeros((images.shape[0] * 2, images.shape[1]),
                              dtype=np.float32)
        for i in xrange(images.shape[0]):
            comparison[2 * i] = images[i]
            comparison[2 * i + 1] = reconstructions[i]
        tflib.save_images.save_images(
            comparison.reshape((-1, 28, 28)),
            os.path.join(self.output_path, 'examples/recs_{}.png'.format(frame)))
        return comparison


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=80, help='batch size')
    parser.add_argument('--epoch', type=int, default=1000, help='epoch')
    parser.add_argument('--z_dim', type=int, default=1024, help='dimension of z')
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='latent dimension')
    parser.add_argument('--iterations', type=int, default=100000,
                        help='training steps')
    parser.add_argument('--dis_iter', type=int, default=5,
                        help='discriminator steps')
    parser.add_argument('--c_gp_x', type=float, default=10.,
                        help='coefficient for gradient penalty x')
    parser.add_argument('--lamda', type=float, default=.1,
                        help='coefficient for divergence of z')
    parser.add_argument('--output_path', type=str, default='./',
                        help='output path')
    parser.add_argument('--dataset_path', type=str, default='/workspace/imgsynth/hmpool5',
                        help='dataset path')
    args = parser.parse_args()


    # dataset iterator
    """
    train_gen, dev_gen, test_gen = tflib.mnist.load(args.batch_size, args.batch_size)

    def inf_train_gen():
        while True:
            for instances, labels in train_gen():
                yield instances

    _, _, test_data = tflib.mnist.load_data()
    fixed_images = test_data[0][:32]
    del test_data
    """

    tf.set_random_seed(326)
    np.random.seed(326)
    fixed_noise = np.random.randn(64, args.z_dim)

    mnistWganInv = MnistWganInv(
        x_dim=784, z_dim=args.z_dim, latent_dim=args.latent_dim,
        batch_size=args.batch_size, c_gp_x=args.c_gp_x, lamda=args.lamda,
        output_path=args.output_path)

    saver = tf.train.Saver(max_to_keep=1000)


    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    with tf.Session(config=config) as session:
        session.run(tf.global_variables_initializer())

        images = noise = gen_cost = dis_cost = inv_cost = None
        dis_cost_lst, inv_cost_lst = [], []
        data_files = glob(os.path.join(args.dataset, "*.npz"))
        data_files = sorted(data_files)
        data_files = np.array(data_files) # for tl.iterate.minibatches
        iteration=0
        for epoch in range(args.epoch):
            try:
                minibatch = tl.iterate.minibatches(inputs=data_files, targets=data_files, batch_size=args.batch_size, shuffle=True)
                while True:
                    iterations+=1
                    for i in range(args.dis_iter):
                        noise = np.random.randn(args.batch_size, args.z_dim)
                        batch_files,_ = minibatch.__next__()

                        dis_cost_lst += [mnistWganInv.train_dis(session, images, noise)]
                        inv_cost_lst += [mnistWganInv.train_inv(session, images, noise)]

                    gen_cost = mnistWganInv.train_gen(session, images, noise)
                    dis_cost = np.mean(dis_cost_lst)
                    inv_cost = np.mean(inv_cost_lst)

                    tflib.plot.plot('train gen cost', gen_cost)
                    tflib.plot.plot('train dis cost', dis_cost)
                    tflib.plot.plot('train inv cost', inv_cost)

                    if iteration % 100 == 99:
                        mnistWganInv.generate_from_noise(session, fixed_noise, iteration)
                        mnistWganInv.reconstruct_images(session, fixed_images, iteration)

                    if iteration % 1000 == 999:
                        save_path = saver.save(session, os.path.join(
                            args.output_path, 'models/model'), global_step=iteration)

                    if iteration % 1000 == 999:
                        dev_dis_cost_lst, dev_inv_cost_lst = [], []
                        for dev_images, _ in dev_gen():
                            noise = np.random.randn(args.batch_size, args.z_dim)
                            dev_dis_cost, dev_inv_cost = session.run(
                                [mnistWganInv.dis_cost, mnistWganInv.inv_cost],
                                feed_dict={mnistWganInv.x: dev_images,
                                           mnistWganInv.z: noise})
                            dev_dis_cost_lst += [dev_dis_cost]
                            dev_inv_cost_lst += [dev_inv_cost]
                        tflib.plot.plot('dev dis cost', np.mean(dev_dis_cost_lst))
                        tflib.plot.plot('dev inv cost', np.mean(dev_inv_cost_lst))

                    if iteration < 5 or iteration % 100 == 99:
                        tflib.plot.flush(os.path.join(args.output_path, 'models'))
            except StopIteration:
                pass
                

                tflib.plot.tick()


