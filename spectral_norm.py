import tensorflow as tf

from tensorlayer.layers.core import Layer
from tensorlayer.layers.core import LayersConfig

from tensorlayer import tl_logging as logging
from tensorlayer.layers.utils import get_collection_trainable
from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import private_method

"""
    author: xuchao
    modified: 2018-7-21 21:00

"""


class SpectralConv2dLayer(Layer):
    """
    The :class:`SpectralConv2dLayer` class is a 2D CNN layer with spectral norm, see `tf.nn.conv2d <https://www.tensorflow.org/versions/master/api_docs/python/nn.html#conv2d>`__.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer.
    act : activation function
        The activation function of this layer.
    shape : tuple of int
        The shape of the filters: (filter_height, filter_width, in_channels, out_channels).
    strides : tuple of int
        The sliding window strides of corresponding input dimensions.
        It must be in the same order as the ``shape`` parameter.
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    W_init : initializer
        The initializer for the weight matrix.
    b_init : initializer or None
        The initializer for the bias vector. If None, skip biases.
    W_init_args : dictionary
        The arguments for the weight matrix initializer.
    b_init_args : dictionary
        The arguments for the bias vector initializer.
    use_cudnn_on_gpu : bool
        Default is False.
    use_sn: bool
        Default is False.
    data_format : str
        "NHWC" or "NCHW", default is "NHWC".
    name : str
        A unique layer name.

    Notes
    -----
    - shape = [h, w, the number of output channel of previous layer, the number of output channels]
    - the number of output channel of a layer is its last dimension.

    Examples
    --------
    With TensorLayer
    Without TensorLayer, you can implement 2D convolution as follow.

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            act=None,
            shape=(5, 5, 1, 100),
            strides=(1, 1, 1, 1),
            padding='SAME',
            W_init=tf.truncated_normal_initializer(stddev=0.02),
            b_init=tf.constant_initializer(value=0.0),
            W_init_args=None,
            b_init_args=None,
            use_cudnn_on_gpu=True,
            use_sn=False,
            data_format=None,
            name='cnn_layer',
    ):
        super(SpectralConv2dLayer, self
             ).__init__(prev_layer=prev_layer, act=act, W_init_args=W_init_args, b_init_args=b_init_args, name=name)

        logging.info(
            "SpectralConv2dLayer %s: shape: %s strides: %s pad: %s act: %s spectral: %s" % (
                self.name, str(shape), str(strides), padding, self.act.__name__
                if self.act is not None else 'No Activation', use_sn
            )
        )
        # check layer name (fixed)
        Layer.__init__(self, prev_layer=prev_layer, name=name)

        # the input of this layer is the output of previous layer (fixed)
        self.inputs = prev_layer.outputs

        with tf.variable_scope(name):
            W = tf.get_variable(
                name='W_conv2d', shape=shape, initializer=W_init, dtype=LayersConfig.tf_dtype, **self.W_init_args
            )
            if use_sn:
                self.outputs = tf.nn.conv2d(
                    self.inputs, self._Spectral_norm(W), strides=strides, padding=padding, use_cudnn_on_gpu=use_cudnn_on_gpu,
                    data_format=data_format)

            else:
                self.outputs = tf.nn.conv2d(
                    self.inputs, W, strides=strides, padding=padding, use_cudnn_on_gpu=use_cudnn_on_gpu,
                    data_format=data_format)
            if b_init:
                b = tf.get_variable(
                    name='b_conv2d', shape=(shape[-1]), initializer=b_init, dtype=LayersConfig.tf_dtype,
                    **self.b_init_args
                )

                self.outputs = tf.nn.bias_add(self.outputs, b, name='bias_add')

            self.outputs = self._apply_activation(self.outputs)

        new_variables = get_collection_trainable(self.name)
        self._add_layers(self.outputs)
        self._add_params(new_variables)


    @private_method
    def _Spectral_norm(self, w):
        w_shape = w.shape.as_list()
        w = tf.reshape(w, [-1, w_shape[-1]])

        u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

        u_hat = u
        # v_hat = None

        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_, axis=None, epsilon=1e-12)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_, axis=None, epsilon=1e-12)

        sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
        w_norm = w / sigma

        with tf.control_dependencies([u.assign(u_hat)]):
            w_norm = tf.reshape(w_norm, w_shape)

        return w_norm



class SpectralDenseLayer(Layer):
    """The :class:`SpectralDenseLayer` class is a fully connected layer with spectral norm

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer.
    n_units : int
        The number of units of this layer.
    act : activation function
        The activation function of this layer.
    W_init : initializer
        The initializer for the weight matrix.
    b_init : initializer or None
        The initializer for the bias vector. If None, skip biases.
    W_init_args : dictionary
        The arguments for the weight matrix initializer.
    b_init_args : dictionary
        The arguments for the bias vector initializer.
    use_sn: bool
        Default is False.
    name : a str
        A unique layer name.

    Notes
    -----
    If the layer input has more than two axes, it needs to be flatten by using :class:`FlattenLayer`.

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            n_units=100,
            act=None,
            W_init=tf.truncated_normal_initializer(stddev=0.1),
            b_init=tf.constant_initializer(value=0.0),
            W_init_args=None,
            b_init_args=None,
            use_sn=False,
            name='dense',
    ):

        super(SpectralDenseLayer, self
             ).__init__(prev_layer=prev_layer, act=act, W_init_args=W_init_args, b_init_args=b_init_args, name=name)

        logging.info(
            "DenseLayer  %s: %d %s spectral: %s" %
            (self.name, n_units, self.act.__name__ if self.act is not None else 'No Activation', use_sn)
        )
        # check layer name (fixed)
        Layer.__init__(self, prev_layer=prev_layer, name=name)

        # the input of this layer is the output of previous layer (fixed)
        self.inputs = prev_layer.outputs

        self.n_units = n_units

        if self.inputs.get_shape().ndims != 2:
            raise AssertionError("The input dimension must be rank 2, please reshape or flatten it")

        n_in = int(self.inputs.get_shape()[-1])

        with tf.variable_scope(name):
            W = tf.get_variable(
                name='W', shape=(n_in, n_units), initializer=W_init, dtype=LayersConfig.tf_dtype, **self.W_init_args
            )

            if use_sn:  # if set spectral norm True
                self.outputs = tf.matmul(self.inputs, self._Spectral_norm(W))

            else:
                self.outputs = tf.matmul(self.inputs, W)

            if b_init is not None:
                try:
                    b = tf.get_variable(
                        name='b', shape=(n_units), initializer=b_init, dtype=LayersConfig.tf_dtype, **self.b_init_args
                    )
                except Exception:  # If initializer is a constant, do not specify shape.
                    b = tf.get_variable(name='b', initializer=b_init, dtype=LayersConfig.tf_dtype, **self.b_init_args)

                self.outputs = tf.nn.bias_add(self.outputs, b, name='bias_add')

            self.outputs = self._apply_activation(self.outputs)

        new_variables = get_collection_trainable(self.name)
        self._add_layers(self.outputs)
        self._add_params(new_variables)

    @private_method
    def _Spectral_norm(self, w):
        w_shape = w.shape.as_list()
        w = tf.reshape(w, [-1, w_shape[-1]])

        u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

        u_hat = u
        # v_hat = None

        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_, axis=None, epsilon=1e-12)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_, axis=None, epsilon=1e-12)

        sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
        w_norm = w / sigma

        with tf.control_dependencies([u.assign(u_hat)]):
            w_norm = tf.reshape(w_norm, w_shape)

        return w_norm


