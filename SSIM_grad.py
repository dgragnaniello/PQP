import numpy as np
import tensorflow as tf

def structural_similarity_tf(X, Y, K1=0.01, K2=0.03, win_size=7, data_range=255.0, use_sample_covariance=False):
    nch = tf.shape(X)[-1]
    kernel = tf.cast(tf.fill([win_size, win_size, nch, 1], 1 / (win_size * win_size)), X.dtype)
    filter_args = {'filter': kernel, 'strides': [1] * 4, 'padding': 'VALID', 'data_format': 'NHWC'}

    # filter has already normalized by NP
    if use_sample_covariance:
        cov_norm = win_size * win_size / (win_size * win_size - 1)  # sample covariance
    else:
        cov_norm = 1.0  # population covariance to match Wang et. al. 2004

    # compute means
    ux = tf.nn.depthwise_conv2d(X, **filter_args)
    uy = tf.nn.depthwise_conv2d(Y, **filter_args)

    # compute variances and covariances
    uxx = tf.nn.depthwise_conv2d(X * X, **filter_args)
    uyy = tf.nn.depthwise_conv2d(Y * Y, **filter_args)
    uxy = tf.nn.depthwise_conv2d(X * Y, **filter_args)
    vxx = cov_norm * (uxx - ux * ux)
    vyy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)

    R = data_range
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2

    S = (2 * ux * uy + C1) * (2 * vxy + C2) / \
        ((ux * ux + uy * uy + C1) * (vxx + vyy + C2))

    ssim = tf.reduce_mean(S, axis=3)
    ssim = tf.reduce_mean(ssim, axis=2)
    ssim = tf.reduce_mean(ssim, axis=1)
    ssim = tf.reduce_sum(ssim)
    return ssim

def build_ssim_graph(batch_shape=(None, 112, 112, 3)):
    X_tf = tf.placeholder(tf.float32, shape=batch_shape, name='X')
    Y_tf = tf.placeholder(tf.float32, shape=batch_shape, name='Y')
    S_tf = structural_similarity_tf(X_tf, Y_tf)
    G_tf, = tf.gradients(S_tf, Y_tf)
    Ga_tf = tf.reduce_max(tf.abs(G_tf), axis=-1, keep_dims=True)
    config_tf = tf.ConfigProto()
    config_tf.gpu_options.allow_growth = True
    tf_sess = tf.Session(config=config_tf)

    def ssim_grad_tf(X, Y):
        if X.ndim == 3 and Y.ndim == 3:
            X = np.expand_dims(X, axis=0)
            Y = np.expand_dims(Y, axis=0)
            G = tf_sess.run(Ga_tf, {X_tf: X, Y_tf: Y})
            return G[0]
        else:
            assert X.ndim == Y.ndim == 4
            return tf_sess.run(Ga_tf, {X_tf: X, Y_tf: Y})

    grad_ssim_tf = lambda X, Y: ssim_grad_tf(X, Y)
    return grad_ssim_tf, tf_sess
