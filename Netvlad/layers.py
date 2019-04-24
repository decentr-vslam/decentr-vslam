import tensorflow as tf

def netVLAD(inputs, num_clusters, assign_weight_initializer=None, 
            cluster_initializer=None, skip_postnorm=False):
    # D: number of (descriptor) dimensions.
    # s: soft-assignment.

    K = num_clusters
    D = inputs.get_shape()[-1]
    s = tf.layers.conv2d(inputs, K, 1, use_bias=False,
                         kernel_initializer=assign_weight_initializer,
                         name='assignment')
    a = tf.nn.softmax(s)
    a = tf.expand_dims(a, -2)
    C = tf.get_variable('cluster_centers', [1, 1, 1, D, K],
                        initializer=cluster_initializer,
                        dtype=inputs.dtype)
    v = tf.expand_dims(inputs, -1) + C
    v = a * v
    v = tf.reduce_sum(v, axis=[1, 2])
    v = tf.transpose(v, perm=[0, 2, 1])
    if not skip_postnorm:
        v = matconvnetNormalize(v, 1e-12)
        v = tf.transpose(v, perm=[0, 2, 1])
        v = matconvnetNormalize(tf.layers.flatten(v), 1e-12)
    return v

def matconvnetNormalize(inputs, epsilon):
    return inputs / tf.sqrt(tf.reduce_sum(inputs ** 2, axis=-1, keep_dims=True) + epsilon)
