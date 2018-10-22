from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
def coord2d(inputs,name=None):
    def zrange(dim):
        x=tf.range(dim)/(dim-1)
        return (x*2)-1.0
    
    def xlayer(shape):
        x=zrange(shape[1])
        x=tf.expand_dims(x,0)
        x=tf.expand_dims(x,0)
        x=tf.manip.tile(x,[shape[0],shape[2],1])
        x=tf.expand_dims(x,-1)
        return x

    def ylayer(shape):
        y=zrange(shape[2])
        y=tf.reshape(y,(shape[2],1))
        y=tf.expand_dims(y,0)
        y=tf.manip.tile(y,[shape[0],1,shape[1]])
        y=tf.expand_dims(y,-1)
        return y

    if not name:
        name='coord2d'
    with tf.name_scope(name):
        shape=tf.shape(inputs)

        x=tf.cast(xlayer(shape),
                dtype=inputs.dtype)
        y=tf.cast(ylayer(shape),
                dtype=inputs.dtype)

        return tf.concat([inputs,x,y],-1)
