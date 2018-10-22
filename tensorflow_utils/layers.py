from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
class discriminator:
    def __init__(self,inputs,reuse=False,name='CNN'):
        print('Initialize {}'.format(name))
        
        self.rate=tf.placeholder(tf.float64)
        self.inputs=inputs

        self.build(self.inputs,reuse=reuse,name=name)
        print(self.__dict__)

    def build(self,inputs,reuse,name):
        with tf.variable_scope(name,reuse=reuse):
            self.dense1=self.dense_conv2d(
                    self.inputs,
                    k=12,l=12,
                    name='dense_conv_1')

            self.trans1=self.transition(
                    self.dense1,
                    name='transition_1')

            self.dense2=self.dense_conv2d(
                    self.trans1,
                    k=24,l=12,
                    name='dense_conv_2')

            self.trans2=self.transition(
                    self.dense2,
                    name='transition_2')

            self.dense3=self.dense_conv2d(
                    self.trans2,
                    k=48,l=12,
                    name='dense_conv_3')

            self.output=self.dense3

    def dense_conv2d(self,inputs,kernel_size=[3,3],k=12,l=6,name='dense_conv2d'):
        kwargs={'filters': k,
                'kernel_size': kernel_size,
                'padding': 'same',
                'activation': tf.nn.leaky_relu} 

        with tf.name_scope(name):
            c=inputs
            for layer in range(l):
                x=tf.layers.conv2d(inputs=c,**kwargs)
                c=tf.concat([c,x],3)
            setattr(self,name,c)

            return c
    def transition(self,inputs,name='pooling'):
        t=tf.layers.average_pooling2d(
                inputs=inputs,
                pool_size=[2,2],
                strides=2,
                name=name)
        setattr(self,name,t)
        return t

