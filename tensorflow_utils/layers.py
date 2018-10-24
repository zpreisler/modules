from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow_utils import coord2d
class discriminator:
    def __init__(self,inputs,reuse=False,name='CNN',k=4,l=4):
        print('Initialize {}'.format(name))
        
        self.rate=tf.placeholder(tf.float64)
        self.inputs=inputs
        self.coord=coord2d((inputs/255.0*-2.0)+1.0,name='coord2d')

        self.build(self.coord,reuse=reuse,name=name,k=k,l=l)

    def define_output(self,labels):
        self.define_loss(labels)
        self.define_optimizer()
        self.define_training()

        self.prediction=tf.argmax(self.softmax,1,output_type=tf.int32)
        self.true=labels
        equality=tf.equal(self.prediction,self.true)
        self.accuracy=tf.reduce_mean(tf.cast(equality,tf.float64))

    def define_loss(self,labels):
        self.logits=tf.layers.dense(
                inputs=self.dense_output,
                units=10)
        self.labels=tf.one_hot(labels,10)
        self.softmax=tf.nn.softmax(self.logits)
        self.loss=tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=self.labels,
                    logits=self.logits
                    )
                )

    def define_optimizer(self):
        self.optimizer=tf.train.AdamOptimizer(learning_rate=self.rate)

    def define_training(self):
        self.train=self.optimizer.minimize(self.loss)

    def build(self,inputs,reuse,name,k=4,l=4):
        with tf.variable_scope(name,reuse=reuse):
            self.dense_conv1=self.dense_conv2d(
                    self.inputs,
                    k=k,l=l,
                    name='dense_conv1')

            self.transition1=self.transition(
                    self.dense_conv1,
                    name='transition_1')
            print("transition1",self.transition1.shape)

            self.dense_conv2=self.dense_conv2d(
                    self.transition1,
                    k=k,l=l,
                    name='dense_conv2')

            self.transition2=self.transition(
                    self.dense_conv2,
                    name='transition2')
            print("transition2",self.transition2.shape)

            self.dense_conv3=self.dense_conv2d(
                    self.transition2,
                    k=k,l=l,
                    name='dense_conv3')

            self.transition3=self.transition(
                    self.dense_conv3,
                    name='transition3')
            print("transition3",self.transition3.shape)

            self.dense_conv4=self.dense_conv2d(
                    self.transition3,
                    k=k,l=l,
                    name='dense_conv4')

            self.transition4=self.transition(
                    self.dense_conv4,
                    name='transition4')
            print("transition4",self.transition4.shape)

            self.dense_conv5=self.dense_conv2d(
                    self.transition4,
                    k=k,l=l,
                    name='dense_conv4')

            s=self.dense_conv5.shape
            print('flatten:',s)
            print('flatten_size:',s[1]*s[2]*s[3])
            self.flatten=tf.reshape(self.dense_conv5,[-1,s[1]*s[2]*s[3]])

            self.dense_output=tf.layers.dense(
                    inputs=self.flatten,
                    units=1000,
                    activation=tf.nn.leaky_relu,
                    name='dense_output')

            self.output=self.dense_output

    def dense_conv2d(self,inputs,kernel_size=[3,3],k=12,l=6,name='_dense_conv2d_'):
        kwargs={'filters': k,
                'kernel_size': kernel_size,
                'padding': 'same',
                'activation': tf.nn.leaky_relu} 

        with tf.name_scope(name):
            c=inputs
            for layer in range(l):
                x=tf.layers.conv2d(inputs=c,**kwargs)
                c=tf.concat([c,x],3)
            return c

    def transition(self,inputs,name='_transition_'):
        t=tf.layers.average_pooling2d(
                inputs=inputs,
                pool_size=[2,2],
                strides=2,
                name=name)
        setattr(self,name,t)
        return t
