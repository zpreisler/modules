from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
def get_labels_from_filenames(files,d={'h':1,'s':2,'x':3}):
    from numpy import array
    labels=[]
    for f in files:
        n=f[f.rfind('/')+1]
        label=d[n]
        labels+=[label]
    return array(labels,dtype='int32')

def parse(dataset):
    f=dataset['images']
    image_string=tf.read_file(f)
    image_decoded=tf.image.decode_png(image_string,1)
    image_cropped=tf.image.central_crop(image_decoded,0.5)
    image_resized=tf.image.resize_images(image_cropped,[128,128])

    images=image_resized
    #images=tf.expand_dims(images,0)
    #images=tf.manip.tile(images,[10,1,1,1])
    dataset['images']=images

    return dataset

def image_pipeline(train_data,eval_data=None,batch_size=1):
    traininig_iterator=None
    evaluation_iterator=None
    handle=tf.placeholder(tf.string,shape=[])

    train_image_dataset=tf.data.Dataset.from_tensor_slices(train_data)
    train_dataset=train_image_dataset.map(parse)
    train_dataset=train_dataset.repeat(100).shuffle(len(train_data['images'])).batch(batch_size)

    iterator=tf.data.Iterator.from_string_handle(
            handle,
            train_dataset.output_types,
            train_dataset.output_shapes)
    next_element=iterator.get_next()
    training_iterator=train_dataset.make_initializable_iterator()

    if eval_data is not None:
        eval_image_dataset=tf.data.Dataset.from_tensor_slices(eval_data)
        eval_dataset=eval_image_dataset.map(parse)
        eval_dataset=eval_dataset.repeat().batch(len(eval_data['images']))
        evaluation_iterator=eval_dataset.make_initializable_iterator()

    return handle,next_element,training_iterator,evaluation_iterator

    #iterator=tf.data.Iterator.from_structure(
    #        dataset.output_types,
    #        dataset.output_shapes)
    #next_element=iterator.get_next()
    #init_train_op=iterator.make_initializer(dataset)

    #return next_element,init_train_op
