from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
def parse(dataset):
    image_string=tf.read_file(dataset['images'])
    image_decoded=tf.image.decode_png(image_string,1)
    image_cropped=tf.image.central_crop(image_decoded,0.5)
    image_resized=tf.image.resize_images(image_cropped,[128,128])

    images=image_resized
    #images=tf.expand_dims(images,0)
    #images=tf.manip.tile(images,[10,1,1,1])
    dataset['images']=images

    return dataset

def image_pipeline(files,batch_size=1):
    image_dataset=tf.data.Dataset.from_tensor_slices(files)

    dataset=image_dataset.map(parse)
    dataset=dataset.batch(batch_size)

    iterator=tf.data.Iterator.from_structure(
            dataset.output_types,
            dataset.output_shapes)
    next_element=iterator.get_next()
    init_train_op=iterator.make_initializer(dataset)

    return next_element,init_train_op
