import tensorflow as tf
import PIL.Image
import numpy as np
import os
from read import read_data
import scipy.misc

def split_pic(data):
	print(data.shape)
	res = np.split(data, 2, axis = 2)
	print(res[0].shape)
	return res[0], res[1]


if __name__ == "__main__":
    data = read_data('DATA/', 200)
    data = data * 255
    data = data.astype(dtype = 'uint8')

    svg, pxl = split_pic(data)
    out = tf.placeholder('float32', [256, 256, 3])
    out1 = tf.image.resize_images(out, [64, 64], 0)
    out2 = tf.image.resize_images(out1, [256, 256], 0)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(200):
            #out_pxl = sess.run(out2, {out : pxl[i]})
            #out = scipy.misc.imresize(pxl[i], (64, 64), 'bilinear')
            #out = scipy.misc.imresize(out, (256, 256), 'bilinear')
            #scipy.misc.imsave('test_input/%d.png' %i, out_pxl)
            #PIL.Image.fromarray(pxl[i]).resize((64,64)).resize((256, 256)).save('test_input2/%d.png' % i)
            PIL.Image.fromarray(svg[i]).save('LR/%d.png' % i)
            PIL.Image.fromarray(pxl[i]).save('HR/%d.png' % i)

