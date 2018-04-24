import cv2
import os
import time
import random
import scipy

import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from read_pic_pad import read_data, rename
from split import split_pic
from model import Vgg19_simple_api
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr


import logging

def lrelu(x):
    return tf.maximum(x*0.2,x)

def batch_data(data, padding_nums, max_idx, batch_size): # random batch
        epoch = max_idx // batch_size
        idx_list = np.arange(max_idx)
        random.shuffle(idx_list)
        output = np.array([data[i] for i in idx_list])
        pad_output = np.array([padding_nums[i] for i in idx_list])
        return epoch, np.split(output, epoch), np.split(pad_output, epoch)

def clip_pic(padding_pic, pn):
    rslt = padding_pic.shape[0]
    return padding_pic[pn[0][0]: rslt - pn[0][1], pn[1][0]: rslt - pn[1][1], :]

def clip_pics(padding_pics, padding_nums):
    clip_res = []
    for i in range(padding_pics.shape[0]):
        pn = padding_nums[i]
        padding_pic = padding_pics[i]
        clip_res += [clip_pic(padding_pic, pn)]
    return clip_res

def cal_loss(picA, picB):
    res = np.zeros([2])
    picA = picA.clip(0, 255)
    picB = picB.clip(0, 255)
    res[0] = compare_psnr(picA, picB, data_range = 255)
    res[1] = compare_ssim(picA, picB, data_range = 255, multichannel = True)
    return res

def save_list(path_prefix, list):
    array = np.array(list)
    array[:, 0].tofile(path_prefix + '_psnr.bin')
    array[:, 1].tofile(path_prefix + '_ssim.bin')

def weight_variable(shape):  
        initial = tf.truncated_normal(shape, stddev=0.1)  
        return tf.Variable(initial)

def layer_from_RGB(input, kernel_size):
        b_conv = weight_variable([kernel_size])
        filter_from_RGB = weight_variable([1, 1, 3, kernel_size])
        output = (tf.nn.conv2d(input, filter_from_RGB, strides=[1, 1, 1, 1], padding='SAME'))# + b_conv
        return output

def layer_to_RGB(input, input_channels):
        b_conv = weight_variable([3])
        filter_from_RGB = weight_variable([1, 1, input_channels, 3])
        output = (tf.nn.conv2d(input, filter_from_RGB, strides=[1, 1, 1, 1], padding='SAME'))# + b_conv
        return output

def conv_2D(input, shape, kernel_size):
        b_conv = weight_variable([kernel_size])
        filter_conv2d = weight_variable(shape)
        output = (tf.nn.conv2d(input, filter_conv2d, strides=[1, 1, 1, 1], padding='SAME')) + b_conv
        return output

def up_scale(input, pre_size):
        return tf.image.resize_images(images = input, size = [2 * pre_size, 2 * pre_size], method = 2)#nn

def down_scale(input, pre_size):
        now_size = pre_size // 2
        return tf.image.resize_images(images = input, size = [now_size, now_size], method = 2)#area

def batch_normalization(input):
        return tf.layers.batch_normalization(inputs = input)#no channels first so axis = 0

def pixel_normalization(layer):
    input = layer.outputs
    layer.outputs = input * tf.rsqrt(tf.reduce_mean(tf.square(input), axis = 3, keep_dims = True) + 1e-8)
    return layer

def WScaleLayer(layer):
    W = layer.outputs
    scale = tf.sqrt(tf.reduce_mean(tf.square(W)))
    layer.outputs = W / scale
    return layer

def net_mix(
        layer1,
        layer2
        ):
    resolution = layer1.outputs.shape[2]
    layer1.outputs = tf.transpose(layer1.outputs, [0, 3, 2, 1])
    layer2.outputs = tf.transpose(layer2.outputs, [0, 3, 2, 1])
    LR_weight = tf.Variable(tf.zeros([resolution, resolution]), name = 'LR_weight_%d' % resolution)
    #SR_weight = tf.Variable(tf.zeros([resolution, resolution]), name = 'SR_weight_%d' % resolution)
    LR_mix = LambdaLayer(layer1, lambda x : x * tf.nn.sigmoid(LR_weight), name = 'LR_mix_%d' % resolution)
    SR_mix = LambdaLayer(layer2, lambda x : x * (1 - tf.nn.sigmoid(LR_weight)), name = 'SR_mix_%d' % resolution)
    out = ElementwiseLayer([LR_mix, SR_mix], combine_fn = tf.add, name = 'out_mix_%d' % resolution)
    out.outputs = tf.transpose(out.outputs, [0, 3, 2, 1])
    out.all_params.extend([LR_weight])
    return out


def net_CT_blend(
        LR_image,
        SR_image,
        reuse = False
        ):
    resolution = LR_image.shape[2]
    with tf.variable_scope("my_CT_blend_%d" % resolution, reuse = reuse) as vs:
        LR_input = InputLayer(LR_image, name = 'LR_transpose_%d' % resolution)
        SR_input = InputLayer(SR_image, name = 'SR_transpose_%d' % resolution)
        out = net_mix(LR_input, SR_input)
        print('len of the params is ')
        print(len(out.all_params))
        return out

def my_GAN_G2(
        image,
        lowest_resolution_log2 = 4,
        mix_weight = 0.0,
        is_train = False,
        reuse = False,
        use_formal_pic = True,
        use_WS = False,
        use_PN = False,
        blend_inG = False,
        method = 0
        ):
    #args:
    #image : the batch image
    #lowest_resolution_log2 : the minest pic log2(size)
    w_init = tf.random_normal_initializer(stddev = 0.02)
    b_init = None
    g_init = tf.random_normal_initializer(1., 0.02)

    resolution = int(image.shape[1]) 
    resolution_log2 = int(np.log2(resolution))
    def WS(layer) : return WScaleLayer(layer) if use_WS else layer
    def PN(layer) : return pixel_normalization(layer) if use_PN else layer

    images = []
    #resize image to make progressive resolution
    for i in range(lowest_resolution_log2, resolution_log2 + 1):
        size = [2 ** i, 2 ** i]
        next_image = tf.image.resize_images(image, size=size, method=method)
        images += [next_image]

    #init the net
    print(reuse)
    with tf.variable_scope("my_GAN_G", reuse = reuse):
        tl.layers.set_name_reuse(reuse)
        mix_rates = []
        G_ouputs = []
        feature_list = []
        for i in range(lowest_resolution_log2, resolution_log2 + 1):
            idx = i - lowest_resolution_log2
            cur_resolutin = 2 ** i
            n = InputLayer(images[idx], name = "input%dx%d" % (cur_resolutin, cur_resolutin))
            temp = n
            n = PN(n)
            n = WS(Conv2d(n, 64, (3, 3), (1, 1), act = tf.nn.relu, padding = 'SAME', W_init = w_init, b_init = b_init, name = 'fromRGB%d' % cur_resolutin))
            n = PN(n)
            n = BatchNormLayer(n, is_train = is_train, gamma_init = g_init, name = 'BN1%d' % cur_resolutin)
            n2 = WS(Conv2d(n, 64, (3, 3), (1, 1), act = tf.nn.relu, padding = 'SAME', W_init = w_init, b_init = b_init, name = 'conv%d' % cur_resolutin))
            n2 = PN(n2)
            n2 = BatchNormLayer(n2, is_train = is_train, gamma_init = g_init, name = 'BN2%d' % cur_resolutin)
            n3 = (Conv2d(n2, 3, (3, 3), (1, 1), act = tf.nn.tanh, padding = 'SAME', W_init = w_init, b_init = b_init, name = 'toRGB%d' % cur_resolutin))
            n3 = LambdaLayer(n3, lambda x : x * 255, name = 'pxl_scale_%d' % cur_resolutin)
            out = n3
            #out.outputs = tf.clip_by_value(out.outputs, 0., 1.)
            #mix_rate = tf.constant(mix_weight, shape = [out.outputs.shape[1],out.outputs.shape[2],out.outputs.shape[3]])
            #mix_rate = (tf.Variable(mix_rate, trainable = True)) 
            mix_rate = tf.Variable(mix_weight)
            mix_rate1 = tf.nn.sigmoid(mix_rate)
            pic_rate = tf.Variable(mix_weight)
            pic_rate1 = tf.nn.sigmoid(pic_rate)
            if idx > 0:
               last_feature = WS(G_ouputs[-1])
               last_feature = WS(Conv2d(last_feature, 64, (3, 3), (1, 1), act = tf.nn.relu, padding = 'SAME', W_init = w_init, b_init = b_init, name = 'upfromRGB%d' % cur_resolutin))
               last_feature = WS(Conv2d(last_feature, 64, (3, 3), (1, 1), act = tf.nn.relu, padding = 'SAME', W_init = w_init, b_init = b_init, name = 'upconv%d' % cur_resolutin))
               last_feature = WS(Conv2d(last_feature, 12, (3, 3), (1, 1), act = tf.nn.relu, padding = 'SAME', W_init = w_init, b_init = b_init, name = 'uptofeature%d' % cur_resolutin))
               last_rgb = (SubpixelConv2d(last_feature, act = tf.nn.tanh, name = 'sub_conv%d' % cur_resolutin))
               last_rgb = LambdaLayer(last_rgb, lambda x : x * 255, name = 'feature_scale_%d' % cur_resolutin)
               out = ElementwiseLayer([out, last_rgb], tf.add, name = 'combine_lastrgb%d' % cur_resolutin)
            if blend_inG:
                out.outputs = net_CT_blend(temp.outputs, out.outputs)
            if use_formal_pic:
               out = ElementwiseLayer([out, temp], tf.add, name = 'combine_formal_pic%d' % cur_resolutin)
            G_ouputs += [out]
            mix_rates += [(mix_rate, pic_rate)]
        return G_ouputs, mix_rates

def my_GAN_D1(
        images, 
        is_train = False,
        fmap_max = 64,
        fmap_base = 1024,
        reuse = False,
        use_sigmoid = True,
        use_WS = False
        ):
    w_init = tf.random_normal_initializer(stddev = 0.02)
    b_init = None
    g_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x : tl.act.lrelu(x, 0.02)

    def nf(stage) : return min(fmap_base // (2 ** stage), fmap_max)
    def WS(layer) : return WScaleLayer(layer) if use_WS else layer

    resolution = int(images.shape[1])
    resolution_log2 = int(np.floor(np.log2(resolution)))


    with tf.variable_scope("my_GAN_D_%d" % resolution, reuse = reuse) as vs:
        print(reuse)
        tl.layers.set_name_reuse(reuse)
        net = InputLayer(images, name = "input%d" % resolution)
        for i in range(resolution_log2 - 1, 1, -1):
            net = WS(Conv2d(net, nf(i), (3, 3), (1, 1), act = lrelu, padding = 'SAME', 
                    W_init = w_init, name = '%d_conv1%d' % (resolution, i)))
            net = WS(Conv2d(net, nf(i - 1), (3, 3), (1, 1), act = lrelu, padding = 'SAME', 
                    W_init = w_init, name = '%d_conv2%d' % (resolution, i)))
            net = WS(MeanPool2d(net, filter_size = (2, 2), strides = (2, 2), 
                    padding = 'SAME', name = '%d_pool1%d' % (resolution, i)))
        net_ho = FlattenLayer(net, name = 'flatten%d' % resolution)
        net_ho = DenseLayer(net_ho, n_units = 1, act = tf.identity, 
                W_init = w_init, name = 'dense%d' % resolution)
        logits = net_ho.outputs
        net_ho.outputs = tf.nn.sigmoid(net_ho.outputs) if use_sigmoid else net_ho.outputs
    return net_ho, logits

def network_new2(
        top_dir = "sr_tanh/",
        svg_dir = "dataset/Test/", #test_data
        pxl_dir = "dataset/Train/", #train_data
        output_dir = "pic_smooth/",
        test_output_dir = 'test_output/',
        checkpoint_dir = "save_model",
        checkpoint_dir1 = "save_model",
        model_name = "model4",
        big_loop = 1,
        scale_num = 2,
        epoch_init = 5000,
        strides = 20,
        batch_size = 4,
        max_idx = 92,
        data_size = 92,
        lr_init = 1e-3,
        learning_rate = 1e-5,
        vgg_weight_list = [1, 1, 5e-1, 1e-1],
        use_vgg = False,
        use_L1_loss = False,
        wgan = False,
        init_g = True,
        init_d = True,
        init_b = False,
        method = 0,
        lowest_resolution_log2 = 4,
        train_net = True,
        generate_pics = True,
        resume_network = False): 

    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler(top_dir + "log.txt")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


    for idx, val in enumerate(network_new2.__defaults__):
        logger.info(str(network_new2.__code__.co_varnames[idx]) + ' == ' + str(val))

    output_dir = top_dir + output_dir
    test_output_dir = top_dir + test_output_dir
    checkpoint_dir = top_dir + checkpoint_dir 
    checkpoint_dir1 = top_dir + checkpoint_dir1 
    
    logger.info("start building the net")

    if use_vgg:
        print('use vgg')     
    t_target_image_data, image_padding_nums = read_data(pxl_dir, data_size)   
    resolution = t_target_image_data.shape[1] / scale_num
    target_resolution = t_target_image_data.shape[1]
    resolution_log2 = int(np.floor(np.log2(resolution)))
    target_resolution_log2 = int(np.floor(np.log2(target_resolution)))

    #image = tf.image.resize_images(t_image, size=[64, 64], method=2)
    t_image_target = tf.placeholder('float32', [None, target_resolution, target_resolution, 3], name = 't_image_target')
    t_image_ = tf.image.resize_images(t_image_target, size=[target_resolution // scale_num, target_resolution // scale_num], method=method)
    t_image = tf.image.resize_images(t_image_, size=[target_resolution, target_resolution], method=method)

    t_image_target_list = []
    t_image_list = []

    #generate list of pics from 2 ** 2 resolution to t_image_size resolution
    net_Gs, mix_rates = my_GAN_G2(t_image, is_train = True, reuse = False)
    print("init Gs")
    net_Gs[-1].print_params(False)
    net_g_test, _ = my_GAN_G2(t_image, is_train = False, reuse = True)
    print("init g_test")

    if use_vgg:
        t_target_image_224 = tf.placeholder('float32', [None, 224, 224, 3], name = 't_image_224')
        t_predict_image_224 = tf.placeholder('float32', [None, 224, 224, 3], name = 't_target_224')
        net_vgg, vgg_target_emb = Vgg19_simple_api((t_target_image_224+1)/2, reuse=False)

    #initialize the list to store different level net
    net_ds = []
    b_outputs = []
    logits_reals = []

    logits_fakes = []
    logits_fakes2 = []

    d_loss_list  = []
    b_loss_list  = []
    d_loss3_list  = []
    mse_loss_list = []
    g_gan_loss_list = []
    g_loss_list = []

    g_init_optimizer_list = []
    d_init_optimizer_list = []

    g_optimizer_list = []
    d_optimizer_list = []
    b_optimizer_list = []

    w_clip_list = []

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)
    
    print("init Ds")
    for i in range(lowest_resolution_log2, target_resolution_log2 + 1):
        idx = i - lowest_resolution_log2
        cur_resolution = 2 ** i
        size = [cur_resolution, cur_resolution]

        target_i = tf.image.resize_images(t_image_target, size=size, method=method)
        image_i = tf.image.resize_images(t_image, size=size, method=method)
        t_image_target_list += [target_i]
        t_image_list += [image_i]


        if use_vgg:
            t_target_image_224 = tf.image.resize_images(t_image_target, size=[224, 224], method=1, align_corners=False) # resize_target_image_for_vgg # http://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/layers.html#UpSampling2dLayer
            add_dimens = tf.zeros_like(t_target_image_224)
            print(add_dimens.dtype)
            print(t_target_image_224.dtype)
            t_predict_image_224 = tf.image.resize_images(net_Gs[idx].outputs, size=[224, 224], method=1, align_corners=False) # resize_generate_image_for_vgg
            net_vgg, vgg_target_emb = Vgg19_simple_api((t_target_image_224+1)/2, reuse=True)
            _, vgg_predict_emb = Vgg19_simple_api((t_predict_image_224+1)/2, reuse=True)

        #initialize the D_reals and D_fake
        net_d, logits_real = my_GAN_D1(target_i, is_train = True, reuse = False, use_sigmoid = not wgan)
        _,     logits_fake = my_GAN_D1(net_Gs[idx].outputs, is_train = True, reuse = True, use_sigmoid = not wgan)
        _,     logits_fake2 = my_GAN_D1(image_i, is_train = True, reuse = True, use_sigmoid = not wgan)

        blend_output = net_CT_blend(image_i, net_Gs[idx].outputs)
        b_outputs += [blend_output]

        net_ds += [net_d]
        logits_reals += [logits_real]
        logits_fakes += [logits_fake]
        logits_fakes2 += [logits_fake2]

        mix_factors = np.random.uniform(size = [1, 1, 1, int(target_i.shape[3])])
        print(mix_factors.shape)
        mix_pic = net_Gs[idx].outputs * mix_factors + target_i * (1 - mix_factors)

        _,     logits_mix = my_GAN_D1(mix_pic, is_train = True, reuse = True)

        d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, 
                tf.ones_like(logits_real), name = 'd1_%d' % cur_resolution)

        d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, 
                tf.zeros_like(logits_fake), name = 'd2_%d' % cur_resolution)

        d_loss3 = tl.cost.sigmoid_cross_entropy(logits_fake2, 
                tf.zeros_like(logits_fake2), name = 'd3_%d' % cur_resolution)

        d_loss4 = (tf.reduce_mean(logits_fake2)) - (tf.reduce_mean(logits_real))

        d_loss4 = tf.nn.sigmoid(d_loss4) #make sure in [0, 1]

        d_loss = 1 * (d_loss1 + d_loss2) #+ d_loss3  + d_loss4
        d_loss += 0. 

        d_loss3 += d_loss1

        use_vgg22 = True

        vgg_loss = 0
        if use_vgg:
            for i, vgg_target in enumerate(vgg_target_emb):
                vgg_loss += vgg_weight_list[i] * tl.cost.mean_squared_error(vgg_predict_emb[i].outputs, vgg_target.outputs, is_mean=True)
        g_gan_loss1 = tl.cost.sigmoid_cross_entropy(logits_fake, 
                tf.ones_like(logits_fake), name = 'g_%d' % cur_resolution)
        g_gan_loss2 = (tf.reduce_mean(logits_fake2)) - (tf.reduce_mean(logits_fake))
        g_gan_loss2 = tf.nn.sigmoid(g_gan_loss2) #make sure in [0, 1]

        g_gan_loss = g_gan_loss1# + g_gan_loss2

        mse_loss = tl.cost.mean_squared_error(net_Gs[idx].outputs, target_i, is_mean = True)
        if use_L1_loss:
            mes_loss = tf.reduce_mean(tf.reduce_mean(tf.abs(net_Gs[idx].outputs - target_i)))
        g_gan_loss_list += [g_gan_loss]
        mse_loss_list += [mse_loss]

        g_loss = 1e-3 * g_gan_loss + mse_loss

        L1_norm = tf.reduce_mean(tf.reduce_mean(net_Gs[idx].outputs))

        def TV_loss(x):
            loss1 = x[:, :, 1:, :] - x[:, :, :-1, :] ** 2
            loss2 = x[:, 1:, :, :] - x[:, :-1, :, :] ** 2
            return tf.reduce_sum(tf.reduce_sum(loss1)) + tf.reduce_sum(tf.reduce_sum(loss2))

        tV_loss = TV_loss(net_Gs[idx].outputs)

        b_loss = tl.cost.mean_squared_error(blend_output.outputs, target_i, is_mean = True)

        if i >= 7: g_loss += vgg_loss

        #g_loss += vgg_loss

        g_vars = tl.layers.get_variables_with_name('my_GAN_G', True, True)
        d_vars = tl.layers.get_variables_with_name('my_GAN_D_%d' % cur_resolution, True, True)
        b_vars = tl.layers.get_variables_with_name('my_CT_blend_%d' % cur_resolution, True, True)

        g_optim_init = tf.train.AdamOptimizer(lr_v, 0.9).minimize(mse_loss, var_list = g_vars)
        g_init_optimizer_list += [g_optim_init]

        d_optim_init = tf.train.AdamOptimizer(lr_v, 0.9).minimize(d_loss3, var_list = d_vars)

        g_optim = tf.train.AdamOptimizer(lr_v, 0.9).minimize(g_loss, var_list = g_vars)
        d_optim = tf.train.AdamOptimizer(lr_v, 0.9).minimize(d_loss, var_list = d_vars)

        b_optim = tf.train.AdamOptimizer(lr_v, 0.9).minimize(b_loss, var_list = b_vars)

        #WGAN
        if wgan:
            print('mode is wgan')
            g_loss = -(tf.reduce_mean(logits_fake)) + vgg_loss
            d_loss = (tf.reduce_mean(logits_fake)) - (tf.reduce_mean(logits_real))
            d_loss3 = (tf.reduce_mean(logits_fake2)) - (tf.reduce_mean(logits_real))

            mix_grads = tf.gradients(tf.reduce_sum(logits_mix), mix_pic)
            mix_norms = tf.sqrt(tf.reduce_sum(tf.square(mix_grads), axis = [1, 2, 3]))
            
            addtion = tf.reduce_mean(tf.square(mix_norms - 1.)) * 5.0
            #d_loss = d_loss + d_loss3 + addtion + tl.cost.mean_squared_error(logits_real, tf.zeros_like(logits_real)) * 1e-3
            d_loss = d_loss + d_loss3

            g_optim = tf.train.RMSPropOptimizer(learning_rate).minimize(g_loss, var_list = g_vars)
            d_optim = tf.train.RMSPropOptimizer(learning_rate).minimize(d_loss, var_list = d_vars)

            d_optim_init = tf.train.RMSPropOptimizer(learning_rate).minimize(d_loss3, var_list = d_vars)
            clip_ops = []
            for var in d_vars:
                clip_bound = [-1.0, 1.0]
                clip_ops.append(
                    tf.assign(
                        var, 
                        tf.clip_by_value(var, clip_bound[0], clip_bound[1])
                    )
                )
            clip_disc_weights = tf.group(*clip_ops)
            w_clip_list += [clip_disc_weights]


        d_loss_list += [d_loss]
        d_loss3_list += [d_loss3]
        g_loss_list += [g_loss]
        b_loss_list += [b_loss]
        g_optimizer_list += [g_optim]
        d_optimizer_list += [d_optim]
        b_optimizer_list += [b_optim]
        d_init_optimizer_list += [d_optim_init]

        print("init Res : %d D" % cur_resolution)

    #Restore Model
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    sess = tf.Session(config = config)
    tl.layers.initialize_global_variables(sess)

    #......code for restore model
    if use_vgg:
        vgg19_npy_path = "vgg19.npy"
        if not os.path.isfile(vgg19_npy_path):
            print("Please download vgg19.npz from : https://github.com/machrisaa/tensorflow-vgg")
            exit()
        npz = np.load(vgg19_npy_path, encoding='latin1').item()

        params = []
        for val in sorted( npz.items() ):
            W = np.asarray(val[1][0])
            b = np.asarray(val[1][1])
            print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
            params.extend([W, b])
            if(len(params) == len(net_vgg.all_params)): break
        tl.files.assign_params(sess, params, net_vgg)
        # net_vgg.print_params(False)
        # net_vgg.print_layers()

    #Read Data
    #t_image_data, t_target_image_data = split_pic(read_data(svg_dir, data_size))        

    #initialize G


    for temp_i in range(big_loop):

        decay_every = epoch_init // 2
        lr_decay = 0.1

        logger.info("start training the net")

        for R in range(lowest_resolution_log2, target_resolution_log2 + 1):
            idx = R - lowest_resolution_log2
            if resume_network or not train_net:
                tl.files.load_and_assign_npz_dict(sess = sess, name = checkpoint_dir1 + '/g_%d_level_my_gan.npz' % R, network = net_Gs[idx])
                tl.files.load_and_assign_npz_dict(sess = sess, name = checkpoint_dir1 + '/b_%d_level_my_gan.npz' % R, network = b_outputs[idx])
                #tl.files.load_and_assign_npz_dict(sess = sess, name = checkpoint_dir1 + '/d_%d_level_my_gan.npz' % R, network = net_ds[idx])

            total_mse_loss = 0
            mse_loss = mse_loss_list[idx]
            g_optim_init = g_init_optimizer_list[idx]

            total_d3_loss = 0
            d_loss3 = d_loss3_list[idx]
            d_optim = d_optimizer_list[idx]
            d_loss = d_loss_list[idx]
            
            d_optim_init = d_init_optimizer_list[idx]

            ni = int(np.sqrt(batch_size))
            out_svg = sess.run(t_image_list[idx], {t_image_target : t_target_image_data[0:batch_size]})
            out_pxl = sess.run(t_image_target_list[idx], {t_image_target : t_target_image_data[0:batch_size]})
            print(out_pxl[0])
            print(out_pxl.dtype)
            tl.vis.save_images(out_svg, [ni, ni], output_dir + "R_%d_svg.png" % (R))
            tl.vis.save_images(out_pxl, [ni, ni], output_dir + "R_%d_pxl.png" % (R))
            
            f = open('log%d.txt' % R, 'w')
            pre_loss_list = []
            now_loss_list = []
            if init_g and train_net:
                #fix lr_v
                print('init g')
                sess.run(tf.assign(lr_v, lr_init))
                
                for epoch in range(epoch_init + 1):
                    iters, data, padding_nums = batch_data(t_target_image_data, image_padding_nums, max_idx, batch_size)
                    total_mse_loss = 0
                    total_pre_loss = np.zeros([2])
                    total_now_loss = np.zeros([2])
                    for i in range(iters):
                        errM, _ = sess.run([mse_loss, g_optim_init], {t_image_target : data[i]})
                        total_mse_loss += errM
                        if R == target_resolution_log2:#final steps
                            lowR_pics, output_pics, GT_pics = sess.run([t_image_list[idx], net_g_test[idx].outputs, t_image_target_list[idx]], 
                                {t_image_target : data[i]})
                            pre_lowR_pics = clip_pics(lowR_pics, padding_nums[i])
                            pre_output_pics = clip_pics(output_pics, padding_nums[i])
                            pre_GT_pics = clip_pics(GT_pics, padding_nums[i])
                            for ii in range(data[i].shape[0]):
                                pre_loss = cal_loss(pre_lowR_pics[ii], pre_GT_pics[ii])
                                now_loss = cal_loss(pre_output_pics[ii], pre_GT_pics[ii])
                                total_pre_loss += pre_loss
                                total_now_loss += now_loss
                    pre_loss_list += [total_pre_loss / max_idx]
                    now_loss_list += [total_now_loss / max_idx]
                    print("[%d/%d] total_mse_loss = %f errM = %f" % (epoch, epoch_init, total_mse_loss, errM))
                    ## save model
                    if (epoch % strides == 0):
                        print("save img %d" % R)
                        out, logits_real, logits_fake, logits_fake2 = sess.run([net_g_test[idx].outputs, 
                            tf.nn.sigmoid(logits_reals[idx]), tf.nn.sigmoid(logits_fakes[idx]), tf.nn.sigmoid(logits_fakes2[idx])]
                            ,{t_image_target : t_target_image_data[0:batch_size]})
                        print(out[0])
                        print(out.dtype)
                        tl.vis.save_images(out, [ni, ni], output_dir + "R_%d_init_%d.png" % (R, epoch))
                        if epoch % 10 == 0:
                            tl.files.save_npz_dict(net_Gs[idx].all_params, name=checkpoint_dir+('/g_%d_level_{}_init.npz' % R).format(tl.global_flag['mode']), sess=sess)
                print("R %d total_mse_loss = %f" % (2 ** R, total_mse_loss))

                save_list(top_dir + 'init_g_pre', pre_loss_list)
                save_list(top_dir + 'init_g_now', now_loss_list)
                pre_loss_list = []
                now_loss_list = []
                
            if init_d and train_net:
                #fix lr_v
                print('init d')
                sess.run(tf.assign(lr_v, lr_init))
                
                for epoch in range(epoch_init + 1):
                    iters, data, padding_nums = batch_data(t_target_image_data, image_padding_nums, max_idx, batch_size)
                    for i in range(iters):
                        errD3, errD, _ = sess.run([d_loss3, d_loss, d_optim_init], {t_image_target : data[i]})
                        total_d3_loss += errD3
                    print("[%d/%d] d_loss = %f, errD3 = %f" % (epoch, epoch_init, errD, errD3))
                    ## save model
                    if (epoch != 0) and (epoch % 5 == 0):
                        tl.files.save_npz_dict(net_ds[idx].all_params, name=checkpoint_dir+'/d_{}_init.npz'.format(tl.global_flag['mode']), sess=sess)
                    if epoch % 10 == 0:
                        out, logits_real, logits_fake, logits_fake2 = sess.run([net_g_test[idx].outputs, 
                            tf.nn.sigmoid(logits_reals[idx]), tf.nn.sigmoid(logits_fakes[idx]), tf.nn.sigmoid(logits_fakes2[idx])]
                            ,{t_image_target : t_target_image_data[0:batch_size]})
                        print("logits_real", file = f)
                        print(logits_real, file = f) 
                        print("logits_fake", file = f)
                        print(logits_fake, file = f)
                        print("logits_fake2", file = f)
                        print(logits_fake2, file = f)

                print("R %d total_d3_loss = %f" % (2 ** R, total_d3_loss))
                print("init g or d end", file = f)
            #train GAN
            g_optim = g_optimizer_list[idx]
            d_optim = d_optimizer_list[idx]
            d_loss = d_loss_list[idx]
            g_loss = g_loss_list[idx]
            mse_loss = mse_loss_list[idx]
            g_gan_loss = g_gan_loss_list[idx]
            mix_rate, pic_rate = mix_rates[idx]


            increas = 2. / epoch_init
            mix_rate_vals = np.arange(0., 1. + increas, increas)
            
            last_errD = 0.
            last_errG = 0.
            if train_net :
                for epoch in range(epoch_init + 1):
                    if epoch != 0 and (epoch % decay_every == 0):
                        new_lr_decay = lr_decay ** (epoch // decay_every)
                        sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
                    elif epoch == 0:
                        sess.run(tf.assign(lr_v, lr_init))
                        #mix_mat = np.zeros([t_image_list[idx].shape[i] for i in range(1, 4)], dtype = 'float32')
                        sess.run(tf.assign(mix_rate, 0))
                        sess.run(tf.assign(pic_rate, 0))


                    total_d_loss = 0
                    total_g_loss = 0
                    total_mse_loss = 0
                    iters, data, padding_nums = batch_data(t_target_image_data, image_padding_nums, max_idx, batch_size)
                    total_pre_loss = np.zeros([2])
                    total_now_loss = np.zeros([2])
                    for i in range(iters):
                        #update G
                        if wgan : 
                            errG, errM, errA, _ = sess.run([g_loss, mse_loss, g_gan_loss, g_optim], 
                                    {t_image_target : data[i]})
                        #update D
                        if True: #last_errG * 1e3 <= last_errD * 10: # D learning too fast
                            flag = 1
                            errD, _ = sess.run([d_loss, d_optim], 
                                    {t_image_target : data[i]})
                        #print("[%d/%d] epoch %d times d_loss : %f" % (epoch, epoch_init, i, errD))
                        #update G
                        if not wgan : 
                            #print("train G")
                            errG, errM, errA, _ = sess.run([g_loss, mse_loss, g_gan_loss, g_optim], 
                                    {t_image_target : data[i]})
                        #print("[%d/%d] epoch %d times, g_loss : %f, mse_loss : %f, g_gan_loss : %f" 
                        #        % (epoch, epoch_init, i, errG, errM, errA))
                        #clip var_val
                        if wgan :
                            _ = sess.run(w_clip_list[idx])
                        last_errD = errD
                        last_errG = errA
                        total_d_loss += errD
                        total_g_loss += errG
                        total_mse_loss += errM
                        if R == target_resolution_log2:#final steps
                            lowR_pics, output_pics, GT_pics = sess.run([t_image_list[idx], net_g_test[idx].outputs, t_image_target_list[idx]], 
                                {t_image_target : data[i]})
                            pre_lowR_pics = clip_pics(lowR_pics, padding_nums[i])
                            pre_output_pics = clip_pics(output_pics, padding_nums[i])
                            pre_GT_pics = clip_pics(GT_pics, padding_nums[i])
                            for ii in range(data[i].shape[0]):
                                pre_loss = cal_loss(pre_lowR_pics[ii], pre_GT_pics[ii])
                                now_loss = cal_loss(pre_output_pics[ii], pre_GT_pics[ii])
                                total_pre_loss += pre_loss
                                total_now_loss += now_loss
                    pre_loss_list += [total_pre_loss / max_idx]
                    now_loss_list += [total_now_loss / max_idx]
                        
                    print("lastD = %f, lastG = %f" % (last_errD, last_errG))
                    print("[%d/%d] epoch %d times d_loss : %f" % (epoch, epoch_init, i, errD))
                    print("[%d/%d] epoch %d times, errM = %f, mse_loss : %f, g_gan_loss : %f" 
                            % (epoch, epoch_init, i, errM, total_mse_loss, errA))

                    #save genate pic
                    if (epoch % strides == 0):
                        print("save img %d" % R)
                        out, logits_real, logits_fake, logits_fake2 = sess.run([net_g_test[idx].outputs, 
                            tf.nn.sigmoid(logits_reals[idx]), tf.nn.sigmoid(logits_fakes[idx]), tf.nn.sigmoid(logits_fakes2[idx])]
                            ,{t_image_target : t_target_image_data[0:batch_size]})
                        print(out[0])
                        out = out.clip(0, 255)
                        print(out.dtype)
                        tl.vis.save_images(out, [ni, ni], output_dir + "R_%d_train_%d.png" % (R, epoch))
                        #increase the mix_rate from 0 to 1 linearly
                        mix_rate_val = tf.nn.sigmoid(mix_rate).eval(session = sess)
                        mix_pic_val = tf.nn.sigmoid(pic_rate).eval(session = sess)
                        print("logits_real")
                        print(logits_real) 
                        print("logits_fake")
                        print(logits_fake)
                        print("logits_fake2")
                        print(logits_fake2)
                        print("logits_real", file = f)
                        print(logits_real, file = f) 
                        print("logits_fake", file = f)
                        print(logits_fake, file = f)
                        print("logits_fake2", file = f)
                        print(logits_fake2, file = f)
                        if (logits_real ==  logits_fake).all() :
                            print("optimize well")
                            print("optimize well", file = f)
                        print("mix_rate, pic_rate")
                        print(mix_rate_val, mix_pic_val)
                        print("mix_rate, pic_rate", file = f)
                        print(mix_rate_val, mix_pic_val, file = f)
                    ## save model
                    if (epoch != 0) and (epoch % 10 == 0):
                        tl.files.save_npz_dict(net_Gs[idx].all_params, name=checkpoint_dir+('/g_%d_level_{}.npz' % R).format(tl.global_flag['mode']), sess=sess)
                        tl.files.save_npz_dict(net_d.all_params, name=checkpoint_dir+('/d_%d_level_{}.npz' % R).format(tl.global_flag['mode']), sess=sess)
                save_list(top_dir + 'g_pre', pre_loss_list)
                save_list(top_dir + 'g_now', now_loss_list)
                pre_loss_list = []
                now_loss_list = []
                f.close()
                
                blend_output = b_outputs[idx]
                b_loss = b_loss_list[idx]
                b_optim = b_optimizer_list[idx]
                if not True:
                    #fix lr_v
                    sess.run(tf.assign(lr_v, lr_init))
                    
                    for epoch in range(epoch_init * 3 + 1):
                        iters, data, padding_nums = batch_data(t_target_image_data, image_padding_nums, max_idx, batch_size)
                        for i in range(iters):
                            errM, _ = sess.run([b_loss, b_optim], {t_image_target : data[i]})
                            total_mse_loss += errM
                        print("[%d/%d] total_mse_loss = %f errM = %f" % (epoch, epoch_init, total_mse_loss, errM))
                        ## save model
                        if (epoch % (strides * 3) == 0):
                            print("save img %d" % R)
                            out = sess.run(blend_output.outputs
                                ,{t_image_target : t_target_image_data[0:batch_size]})
                            out = out.clip(0, 255)
                            #print(out[0])
                            print(out.dtype)
                            tl.vis.save_images(out, [ni, ni], output_dir + "b_%d_output_%d.png" % (R, epoch))
                            if epoch % 100 == 0:
                                tl.files.save_npz_dict(blend_output.all_params, name=checkpoint_dir+('/b_%d_level_{}.npz' % R).format(tl.global_flag['mode']), sess=sess)

        logger.info("end training the net")

        if not train_net or generate_pics:
            if init_b :
                sess.run(tf.assign(lr_v, lr_init))
                for epoch in range(epoch_init * 3 + 1):
                    iters, data, padding_nums = batch_data(t_target_image_data, image_padding_nums, max_idx, batch_size)
                    for i in range(iters):
                        errM, _ = sess.run([b_loss, b_optim], {t_image_target : data[i]})
                        total_mse_loss += errM
                    print("[%d/%d] total_mse_loss = %f errM = %f" % (epoch, epoch_init, total_mse_loss, errM))
                    ## save model
                    if (epoch % (strides * 3) == 0):
                        print("save img %d" % R)
                        out = sess.run(blend_output.outputs
                            ,{t_image_target : t_target_image_data[0:batch_size]})
                        out = out.clip(0, 255)
                        #print(out[0])
                        print(out.dtype)
                        tl.vis.save_images(out, [ni, ni], output_dir + "b_%d_output_%d.png" % (R, epoch))
                        if epoch % 100 == 0:
                                tl.files.save_npz_dict(blend_output.all_params, name=checkpoint_dir+('/b_%d_level_{}.npz' % R).format(tl.global_flag['mode']), sess=sess)

            logger.info("load params")

            tl.files.load_and_assign_npz_dict(sess = sess, name = checkpoint_dir1 + '/g_%d_level_my_gan.npz' % R, network = net_Gs[-1])
            tl.files.load_and_assign_npz_dict(sess = sess, name = checkpoint_dir1 + '/b_%d_level_my_gan.npz' % R, network = b_outputs[-1])
                    
            logger.info("read pics")
            test_set_dir = ["Set5/", "Set14/"]
            test_no = [5, 13]
            for j in range(2):
                data_pxl, pic_pad_nums = read_data(svg_dir + test_set_dir[j], num = test_no[j])
                iters = data_pxl.shape[0]
                data_pxl = np.split(data_pxl, iters)
                #iters, data = batch_data((t_image_data, t_target_image_data), 100, batch_size)
                logger.info('start evaluating pics')
                for i in range(iters):
                    print("save img %d" % R)
                    out = sess.run(net_g_test[idx].outputs, {t_image_target : data_pxl[i]})
                    out = out.clip(0, 255)
                    out = np.array([clip_pic(out[0], pic_pad_nums[i])])
                    tl.vis.save_images(out, [1, 1], test_output_dir + test_set_dir[j] + "g_%d_output_%d.png" % (R, i))
                    out = sess.run(b_outputs[idx].outputs, {t_image_target : data_pxl[i]})
                    out = out.clip(0, 255)
                    out = np.array([clip_pic(out[0], pic_pad_nums[i])])
                    tl.vis.save_images(out, [1, 1], test_output_dir +  test_set_dir[j] +"b_%d_output_%d.png" % (R, i))
                    out = sess.run(t_image, {t_image_target : data_pxl[i]})
                    out = np.array([clip_pic(out[0], pic_pad_nums[i])])
                    tl.vis.save_images(out, [1, 1], test_output_dir +  test_set_dir[j] +"svg_%d_%d.png" % (R, i))
                    out = sess.run(t_image_target, {t_image_target : data_pxl[i]})
                    out = np.array([clip_pic(out[0], pic_pad_nums[i])])
                    tl.vis.save_images(out, [1, 1], test_output_dir +  test_set_dir[j] +"pxl_%d_%d.png" % (R, i))
            logger.info('end evaluating pics')


def evaluate(
    input_dir = "test_input/",
    output_dir = "test_output/",
    checkpoint_dir = "save_model2",
    batch_size = 4
    ): 
    t_image_data = read_data(input_dir)        

    shape = t_image_data.shape 

    print(shape)

    t_image = tf.placeholder('float32', [None, shape[1], shape[2], shape[3]], name = 'input_image')

    outputs, _ = my_GAN_G2(t_image, is_train = False, reuse = False)
    print(len(outputs))
                       
    sess = tf.Session(config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    for i in range(len(outputs)):
        R = i + 4
        tl.files.load_and_assign_npz_dict(sess = sess, name = checkpoint_dir + '/g_%d_level_my_gan.npz' % R, network = outputs[i])

    epoch = shape[0] // batch_size
    ni = int(np.sqrt(batch_size))

    for i in range(epoch):
        data =  t_image_data[batch_size * i:batch_size * (i + 1)]
        output = sess.run(outputs[-1].outputs, {t_image : data})
        tl.vis.save_images(output, [ni, ni], output_dir + '/%d_output.png' % i)
        #output_bicu = scipy.misc.imresize(data[0], [shape[1] * 4, shape[1] * 4], interp = 'bicubic', mode = None)
        #tl.vis.save_image(output_bicu, output_dir + '/%d_output_bicu.png' % i)



if __name__ == "__main__":
        #network()


        tl.global_flag['mode'] = 'my_gan'
        #network_new()
        #evaluate()
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '2'
        tl.global_flag['mode'] = 'my_gan'
        for i in range(1):
            network_new2(train_net = True, resume_network = False, use_L1_loss = True, big_loop = 1, use_vgg = True)
