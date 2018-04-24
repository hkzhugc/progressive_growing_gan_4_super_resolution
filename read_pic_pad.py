import tensorflow as tf
import PIL.Image
import numpy as np
import os
import cv2

def read_data(path, num, scale = False):
        list_ = []
        list1_ = []
        for root, dirs, files in os.walk(path):
                for file in files:
                        if len(list_) >= num:
                                break
                        if file.endswith('png') or file.endswith('jpg'):
                                print(path + file)
                                pil_read = cv2.imread(path + file)
                                pil_read = cv2.cvtColor(pil_read, cv2.COLOR_BGR2RGB)
                                print(path + file[:-3] + 'bin')
                                np_read = np.fromfile(path + file[: -3] + 'bin', dtype = np.int32)
                                print(np_read)
                                np_read.resize(3,2)
                                if scale:
                                    pil_read = pil_read / 255
                                list_.append(pil_read)
                                list1_.append(np_read)
                                #list_.append(np.fromstring(pil_read.tobytes(), dtype = np.uint8).reshape(pil_read.size[0], pil_read.size[1], 3))
        if scale:
            return np.array(list_), np.array(list1_)
        return np.array(list_, dtype='uint8'), np.array(list1_)

def rename(path):
    i = 0
    for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('png') or file.endswith('jpg'):
                    os.rename(path + file, path + '%d' % i + '.jpg')
                    i = i + 1
