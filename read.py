import tensorflow as tf
import PIL.Image
import numpy as np
import os
import cv2
import scipy.misc

def read_data(path, num = None, color = 'a'):
        list_ = []
        for root, dirs, files in os.walk(path):
                for file in files:
                        if num != None and len(list_) >= num:
                                break
                        if file.endswith('png') or file.endswith('jpg'):
                                print(file)
                                pil_read = cv2.imread(path + file)
                                if color == 'y':
                                    ycrcb = cv2.cvtColor(pil_read, cv2.COLOR_BGR2YCrCb)
                                    pil_read,_,_ = cv2.split(ycrcb)
                                elif color == 'r':
                                    ycrcb = cv2.cvtColor(pil_read, cv2.COLOR_BGR2YCrCb)
                                    _, pil_read, _ = cv2.split(ycrcb)
                                elif color == 'b':
                                    ycrcb = cv2.cvtColor(pil_read, cv2.COLOR_BGR2YCrCb)
                                    _, _, pil_read = cv2.split(ycrcb)
                                else:
                                    pil_read = cv2.cvtColor(pil_read, cv2.COLOR_BGR2RGB)
                                list_.append(pil_read / 255)
                                #list_.append(np.fromstring(pil_read.tobytes(), dtype = np.uint8).reshape(pil_read.size[0], pil_read.size[1], 3))
        return np.array(list_)

def rename(path):
    i = 0
    for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('png') or file.endswith('jpg'):
                    os.rename(path + file, path + '%d' % i + '.jpg')
                    i = i + 1
