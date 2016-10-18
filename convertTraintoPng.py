#!/usr/bin/env python
import struct
import numpy as np
#import matplotlib.pyplot as plt
import Image
import sys
import os

input_path = sys.argv[1]
output_path = sys.argv[2]

# =====read train labels=====
label_train_file = input_path + '/train-labels-idx1-ubyte'
label_train_fp = open(label_train_file, 'rb')
label_train_buf = label_train_fp.read()

label_train_index=0
label_magic, label_numImages = struct.unpack_from('>II', label_train_buf, label_train_index)
label_train_index += struct.calcsize('>II')
train_labels = struct.unpack_from('>60000B', label_train_buf, label_train_index)

# =====read train images=====
label_train_map = {}
train_file = input_path + '/train-images-idx3-ubyte'
train_fp = open(train_file, 'rb')
buf = train_fp.read()

index=0
magic,numImages,numRows,numColumns=struct.unpack_from('>IIII',buf,index)
index+=struct.calcsize('>IIII')

k = 0
for image in range(0,numImages):
    train_label = train_labels[k]
    if(label_train_map.has_key(train_label)):
        ids = label_train_map[train_label] + 1
        label_train_map[train_label] += 1
    else:
        label_train_map[train_label] = 0
        ids = 0

    k += 1
    im=struct.unpack_from('>784B',buf,index)
    index+=struct.calcsize('>784B')

    im=np.array(im,dtype='uint8')
    im=im.reshape(28,28)
    #fig=plt.figure()
    #plotwindow=fig.add_subplot(111)
    #plt.imshow(im,cmap='gray')
    #plt.show()
    im=Image.fromarray(im)
    # Prepare the folders
    if (os.path.exists(output_path+'/'+str(train_label))==False):
        os.makedirs(output_path+'/'+str(train_label))
    im.save(output_path + '/%s/%s.png'%(train_label, ids),'png')
