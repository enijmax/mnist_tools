#!/usr/bin/env python
import struct
import numpy as np
#import matplotlib.pyplot as plt
import Image
import sys
import os

input_path = sys.argv[1]
output_path = sys.argv[2]

# =====read test labels=====
label_test_file = input_path + '/t10k-labels-idx1-ubyte'
label_test_fp = open(label_test_file, 'rb')
label_test_buf = label_test_fp.read()

label_test_index=0
label_magic, label_numImages = struct.unpack_from('>II', label_test_buf, label_test_index)
label_test_index += struct.calcsize('>II')
test_labels = struct.unpack_from('>10000B', label_test_buf, label_test_index)

# =====read test images=====
label_test_map = {}
test_file = input_path + '/t10k-images-idx3-ubyte'
test_fp = open(test_file, 'rb')
buf = test_fp.read()

index=0
magic,numImages,numRows,numColumns=struct.unpack_from('>IIII',buf,index)
index+=struct.calcsize('>IIII')

k = 0
for image in range(0,numImages):
    test_label = test_labels[k]
    if(label_test_map.has_key(test_label)):
        ids = label_test_map[test_label] + 1
        label_test_map[test_label] += 1
    else:
        label_test_map[test_label] = 0
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
    if (os.path.exists(output_path+'/'+str(test_label))==False):
        os.makedirs(output_path+'/'+str(test_label))
    im.save(output_path + '/%s/%s.png'%(test_label, ids),'png')
