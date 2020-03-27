import h5py
import numpy as np
import os

files = ['train.h5', 'train2.h5', 'train3.h5', 'train4.h5', 'train5.h5', 'train6.h5']

if __name__ == '__main__':
    new_data = h5py.File('new.h5', 'w')
    # for key in new_data.keys():
    # print(new_data['data'])
    # print(new_data['label_x4'])
    for i, file in enumerate(files):
        piece = h5py.File(file, 'r')
        if i == 0:
            dat = piece['data'].value
            label_x4 = piece['label_x4'].value

        else:
            dat = np.concatenate((dat, piece['data'].value), axis=0)
            label_x4 = np.concatenate((label_x4, piece['label_x4'].value), axis=0)
    # print(dat.shape, label_x4.shape)

    data_input = new_data.create_dataset("data", data=dat)
    data_label_x4 = new_data.create_dataset("label_x4", data=label_x4)

    # test
    new_data = h5py.File('new.h5', 'r')
    for key in new_data.keys():
        print(new_data[key].value.shape)
