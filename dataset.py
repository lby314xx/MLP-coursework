import torch.utils.data as data
import torch
import h5py
import numpy as np

def data_augment(im,num):
    org_image = im.transpose(1,2,0)
    if num ==0:
        ud_image = np.flipud(org_image)
        tranform = ud_image
    elif num ==1:
        lr_image = np.fliplr(org_image)
        tranform = lr_image
    elif num ==2:
        lr_image = np.fliplr(org_image)
        lrud_image = np.flipud(lr_image)
        tranform = lrud_image
    elif num ==3:
        rotated_image1 = np.rot90(org_image)
        tranform = rotated_image1
    elif num ==4:
        rotated_image2 = np.rot90(org_image, -1)
        tranform = rotated_image2
    elif num ==5:
        rotated_image1 = np.rot90(org_image)
        ud_image1 = np.flipud(rotated_image1)
        tranform = ud_image1
    elif num ==6:
        rotated_image2 = np.rot90(org_image, -1)
        ud_image2 = np.flipud(rotated_image2)
        tranform = ud_image2
    else:
        tranform = org_image
    tranform = tranform.transpose(2,0,1)
    return tranform


class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        hf = h5py.File(file_path)
        self.data = hf.get('data')
        print(self.data.shape)
        self.target = hf.get('label_x4')

    def __getitem__(self, index):
        num = np.random.randint(0, 8)
        im_data = self.data[index,:,:,:]
        rim_data = data_augment(im_data,num)
        data = torch.from_numpy(rim_data.copy()).float()

        im_labelx4 = self.target[index,:,:,:]
        rim_labelx4 = data_augment(im_labelx4,num)
        label_x4 = torch.from_numpy(rim_labelx4.copy()).float()
        # return torch.from_numpy(self.data[index,:,:,:]).float(), torch.from_numpy(self.target[index,:,:,:]).float()
        return data, label_x4

    def __len__(self):
        return self.data.shape[0]