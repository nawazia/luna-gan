import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import numpy as np
import matplotlib.pylab as plt
import glob
import os


def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)

    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
     
    return numpyImage, numpyOrigin, numpySpacing


class LunaDataset(Dataset):
    def __init__(
    self, subsets 
    ):
        self.subs = subsets
        self.files = glob.glob(subsets + '/subset*/*.mhd')
        #print(len(self.files))     // 888
        ...

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        patches = []

        lungCT, _, _ = load_itk_image(self.files[idx])      # Real scan, e.g. (133, 512, 512)
        # Segment lung tissue.
        seg, _, _ =  load_itk_image(self.subs + '/seg-lungs-LUNA16/' + os.path.basename(self.files[idx]))       # Seg scan, e.g. (133, 512, 512)
        for i, data in enumerate(lungCT):
            for jx,jy in np.ndindex(seg[i].shape):
                #print(seg[jx,jy])
                if seg[i,jx,jy]==0:
                    data[jx,jy] = 0
            patches.append(data)
        return patches[idx]


#t = LunaDataset('/Users/admin/Desktop/proj/data/')
#print(np.shape(t[6]))