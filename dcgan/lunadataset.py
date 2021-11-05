import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import numpy as np
import matplotlib.pylab as plt


def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    #print(f'array type: {type(numpyImage)}')
    #print(f'shape of array: {np.shape(numpyImage)}')
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
     
    return numpyImage, numpyOrigin, numpySpacing

class LunaDataset(Dataset):
    def __init__(
    self, file 
    ):
        self.lungCT, _, _ = load_itk_image(file)
        ...

    def __len__(self):
        return len(self.lungCT)

    def __getitem__(self, idx):
        return self.lungCT[idx]