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


def isLung(data, x, y):
    for i in range(x, x+64):
        for j in range(y, y+64):
            if data[i,j]==0:
                return False
    return True


def patch(data, x, y):
    p =  np.zeros((64, 64)) #[[0]*64 for i in range(64)]
    a = 0
    for i in range(x, x+64):
        b = 0
        for j in range(y, y+64):
            p[a,b] = data[i,j]
            b += 1
        a += 1
    return p
            

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
                if seg[i, jx, jy] == 0:
                    data[jx, jy] = 0
#            lungCT[i] = data        # Now lungCT is segmented.
            # Generate patches.
            for x, y in np.ndindex(448, 448):
                if isLung(data, x, y):
                    #ptch = patch(data, x, y)
                    patches.append(patch(data, x, y))
                    #print(patches[-1])
                    plt.figure()
                    plt.imshow(patches[-1])
        #patches = np.array(patches)
        return np.asarray(patches)


#t = LunaDataset('/Users/admin/Desktop/proj/data/')
#print(np.shape(t[6]))