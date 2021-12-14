import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import numpy as np
import matplotlib.pylab as plt
import glob
import os
import pylidc as pl
from skimage.util import view_as_windows


def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)

    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
     
    return numpyImage, numpyOrigin, numpySpacing


def patch(data, x, y):
    p =  np.zeros((64, 64))     # [[0]*64 for i in range(64)]
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
    self, subsets, num_patch_per_ct
    ):
        self.subsets = subsets
        self.num_patch_per_ct = num_patch_per_ct
        self.files = glob.glob(subsets + '/subset*/*.mhd')
        self.files_seg = glob.glob(subsets + '/seg-lungs-LUNA16/*.mhd')
        #print(len(self.files))     // 888
        ...

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        lungCT, _, _ = load_itk_image(self.files[idx])      # Real scan, e.g. (133, 512, 512)
        seg, _, _ =  load_itk_image(self.subsets + '/seg-lungs-LUNA16/' + os.path.basename(self.files[idx]))       # Seg scan, e.g. (133, 512, 512)
        # Segment lung tissue.
        lungMask = np.logical_or(seg == 3, seg == 4).astype('int16')        # 1 = Lung, 0 = Non-lung

        bbox = np.array([ [0, len(lungCT[0])-1], [0, len(lungCT[1])-1], [0, len(lungCT)-1] ])
        ann = pl.query(pl.Annotation).filter(pl.Scan.series_instance_uid == os.path.basename(self.files[idx])[0:-4])[0]
        noduleMask = ann.boolean_mask(bbox=bbox)     # 1 = Nodule, 0 = Non-Nodule.
        #print(noduleMask.shape)
        #   (512, 512, 133)
        noduleMask = np.transpose(noduleMask, (2, 0, 1))
        #   (133, 512, 512)

        mask = np.logical_and(lungMask==1,noduleMask==0).astype('int16')        # 1 = lung+non-nodule, 0 = non-lung/nodule

        selectionMask = mask.copy()
        selectionMask[:] = 0
        selectionMask[:, 32:-32, 32:-32] = mask[:, 32:-32, 32:-32]
        
        valid_idx = np.stack(np.where(selectionMask==1))
        sampled_idx = np.random.randint(0,valid_idx.shape[1],self.num_patch_per_ct)
        #print('len of sampled_idx: ',len(sampled_idx))     # len of sampled_idx:  100

        patch_centres = valid_idx[:,sampled_idx]

        lungCT_pad = np.pad(lungCT,((0,0),(32,31),(32,31)),mode='constant')
        patch_view = view_as_windows(lungCT_pad, [1,64,64])
        # patch_view has size ((lungCT.shape),(patch_size))

        extractedPatches = patch_view[tuple(patch_centres)].copy() # indexing into first 3 dims gives patches for those voxels

        extractedPatches = torch.as_tensor((extractedPatches), dtype=torch.float)
        extractedPatches = extractedPatches.squeeze()
        extractedPatches = extractedPatches.unsqueeze(1)
        #print('len of extractedPatches: ',extractedPatches.size())      # len of extractedPatches:  (100, 1, 64, 64)

        return extractedPatches


#t = LunaDataset('/Users/admin/Desktop/proj/data/', 100)
#print(t[6])