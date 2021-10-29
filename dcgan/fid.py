import torch
import torchvision
import os
import numpy as np
import shutil

import fid_score

def fid(netG, real_samples_path, n_samples,device, outf, delete_samples=True):
    # netG: the generator model
    # real_samples_path: path to the real data samples (this folder should contain ~5000-10000 images)
    # n_samples: the number of samples to generate with netG (should be 5000-10000)
    # device: pytorch device to perform the network passes on
    # outf: the folder where we dump our generated samples while performing the calculation
    # delete_samples: flag to delete the samples when we're done (if not, will potentially build up a lot of data)
    
    newOutf = f'{outf}/samples/'
    os.makedirs(newOutf)
    
    n_steps = n_samples // 50
    
    # according to https://discuss.pytorch.org/t/why-dont-we-put-models-in-train-or-eval-modes-in-dcgan-example/7422/3
    # when running DCGAN in inference, you need to run some generations in train mode to stabilise BN statistics,
    # THEN go to eval mode
    print('Running some generations to stabilise BN stats for DCGAN...')
    for jj in range(20):
        z_noise = torch.randn(50, nz, device=device)
        fake = netG(z_noise)
        
    netG.eval()
    if netG.training == False:
        print('netG in eval mode')
    
    # 1) generate the samples
    with torch.no_grad():
        for ii in range(n_steps):
            if ii == n_steps-1:
                batchSize = np.mod(n_samples,50)
            else:
                batchSize = 50
            print(f'\rStep {ii+1} of {n_steps}',end="")
            z_noise = torch.randn(batchSize, nz, device=device)
                  
            fake = netG(z_noise)
            
            # if our GAN outputs values in range [-1,1] from tanh, want to
            # convert to [0,1] for saving to file
            fake = (fake + 1.) /2.
            
            for jj in range(fake.size(0)):
               vutils.save_image(torch.squeeze(fake[jj, :, 16, :, :]), f'{newOutf}im_{ii}_{jj}.png')
    
    print('')
    netG.train()
    if netG.training == True:
        print('netG in train mode')
    
    # 2) calculate the fid
    fid_value = fid_score.calculate_fid_given_paths(paths = [real_samples_path, f'{newOutf}'],
                                          batch_size = 50,
                                          device = device,
                                          dims = 2048)
    
    # 3) clearup the folder
    if delete_samples:
        shutil.rmtree(f'{outf}/samples/')
    
    # 4) return the value
    return fid_value
