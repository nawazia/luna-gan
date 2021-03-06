from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

import lunadataset as ldset
import fid_score
import numpy as np
import shutil
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | mnist |imagenet | folder | lfw | fake \ luna16')
parser.add_argument('--dataroot', required=False, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--num_patch_per_ct', type=int, default=256, help='number of patches per CT')
parser.add_argument('--batch_size', type=int, default=16, help='batch size, note: num_patch_per_ct//batch_size must be 0')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=512, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr_d', type=float, default=0.0001, help='learning rate for Discriminator, default=0.0001')
parser.add_argument('--lr_g', type=float, default=0.0001, help='learning rate for Generator, default=0.0001')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--generate', action='store_true', help='generate')
parser.add_argument('--dry-run', action='store_true', help='check a single training cycle works')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--classes', default='bedroom', help='comma separated list of classes for the lsun data set')
parser.add_argument('--real_samples_path', default=None, help='path to real samples dir, for FID calculation')
parser.add_argument('--num_gen', type=int, default=33, help='number of generated samples per epoch')


opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
  
if opt.dataroot is None and str(opt.dataset).lower() != 'fake':
    raise ValueError("`dataroot` parameter is required for dataset \"%s\"" % opt.dataset)

if opt.real_samples_path is None:
    raise ValueError("`real_samples_path` parameter is required for FID calculations!")

assert np.mod(opt.num_patch_per_ct, opt.batch_size) == 0

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    nc=3
elif opt.dataset == 'lsun':
    classes = [ c + '_train' for c in opt.classes.split(',')]
    dataset = dset.LSUN(root=opt.dataroot, classes=classes,
                        transform=transforms.Compose([
                            transforms.Resize(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
    nc=3
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    nc=3

elif opt.dataset == 'mnist':
        dataset = dset.MNIST(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                           ]))
        nc=1

elif opt.dataset == 'fake':
    dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
                            transform=transforms.ToTensor())
    nc=3

elif opt.dataset == 'luna16':
    dataset = ldset.LunaDataset('/content/drive/My Drive/luna16/data/', opt.num_patch_per_ct)
    nc=1

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                         shuffle=True, num_workers=int(opt.workers))

device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


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
        z_noise = torch.randn(50, nz, 1, 1, device=device)
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
            z_noise = torch.randn(batchSize, nz, 1, 1, device=device)
                  
            fake = netG(z_noise)
            
            # if our GAN outputs values in range [-1,1] from tanh, want to
            # convert to [0,1] for saving to file
            fake = (fake + 1.) /2.
            
            for jj in range(fake.size(0)):
               vutils.save_image(fake[jj, :, :, :], f'{newOutf}im_{ii}_{jj}.png')
    
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


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


netG = Generator(ngpu).to(device)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = nn.BCELoss()

# fixed_noise = torch.randn(opt.num_patch_per_ct, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))

if opt.dry_run:
    opt.niter = 1

if opt.generate:
    assert opt.netG != ''
    for i in range(opt.num_gen):
        netG.zero_grad()
        fake = netG(torch.randn(1, nz, 1, 1, device=device))
        vutils.save_image(fake.detach(),
                '%s/generated/fake_samples_%03d.png' % (opt.outf, i),
                normalize=True)
    opt.niter = 0

fidscores = np.zeros(opt.niter)

for epoch in range(opt.niter):
    for i, data_full in enumerate(dataloader, 0):

        data_full_shuffle = data_full[0][torch.randperm(data_full.shape[1])] # have to index into data_full, since dataloader returns it with batchsize of 1
        data_full_shuffle_split = torch.split(data_full_shuffle,opt.batch_size)
        
        data_full_split = torch.split(data_full[0],opt.batch_size)

        for jj in range(len(data_full_split)):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            #print('data: ',data.size())     # 1,100,1,64,64
            #data = data.permute(1,0,2,3,4)
            netD.zero_grad()
            real_cpu = data_full_split[jj].to(device)
            #print('real_cpu: ',real_cpu.size())        # 100,1,64,64
            batch_size = real_cpu.size(0)
            #print('batch_size: ',batch_size)       # 100
            label = torch.full((batch_size,), real_label,
                            dtype=real_cpu.dtype, device=device)

            output = netD(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            # real_cpu = data_full_shuffle_split[jj].to(device)
            # batch_size = real_cpu.size(0)
            # label = netD(real_cpu)
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()


            #Re: D_G_z1 & D_G_z2 - The first number is before D is updated and the second number is after D is updated.

            print('[%d/%d][%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                % (epoch, opt.niter, i, len(dataloader), jj, len(data_full_split),
                    errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            if (i == len(dataloader)-1) & (jj == len(data_full_split)-1):
                vutils.save_image(real_cpu,
                        '%s/real_samples.png' % opt.outf,
                        normalize=True)
                newOutf = f'{opt.outf}/{epoch}/'
                os.makedirs(newOutf)
                for i in range(opt.num_gen):
                    fake = netG(torch.randn(1, nz, 1, 1, device=device))
                    vutils.save_image(fake.detach(),
                            '%s/fake_samples_%03d.png' % (newOutf, i),
                            normalize=True)
                FID = fid(netG, opt.real_samples_path, 10000, device, opt.outf)
                print('FID: %.4f' % (FID))
                fidscores[epoch] = FID

            if opt.dry_run:
                break
    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))

plt.figure()
plt.plot(fidscores, '.k')
plt.ylabel('FID score')
plt.xlabel('Epoch number')
plt.savefig('FIDs.png')
