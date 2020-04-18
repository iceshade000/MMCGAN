from __future__ import print_function
import torch.utils.data as data
from torch.utils.data import DataLoader
from PIL import Image
import os
import os.path
import errno
import torch
import codecs
import math
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

def sgdr(period, batch_idx):
    batch_idx = float(batch_idx)
    restart_period = period
    while batch_idx / restart_period > 1.:
        batch_idx = batch_idx - restart_period
    restart_period = restart_period * 2.
    radians = math.pi * (batch_idx / restart_period)
    return 0.5 * (1.0 + math.cos(radians))

class MMMC(data.Dataset):
    def __init__(self, dataset,name='C10',code=None,Encoder=None,Decoder=None,max_epoch=30,batch_size=128,SGDR=True,no_target=False):
        self.root = dataset.root
        self.transform = dataset.transform
        self.target_transform = dataset.target_transform
        self.train = dataset.train  # training set or test set
        self.no_target=no_target
        self.data=dataset.data

        self.labels=dataset.labels
        self.name=name

        if code is not None:
            self.code=code
        else:
            optimizer = optim.Adam([{'params':Encoder.parameters()},{'params':Decoder.parameters()}], lr=1e-4, betas=(0.5, 0.9))
            dataloader=DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=4)
            iteration=0
            for i in range(max_epoch):
                for inputs,targets in dataloader:
                    optimizer.zero_grad()
                    if SGDR:
                        batch_lr = 1e-3 * sgdr(10, iteration)
                        for p in optimizer.param_groups:
                            p['lr'] = batch_lr
                    inputs,targets=inputs.cuda(),targets.cuda()
                    #inputs=inputs*2-1
                    if no_target:
                        mid=Encoder(inputs)
                        recover=Decoder(mid)
                    else:
                        mid=Encoder(inputs,targets)
                        recover=Decoder(mid,targets)
                    L1 = 0.5 * torch.sum((recover - inputs) ** 2) / inputs.size(0)
                    #L2 = 0.5 * torch.sum(mid ** 2) / inputs.size(0)
                    L2=0.5*torch.mean(mid**2)
                    L = L1 + L2 * 0.1
                    L.backward()
                    optimizer.step()
                    iteration+=1
                    if iteration%100==0:
                        print('epoch:',i,',iteration:',iteration,',Loss:',L1.item(),L2.item())
            print('encoder training complete!')
            dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            result=[]
            for inputs, targets in dataloader:
                inputs, targets = inputs.cuda(), targets.cuda()
                if no_target:
                    mid=Encoder(inputs)
                else:
                    mid=Encoder(inputs,targets)
                result.append(mid.data.cpu().numpy())
            code=np.concatenate(tuple(result),0)
            mean = np.mean(code, 0, keepdims=True)
            std = np.std(code, 0, keepdims=True)
            self.code=(code - mean) / std
            np.save('Code/'+name+'.npy',self.code)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        if self.name=='C10':
            img, target = self.data[index], self.labels[index]

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)
            code=self.code[index]
            code=torch.Tensor(code)
            return img, target,code
        else:
            print('dataset name cannot identify, please use: C10')

    def __len__(self):
        return len(self.data)

import h5py as h5
class MMMC_hdf5(data.Dataset):
    def __init__(self, dataset,name='I128_hdf5',code=None,Encoder=None,Decoder=None,max_epoch=30,batch_size=128,SGDR=True,no_target=False):
        self.root = dataset.root
        self.transform = dataset.transform
        self.target_transform = dataset.target_transform

        self.num_imgs = dataset.num_imgs

        # load the entire dataset into memory?
        self.load_in_mem = dataset.load_in_mem

        if self.load_in_mem:
            self.data = dataset.data
            self.labels = dataset.labels

        self.no_target=no_target

        if code is not None:
            self.code=code
        else:
            optimizer = optim.Adam([{'params':Encoder.parameters()},{'params':Decoder.parameters()}], lr=1e-4, betas=(0.5, 0.9))
            dataloader=DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=4)
            iteration=0
            for i in range(max_epoch):
                for inputs,targets in dataloader:
                    optimizer.zero_grad()
                    if SGDR:
                        batch_lr = 1e-3 * sgdr(10, iteration)
                        for p in optimizer.param_groups:
                            p['lr'] = batch_lr
                    inputs,targets=inputs.cuda(),targets.cuda()
                    #inputs=inputs*2-1
                    if no_target:
                        mid=Encoder(inputs)
                        recover=Decoder(mid)
                    else:
                        mid=Encoder(inputs,targets)
                        recover=Decoder(mid,targets)
                    L1 = 0.5 * torch.sum((recover - inputs) ** 2) / inputs.size(0)
                    #L2 = 0.5 * torch.sum(mid ** 2) / inputs.size(0)
                    L2=0.5*torch.mean(mid**2)
                    L = L1 + L2 * 0.1
                    L.backward()
                    optimizer.step()
                    iteration+=1
                    if iteration%100==0:
                        print('epoch:',i,',iteration:',iteration,',Loss:',L1.item(),L2.item())
            print('encoder training complete!')
            dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            result=[]
            for inputs, targets in dataloader:
                inputs, targets = inputs.cuda(), targets.cuda()
                if no_target:
                    mid=Encoder(inputs)
                else:
                    mid=Encoder(inputs,targets)
                result.append(mid.data.cpu().numpy())
            code=np.concatenate(tuple(result),0)
            mean = np.mean(code, 0, keepdims=True)
            std = np.std(code, 0, keepdims=True)
            self.code=(code - mean) / std
            np.save('Code/'+name+'.npy',self.code)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        if self.load_in_mem:
            img, target = self.data[index], self.labels[index]
        else:
            with h5.File(self.root, 'r') as f:
                img = f['imgs'][index]
                target = f['labels'][index]

        img = ((torch.from_numpy(img).float() / 255) - 0.5) * 2

        if self.target_transform is not None:
            target = self.target_transform(target)


        code=self.code[index]
        code=torch.Tensor(code)
        return img, int(target),code


    def __len__(self):
        return self.num_imgs

import sys

class MMMC_Folder(data.Dataset):
    def __init__(self, dataset,name='CelebA',code=None,Encoder=None,Decoder=None,max_epoch=30,batch_size=128,SGDR=True):
        classes, class_to_idx = dataset.classes, dataset.class_to_idx
        samples = dataset.samples
        if len(samples) == 0:
            raise(RuntimeError('no samples'))

        self.root = dataset.root
        self.loader = dataset.loader
        self.extensions = dataset.extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = dataset.transform
        self.target_transform = dataset.target_transform
        #============================================================
        if code is not None:
            self.code=code
        else:
            optimizer = optim.Adam([{'params':Encoder.parameters()},{'params':Decoder.parameters()}], lr=1e-4, betas=(0.5, 0.9))
            dataloader=DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=4)
            iteration=0
            for i in range(max_epoch):
                for inputs,targets in dataloader:
                    optimizer.zero_grad()
                    if SGDR:
                        batch_lr = 1e-3 * sgdr(10, iteration)
                        for p in optimizer.param_groups:
                            p['lr'] = batch_lr
                    inputs,targets=inputs.cuda(),targets.cuda()
                    #inputs=inputs*2-1
                    mid=Encoder(inputs,targets)
                    recover=Decoder(mid,targets)
                    L1 = 0.5 * torch.sum((recover - inputs) ** 2) / inputs.size(0)
                    #L2 = 0.5 * torch.sum(mid ** 2) / inputs.size(0)
                    L2=0.5*torch.mean(mid**2)
                    L = L1 + L2 * 0.1
                    L.backward()
                    optimizer.step()
                    iteration+=1
                    if iteration%100==0:
                        print('epoch:',i,',iteration:',iteration,',Loss:',L1.item(),L2.item())
            print('encoder training complete!')
            dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            result=[]
            for inputs, targets in dataloader:
                inputs, targets = inputs.cuda(), targets.cuda()
                mid=Encoder(inputs,targets)
                result.append(mid.data.cpu().numpy())
            code=np.concatenate(tuple(result),0)
            mean = np.mean(code, 0, keepdims=True)
            std = np.std(code, 0, keepdims=True)
            self.code=(code - mean) / std
            np.save('Code/'+name+'.npy',self.code)

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        code = self.code[index]
        code = torch.Tensor(code)
        return sample, target,code

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

