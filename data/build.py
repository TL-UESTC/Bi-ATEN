# --------------------------------------------------------
# Reference from https://github.com/microsoft/Swin-Transformer
# --------------------------------------------------------

import os
from random import shuffle

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from .samplers import SubsetRandomSampler
from torch.distributions import Beta
from torch.utils.data import DataLoader, RandomSampler
import matplotlib.pyplot as plt
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', ]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    return Image.open(path).convert('RGB')


def make_dataset(root, label):
    images = []
    labeltxt = open(label)
    for line in labeltxt:
        data = line.strip().split(' ')
        if is_image_file(data[0]):
            path = os.path.join(root, data[0])
        gt = int(data[1])
        item = (path, gt)
        images.append(item)
    return images


def make_augmix_dataset(root, label, class_num=345):
    images = []
    labels = []
    labeltxt = open(label)
    for line in labeltxt:
        data = line.strip().split(' ')
        if is_image_file(data[0]):
            path = os.path.join(root, data[0])
        gt = int(data[1])
        item = (path, gt)
        images.append(path)
        labels.append(F.one_hot(torch.tensor(gt), num_classes=class_num))
    #images = torch.stack(images, 0)
    #labels = torch.stack(labels, 0)
    return images, labels


class ObjectImage(torch.utils.data.Dataset):
    def __init__(self, root, label, transform=None, loader=default_loader, return_path=False):
        imgs = make_dataset(root, label)
        self.root = root
        self.label = label
        self.imgs = imgs
        self.transform = transform
        self.loader = loader
        self.return_path = return_path

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            if type(self.transform).__name__ == 'list':
                img = [t(img) for t in self.transform]
            else:
                img = self.transform(img)
        if not self.return_path:
            return img, target
        else:
            return img, target, path

    def __len__(self):
        return len(self.imgs)


class ObjectImage_mul(torch.utils.data.Dataset):
    """
    :return img, label, index for pseudo labels
    """

    def __init__(self, root, label, transform=None, loader=default_loader):
        imgs = make_dataset(root, label)
        self.root = root
        self.label = label
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            if type(self.transform).__name__ == 'list':
                img = [t(img) for t in self.transform]
            else:
                img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.imgs)


class pseudo_dataset(torch.utils.data.Dataset):
    """
    :return img, label, index for pseudo labels
    """

    def __init__(self, root, label, ps, class_num, transform=None, loader=default_loader):
        imgs = make_dataset(root, label)
        self.root = root
        self.label = label
        self.imgs = imgs
        self.transform = transform
        self.loader = loader
        self.ps = F.one_hot(ps, num_classes=class_num)
        assert len(self.ps)==len(self.imgs)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        img = self.transform(img)
        target = self.ps[index].float()
        return img, target, index

    def __len__(self):
        return len(self.imgs)
    

class AugMix_source(torch.utils.data.Dataset):
    """
    :return img, label, index for pseudo labels
    """

    def __init__(self, label, config, class_num=345, root='', loader=default_loader):
        path, labels = make_augmix_dataset(root, label, class_num=class_num)
        self.img_path = path
        self.mixed_labels = []
        self.labels = torch.stack(labels, 0)
        assert len(self.labels)==len(self.img_path)
        self.range_idx = torch.arange(len(self.img_path))
        self.class_num = class_num
        self.strong_transform = transforms.Compose([
            transforms.Resize((config.DATA.IMG_SIZE + 32, config.DATA.IMG_SIZE + 32)),
            transforms.CenterCrop((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)),
            transforms.RandAugment(3),
            #transforms.AutoAugment(),
            transforms.ToTensor(),
            transforms.Normalize(config.DATA.MEAN, config.DATA.STD),
        ])
        self.loader = loader
        self.beta = Beta(0.6, 0.6)

    def __getitem__(self, index):
        if index < len(self.img_path):
            img = self.loader(self.img_path[index])
            target = self.labels[index].float()
            img = self.strong_transform(img)
            return img, target 
        else:
            idx1, idx2 = torch.randint(0,len(self.img_path),(1,)), torch.randint(0,len(self.img_path),(1,))
            img1, img2 = self.loader(self.img_path[idx1]), self.loader(self.img_path[idx2])
            target1, target2 = self.labels[idx1].float(), self.labels[idx2].float()
            fig1 = self.strong_transform(img1)
            fig2 = self.strong_transform(img2)
            b = self.beta.sample((1,))
            img = b*fig1 + (1-b)*fig2
            target = b*target1 + (1-b)*target2
            return img, target.squeeze()

    def __len__(self):
        return len(self.img_path)*2
    

class AugMix(torch.utils.data.Dataset):
    """
    :return img, label, index for pseudo labels
    """

    def __init__(self, label, config, idx, mapping, ins_num, ps, class_num=345, root='', loader=default_loader):
        '''
        idx: index for TRUSTED sample
        mapping: reciprocal pairs
        ins_num: number of instance per class
        '''
        path, labels = make_augmix_dataset(root, label)
        self.img_path = path
        self.mixed_labels = []
        self.labels = torch.stack(labels, 0)
        if len(ps.shape) == 1:
            ps = F.one_hot(ps, num_classes=class_num)
        self.ps = ps
        assert len(self.ps)==len(self.img_path) and len(self.img_path)==len(self.labels)
        self.trusted_idx = torch.arange(len(ps))[idx]
        self.untrusted_idx = torch.arange(len(ps))[~idx]
        self.range_idx = torch.arange(len(self.img_path))
        self.ins_num = ins_num
        self.mapping = mapping
        self.class_num = class_num
        self.strong_transform = transforms.Compose([
            transforms.Resize((config.DATA.IMG_SIZE + 32, config.DATA.IMG_SIZE + 32)),
            transforms.CenterCrop((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)),
            transforms.RandAugment(3),
            #transforms.AutoAugment(),
            transforms.ToTensor(),
            transforms.Normalize(config.DATA.MEAN, config.DATA.STD),
        ])
        self.no_aug = transforms.Compose([
            transforms.Resize((config.DATA.IMG_SIZE + 32, config.DATA.IMG_SIZE + 32)),
            transforms.CenterCrop((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(config.DATA.MEAN, config.DATA.STD)
        ])
        self.loader = loader
        self.beta = Beta(0.5, 0.5)
        self.complement_data()
        self.auglen = len(self.img_path)
        self.range_idx = torch.arange(len(self.img_path))
        #self.mixup()

    def complement_data(self):
        origin_labels = self.labels
        origin_ps = self.ps
        origin_imgpath = self.img_path
        for c in range(self.class_num):
            c_idx = origin_labels.max(1)[1]==c
            class_cnt = (c_idx).sum()
            if class_cnt == 0:
                continue
            class_idx = self.range_idx[c_idx]
            if class_cnt < self.ins_num:
                to_complement = self.ins_num - class_cnt
                class_idx_idx = torch.randint(0, len(class_idx), (to_complement,))
                sample_idx = class_idx[class_idx_idx]
                tmp = [origin_imgpath[i] for i in sample_idx]
                self.img_path += tmp
                self.labels = torch.concat([self.labels, origin_labels[sample_idx]])
                self.ps = torch.concat([self.ps, origin_ps[sample_idx]])

    def mixup(self):
        flag = [False]*self.class_num
        for c in range(self.class_num):
            if flag[c] or self.mapping[c]==-1:
                flag[c] = True
                continue
            c1 = c
            c2 = self.mapping[c]
            flag[c1] = True
            flag[c2] = True
            idx_c1 = self.ps.max(1)[1]==c1
            idx_c2 = self.ps.max(1)[1]==c2
            c1_num = (idx_c1).sum()
            c2_num = (idx_c2).sum()
            if c1_num == 0 or c2_num == 0:
                continue
            c1_idx = self.range_idx[idx_c1]
            c2_idx = self.range_idx[idx_c2]
            mix_num = int(torch.max(c1_num, c2_num)*1.2)
            c1_idx_idx = torch.randint(0, len(c1_idx), (mix_num,))
            c2_idx_idx = torch.randint(0, len(c2_idx), (mix_num,))
            c1_to_mix = c1_idx[c1_idx_idx]
            c2_to_mix = c2_idx[c2_idx_idx]
            for i in range(mix_num):
                path1 = self.img_path[c1_to_mix[i]]
                path2 = self.img_path[c2_to_mix[i]]
                ps1 = self.ps[c1_to_mix[i]]
                ps2 = self.ps[c2_to_mix[i]]
                self.img_path.append([path1, path2])
                self.mixed_labels.append([ps1.clone(), ps2.clone()])

    def __repr__(self) -> str:
        origin_len = self.idx.sum()
        acccnt = (self.labels[:origin_len].max(1)[1]==self.ps[:origin_len].max(1)[1]).sum()
        acc = round((acccnt / origin_len).item(), 3)
        repr = "instance per class={}\nstrongly-aug instances={}\nmix instances={}\nps label acc={}/{}, {}".format(self.ins_num, self.auglen, len(self.img_path)-self.auglen, acccnt, origin_len, acc)
        return repr 

    def __getitem__(self, index):
        if index < self.auglen:
            img = self.loader(self.img_path[index])
            target = self.ps[index].float()
            img = self.strong_transform(img)
            return img, target 
        else:
            index_ = index - self.auglen
            img1, img2 = self.img_path[index][0], self.img_path[index][1]
            ps1, ps2 = self.mixed_labels[index_]
            fig1 = self.strong_transform(self.loader(img1))
            fig2 = self.strong_transform(self.loader(img2))
            b = self.beta.sample((1,))
            img = b*fig1 + (1-b)*fig2
            target = b*ps1 + (1-b)*ps2
            return img, target

    def __len__(self):
        return self.auglen
    
    def show(self, imgs, tit, index):
        if not isinstance(imgs, list):
            imgs = [imgs]
        fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = img.detach()
            img = TF.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        plt.title(tit)
        plt.savefig('./see/{}.pdf'.format(index))
    
    def see(self, index):
        assert 0<=index<len(self.mixed_labels)
        img1, img2 = self.img_path[index+self.auglen][0], self.img_path[index+self.auglen][1]
        ps1, ps2 = self.mixed_labels[index]
        b = self.beta.sample((1,))
        title = "{}*{} + {}*{}".format(b.item(), ps1.max(0)[1], 1-b.item(), ps2.max(0)[1])
        origin1 = self.no_aug(self.loader(img1))
        origin2 = self.no_aug(self.loader(img2))
        strong1 = self.strong_transform(self.loader(img1))
        strong2 = self.strong_transform(self.loader(img2))
        strong_mix = b*strong1 + (1-b)*strong2
        origin_mix = b*origin1 + (1-b)*origin2
        fig_lis = [origin1, origin2, origin_mix, strong1, strong2, strong_mix]
        grid = make_grid(fig_lis, 3)
        self.show(grid, title, index)
    

class Fixmatch(torch.utils.data.Dataset):
    def __init__(self, label, config, idx, ps, class_num=345, root='', loader=default_loader):
        '''
        idx: available index
        '''
        path, labels = make_augmix_dataset(root, label)
        self.img_path = []
        for i in range(len(path)):
            if idx[i]:
                self.img_path.append(path[i])
        self.labels = torch.stack(labels, 0)[idx]
        if len(ps.shape) == 1:
            ps = F.one_hot(ps, num_classes=class_num)
        self.ps = ps[idx]
        self.range_idx = torch.arange(len(self.img_path))
        self.class_num = class_num
        self.strong_transform = transforms.Compose([
            transforms.Resize((config.DATA.IMG_SIZE + 32, config.DATA.IMG_SIZE + 32)),
            transforms.CenterCrop((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize(config.DATA.MEAN, config.DATA.STD),
        ])
        self.weak_transform = transforms.Compose([
            transforms.Resize((config.DATA.IMG_SIZE + 32, config.DATA.IMG_SIZE + 32)),
            transforms.CenterCrop((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(config.DATA.MEAN, config.DATA.STD),
        ])
        self.loader = loader
        assert len(self.img_path)==len(self.labels)

    def __getitem__(self, index):
        target = self.ps[index]
        img = self.loader(self.img_path[index])
        strong_img = self.strong_transform(img)
        #weak_img = self.weak_transform(img)
        return strong_img, target.float()

    def __len__(self):
        return len(self.img_path)
    

def build_augmix_loader(config, mapping, idx, ins_num, ps, args):
    # idx: augmix index
    # ins_num: samples number in each class for augmix
    # ps: pseudo label (all data)
    dsets = {
        'augmix': {},
        'fixmatch': {},
        'val': {},
    }
    dset_loaders = {
        'augmix': {},
        'fixmatch': {},
        'val': {},
    }
    target_root = os.path.join(config.DATA.DATA_PATH, config.DATA.TARGET + '.txt')
    augmix = AugMix(target_root, config, idx, mapping, ins_num, ps, class_num=args.class_num)
    fixmatch = Fixmatch(target_root, config, ~idx, ps, class_num=args.class_num)
    transform = build_transform(is_train=False, config=config)
    val = ObjectImage_mul('', target_root, transform)
    dsets['augmix'] = augmix
    dsets['fixmatch'] = fixmatch
    dsets['val'] = val
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    sampler_augmix = torch.utils.data.DistributedSampler(
        dsets['augmix'], num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    sampler_fixmatch = torch.utils.data.DistributedSampler(
        dsets['fixmatch'], num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    #sampler_augmix = RandomSampler(dsets['augmix'])
    #sampler_fixmatch = RandomSampler(dsets['fixmatch'])
    indices_t = np.arange(dist.get_rank(), len(dsets['val']), dist.get_world_size())
    sampler_val_t = SubsetRandomSampler(indices_t)

    dset_loaders['augmix'] = torch.utils.data.DataLoader(
        dsets['augmix'], sampler=sampler_augmix,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    dset_loaders['fixmatch'] = torch.utils.data.DataLoader(
        dsets['fixmatch'], sampler=sampler_fixmatch,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    dset_loaders['val'] = torch.utils.data.DataLoader(
        dsets['val'], sampler=sampler_val_t,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    return dsets, dset_loaders


def build_sourceonly_loader(config, args):
    dsets = {i:None for i in args.source_list}
    dsets = {
        'source_train': {},
        'source_val': {},
    }
    dset_loaders = {i:None for i in args.source_list}
    dset_loaders = {
        'source_train': {},
        'source_val': {},
    }
    transform = build_transform(is_train=True, config=config)
    target_root = os.path.join(config.DATA.DATA_PATH, args.source + '.txt')
    #dsets['source_train'] = AugMix_source(target_root, config, args.class_num)
    dsets['source_train'] = ObjectImage_mul('', target_root, transform,)
    transform = build_transform(is_train=False, config=config)
    dsets['source_val'] = ObjectImage('', target_root, transform, return_path=config.RETURN_PATH)
    for s in args.source_list:
        transform = build_transform(is_train=True, config=config)
        target_root = os.path.join(config.DATA.DATA_PATH, s + '.txt')
        dsets[s] = ObjectImage('', target_root, transform, return_path=config.RETURN_PATH)
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    sampler_train_target = torch.utils.data.DistributedSampler(
        dsets['source_train'], num_replicas=num_tasks, rank=global_rank, shuffle=True
    )

    indices_t = np.arange(dist.get_rank(), len(dsets['source_val']), dist.get_world_size())
    sampler_val_t = SubsetRandomSampler(indices_t)

    dset_loaders['source_train'] = torch.utils.data.DataLoader(
        dsets['source_train'], sampler=sampler_train_target,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
    )

    dset_loaders['source_val'] = torch.utils.data.DataLoader(
        dsets['source_val'], sampler=sampler_val_t,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    for s in args.source_list:
        indices_t = np.arange(dist.get_rank(), len(dsets[s]), dist.get_world_size())
        sampler_val_t = SubsetRandomSampler(indices_t)
        dset_loaders[s] = torch.utils.data.DataLoader(
            dsets[s], sampler=sampler_val_t,
            batch_size=config.DATA.BATCH_SIZE,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=False,
        )

    return dsets, dset_loaders


def build_ps_loader(config, args, ps):
    dsets = {
        'target_train': {},
        'val': {},
    }
    dset_loaders = {
        'target_train': {},
        'val': {},
    }
    transform = build_transform(is_train=True, config=config)
    target_root = os.path.join(config.DATA.DATA_PATH, config.DATA.TARGET + '.txt')
    dsets['target_train'] = pseudo_dataset('', target_root, ps, args.class_num, transform)
    transform = build_transform(is_train=False, config=config)
    dsets['val'] = ObjectImage_mul('', target_root, transform)

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    print(num_tasks, global_rank)
    sampler_train_target = torch.utils.data.DistributedSampler(
        dsets['target_train'], num_replicas=num_tasks, rank=global_rank, shuffle=True
    )

    indices_t = np.arange(dist.get_rank(), len(dsets['val']), dist.get_world_size())
    sampler_val_t = SubsetRandomSampler(indices_t)

    dset_loaders['target_train'] = torch.utils.data.DataLoader(
        dsets['target_train'], sampler=sampler_train_target,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
    )

    dset_loaders['val'] = torch.utils.data.DataLoader(
        dsets['val'], sampler=sampler_val_t,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    return dsets, dset_loaders
 

def build_tar_loader(config, args):
    dsets = {
        'target_train': {},
        'val': {},
    }
    dset_loaders = {
        'target_train': {},
        'val': {},
    }
    transform = build_transform(is_train=True, config=config)
    target_root = os.path.join(config.DATA.DATA_PATH, config.DATA.TARGET + '.txt')
    dsets['target_train'] = ObjectImage_mul('', target_root, transform)
    transform = build_transform(is_train=False, config=config)
    dsets['val'] = ObjectImage_mul('', target_root, transform)

    num_tasks = dist.get_world_size()   # 1
    global_rank = dist.get_rank()   # 0
    sampler_train_target = torch.utils.data.DistributedSampler(
        dsets['target_train'], num_replicas=num_tasks, rank=global_rank, shuffle=True
    )

    indices_t = np.arange(dist.get_rank(), len(dsets['val']), dist.get_world_size())
    sampler_val_t = SubsetRandomSampler(indices_t)

    dset_loaders['target_train'] = torch.utils.data.DataLoader(
        dsets['target_train'], sampler=sampler_train_target,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
    )

    dset_loaders['val'] = torch.utils.data.DataLoader(
        dsets['val'], sampler=sampler_val_t,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    return dsets, dset_loaders


def build_sel_loader(config, args):
    dsets = {
        'target_train': {},
        'source_train': {},
        'target_val': {},
        'source_val': {},
    }
    dset_loaders = {
        'target_train': {},
        'source_train': {},
        'target_val': {},
        'source_val': {},
    }
    dsets['source_train'], dsets['target_train'] = build_sel_dataset(is_train=True, config=config, args=args)
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")

    dsets['source_val'], dsets['target_val'] = build_sel_dataset(is_train=False, config=config, args=args)
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    sampler_train_source = torch.utils.data.DistributedSampler(
        dsets['source_train'], num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    sampler_train_target = torch.utils.data.DistributedSampler(
        dsets['target_train'], num_replicas=num_tasks, rank=global_rank, shuffle=True
    )

    indices_t = np.arange(dist.get_rank(), len(dsets['target_val']), dist.get_world_size())
    sampler_val_t = SubsetRandomSampler(indices_t)

    indices_s = np.arange(dist.get_rank(), len(dsets['source_val']), dist.get_world_size())
    sampler_val_s = SubsetRandomSampler(indices_s)

    dset_loaders['source_train'] = torch.utils.data.DataLoader(
        dsets['source_train'], sampler=sampler_train_source,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
    )

    dset_loaders['target_train'] = torch.utils.data.DataLoader(
        dsets['target_train'], sampler=sampler_train_target,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
    )

    dset_loaders['source_val'] = torch.utils.data.DataLoader(
        dsets['source_val'],
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    dset_loaders['target_val'] = torch.utils.data.DataLoader(
        dsets['target_val'], sampler=sampler_val_t,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    return dsets, dset_loaders


def build_loader(config):
    dsets = {
        'target_train': {},
        'source_train': {},
        'target_val': {},
        'source_val': {},
    }
    dset_loaders = {
        'target_train': {},
        'source_train': {},
        'target_val': {},
        'source_val': {},
    }
    dsets['source_train'], dsets['target_train'] = build_dataset(is_train=True, config=config)
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")

    dsets['source_val'], dsets['target_val'] = build_dataset(is_train=False, config=config)
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    sampler_train_source = torch.utils.data.DistributedSampler(
        dsets['source_train'], num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    sampler_train_target = torch.utils.data.DistributedSampler(
        dsets['target_train'], num_replicas=num_tasks, rank=global_rank, shuffle=True
    )

    indices_t = np.arange(dist.get_rank(), len(dsets['target_val']), dist.get_world_size())
    sampler_val_t = SubsetRandomSampler(indices_t)

    indices_s = np.arange(dist.get_rank(), len(dsets['source_val']), dist.get_world_size())
    sampler_val_s = SubsetRandomSampler(indices_s)

    dset_loaders['source_train'] = torch.utils.data.DataLoader(
        dsets['source_train'], sampler=sampler_train_source,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
    )

    dset_loaders['target_train'] = torch.utils.data.DataLoader(
        dsets['target_train'], sampler=sampler_train_target,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
    )

    dset_loaders['source_val'] = torch.utils.data.DataLoader(
        dsets['source_val'],
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    dset_loaders['target_val'] = torch.utils.data.DataLoader(
        dsets['target_val'], sampler=sampler_val_t,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    return dsets, dset_loaders

def build_loader_parallel(config):
    dsets = {
        'target_train': {},
        'source_train': {},
        'target_val': {},
        'source_val': {},
    }
    dset_loaders = {
        'target_train': {},
        'source_train': {},
        'target_val': {},
        'source_val': {},
    }
    dsets['source_train'], dsets['target_train'] = build_dataset(is_train=True, config=config)
    print(f"Successfully build train dataset")

    dsets['source_val'], dsets['target_val'] = build_dataset(is_train=False, config=config)
    print(f"Successfully build val dataset")


    dset_loaders['source_train'] = torch.utils.data.DataLoader(
        dsets['source_train'], shuffle=True,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
    )

    dset_loaders['target_train'] = torch.utils.data.DataLoader(
        dsets['target_train'], shuffle=True,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
    )

    dset_loaders['source_val'] = torch.utils.data.DataLoader(
        dsets['source_val'],
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    dset_loaders['target_val'] = torch.utils.data.DataLoader(
        dsets['target_val'],
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    return dsets, dset_loaders


def merge_txt(root, domain, rdict):
    allx = []
    for d in domain:
        r = rdict[d]
        path = root + d + '.txt'
        x = np.loadtxt(path, dtype=str)
        allx.append(x[:int(len(x)*r)])
    allx = np.concatenate(allx)
    return allx


def build_sel_dataset(is_train, config, args):
    transform = build_transform(is_train, config)
    tosave = merge_txt(args.sel_root+args.target+'/', args.source_list, {i:0.2 for i in args.source_list})
    targetpos = args.sel_root+args.target+'_merge.txt'
    np.savetxt(targetpos, tosave, fmt='%s')
    print("Saved merged dataset to {}, Length={}".format(targetpos, len(tosave)))
    if is_train: 
        #source_root = os.path.join(config.DATA.DATA_PATH, config.DATA.SOURCE + '.txt')
        source_dataset = ObjectImage_mul('', targetpos, transform)
        target_root = os.path.join(config.DATA.DATA_PATH, config.DATA.TARGET + '.txt')
        target_dataset = ObjectImage_mul('', target_root, transform)
        return source_dataset, target_dataset
    else:
        #source_root = os.path.join(config.DATA.DATA_PATH, config.DATA.SOURCE + '.txt')
        source_dataset = ObjectImage('', targetpos, transform, return_path=config.RETURN_PATH)
        target_root = os.path.join(config.DATA.DATA_PATH, config.DATA.TARGET + '.txt')
        target_dataset = ObjectImage('', target_root, transform, return_path=config.RETURN_PATH)
        return source_dataset, target_dataset


def build_dataset(is_train, config):
    transform = build_transform(is_train, config)
    if is_train:
        source_root = os.path.join(config.DATA.DATA_PATH, config.DATA.SOURCE + '.txt')
        source_dataset = ObjectImage('', source_root, transform, return_path=config.RETURN_PATH)
        target_root = os.path.join(config.DATA.DATA_PATH, config.DATA.TARGET + '.txt')
        target_dataset = ObjectImage_mul('', target_root, transform)
        return source_dataset, target_dataset
    else:
        source_root = os.path.join(config.DATA.DATA_PATH, config.DATA.SOURCE + '.txt')
        source_dataset = ObjectImage('', source_root, transform, return_path=config.RETURN_PATH)
        target_root = os.path.join(config.DATA.DATA_PATH, config.DATA.TARGET + '.txt')
        target_dataset = ObjectImage('', target_root, transform, return_path=config.RETURN_PATH)
        return source_dataset, target_dataset


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    mean = config.DATA.MEAN
    std = config.DATA.STD
    if is_train:
        transform = transforms.Compose([
            transforms.Resize((config.DATA.IMG_SIZE + 32, config.DATA.IMG_SIZE + 32)),
            transforms.RandomCrop(config.DATA.IMG_SIZE),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    else:
        transform = transforms.Compose([
            transforms.Resize((config.DATA.IMG_SIZE + 32, config.DATA.IMG_SIZE + 32)),
            transforms.CenterCrop((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    return transform


class ResizeImage:
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))
