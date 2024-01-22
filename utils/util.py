import torch
import os
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
from utils.folder import ImageFolder_ind, DomainFolder
import torchvision.transforms as transforms
import logging
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import cdist
import sys
import functools
from termcolor import colored
data_transforms = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)
Domain2Idx = {
    'clipart': 0,
    'infograph': 1,
    'painting': 2,
    'quickdraw': 3,
    'real': 4,
    'sketch': 5,
}
Idx2Domain = {
    0: 'clipart',
    1: 'infograph',
    2: 'painting',
    3: 'quickdraw',
    4: 'real',
    5: 'sketch',
}

Domain2Idx_officehome = {
    'Art': 0,
    'Clipart': 1,
    'Product': 2,
    'Real_World': 3,
}
Idx2Domain_officehome = {
    0: 'Art',
    1: 'Clipart',
    2: 'Product',
    3: 'Real_World',
}

Domain2Idx_cal = {
    'amazon': 0,
    'caltech': 1,
    'dslr': 2,
    'webcam': 3,
}
Idx2Domain_cal = {
    0: 'amazon',
    1: 'caltech',
    2: 'dslr',
    3: 'webcam',
}


@functools.lru_cache()
def create_logger(output_dir, dist_rank=0, name='', file='1'):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(os.path.join(output_dir, f'log_rank{dist_rank}_{file}.txt'), mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger


@functools.lru_cache()
def create_logger_parallel(output_dir, name=''):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(console_handler)
    
    # create file handlers
    file_handler = logging.FileHandler(os.path.join(output_dir, f'.txt'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)
    return logger


class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch


class InfiniteDataLoader:
    def __init__(self, dataset, batch_size, num_workers):
        super().__init__()
        sampler = RandomSampler(dataset, replacement=True)
        batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=True)

        self._infinite_iterator = iter(DataLoader(dataset, num_workers=num_workers, batch_sampler=_InfiniteSampler(batch_sampler)))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        raise ValueError


def load_checkpoint(model, checkpoint_path, logger=None):
    if logger != None:
        logger.info(f"==============> Resuming form {checkpoint_path}....................")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_dict = model.state_dict()
    pretrained_dict = checkpoint['model']
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    msg = model.load_state_dict(pretrained_dict, strict=False)
    if logger != None:
        logger.info(msg)


def get_domain_idx(target):
    '''return: domain2idx, idx2domain'''
    target_idx = Domain2Idx[target]
    domain2idx = {i:Domain2Idx[i] for i in Domain2Idx if i != target}
    for i in domain2idx:
        if domain2idx[i] > target_idx:
            domain2idx[i] -= 1
    idx2domain = [0]*(len(Idx2Domain)-1)
    for i in domain2idx:
        idx2domain[domain2idx[i]] = i
    return domain2idx, idx2domain


def get_domain_idx_officehome(target):
    '''return: domain2idx, idx2domain'''
    target_idx = Domain2Idx_officehome[target]
    domain2idx = {i:Domain2Idx_officehome[i] for i in Domain2Idx_officehome if i != target}
    for i in domain2idx:
        if domain2idx[i] > target_idx:
            domain2idx[i] -= 1
    idx2domain = [0]*(len(Idx2Domain_officehome)-1)
    for i in domain2idx:
        idx2domain[domain2idx[i]] = i
    return domain2idx, idx2domain


def get_domain_idx_officecal(target):
    '''return: domain2idx, idx2domain'''
    target_idx = Domain2Idx_cal[target]
    domain2idx = {i:Domain2Idx_cal[i] for i in Domain2Idx_cal if i != target}
    for i in domain2idx:
        if domain2idx[i] > target_idx:
            domain2idx[i] -= 1
    idx2domain = [0]*(len(Idx2Domain_cal)-1)
    for i in domain2idx:
        idx2domain[domain2idx[i]] = i
    return domain2idx, idx2domain


def get_mix_loader(args):
    loader_lis = []
    for i in args.base_domains:
        dset = ImageFolder_ind(args.data_root+args.idx2domain[i], transform=data_transforms, target_idx=None)
        loader = InfiniteDataLoader(dset, batch_size=args.batch_size, num_workers=8)
        loader_lis.append(loader)
    return zip(*loader_lis)


def print_args(args):
    formatstr = 'Args:\n'
    for k, v in sorted(vars(args).items()):
        formatstr += '{}: {}\n'.format(k, v)
    logging.info(formatstr)


def obtain_label(loader, netF, netB, netC, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count>args.threshold)
    labelset = labelset[0]
    # print(labelset)

    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

    logging.info(log_str)

    return pred_label.astype('int')


def load_swin_sourceonly(model:nn.Module, path):
    ckp = torch.load(path)
    print("Loading from {}".format(path))
    model.load_state_dict(ckp['model'], strict=False)


def load_swin_backbone(model, path):
    print("Loading backbone from {}".format(path))
    ckp = torch.load(path+'_backbone.pth', map_location='cuda:0')
    model.load_state_dict(ckp, strict=False)
    model.cuda().eval()


def load_swin_bottleneck(model, path):
    print("Loading bottleneck from {}".format(path))
    ckp = torch.load(path+'_bottleneck.pth', map_location='cuda:0')
    model.load_state_dict(ckp,)
    model.cuda().eval()


def load_swin_myclf(model, path):
    print("Loading myclf from {}".format(path))
    ckp = torch.load(path+'_myclf.pth', map_location='cuda:0')
    model.load_state_dict(ckp,)
    model.cuda().eval()


def load_swin_clf(model, path):
    print("Loading classifier from {}".format(path))
    ckp = torch.load(path+'_clf.pth')
    newckp = {}
    for k in ckp.keys():
        newk = k.split('my_fc_head.')[-1]
        newckp[newk] = ckp[k]
    model.load_state_dict(newckp)
    model.cuda().eval()


def load_resnet(model, path):
    ckp = torch.load(path)
    print("Loading from {}".format(path))
    model.load_state_dict(ckp)
    model.cuda().eval()


def load_ens_model(model:nn.Module, path, domain):
    ckp = torch.load(path)['model']
    backbone_dict = {}
    fc_dict = {}
    for k, v in model.named_parameters():
        if k.startswith('my_fc_head'):
            #newk = k.replace('my_fc_head','my_fc')
            newk = k
            fc_dict[newk] = ckp[k]
        if k.startswith('backbone'):
            newk = k.split('backbone.')[-1]
            if k in ckp:
                backbone_dict[newk] = ckp[k]
    os.makedirs('./domainnet_src/', exist_ok=True)
    torch.save(backbone_dict, './domainnet_src/{}_backbone.pth'.format(domain))
    torch.save(fc_dict, './domainnet_src/{}_clf.pth'.format(domain))

    model.load_state_dict(ckp, strict=False)
    print("Resume from {}".format(path))