# --------------------------------------------------------
# Reference from Swin-Transformer https://github.com/microsoft/Swin-Transformer
# Reference from ATDOC https://github.com/tim-learn/ATDOC
# --------------------------------------------------------
from typing import List, Dict
import argparse
import datetime
import time
import random
import numpy as np
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from timm.utils import accuracy, AverageMeter
from tqdm import tqdm

from configs.config import get_config
from data import *
from models.model import *
from models.swin_transformer import SwinTransformer
from utils.util import *
try:
    from apex import amp
except ImportError:
    amp = None
import warnings
import time
#warnings.filterwarnings('ignore')

def inv_lr_scheduler(optimizer, iter_num, power=0.75, gamma=0.001, lr=0.001):
    lr = lr * (1 + gamma * iter_num) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
    return optimizer


def set_train(modellis:List[nn.Module]):
    for i in modellis:
        i.module.train()


def set_eval(modellis:List[nn.Module]):
    for i in modellis:
        i.module.eval()


def init(adapter, args, config):
    dsets, dset_loaders = build_tar_loader(config, args)
    loader = dset_loaders['target_train']
    banks = init_bank(len(dsets['target_train']), config, args)
    q = tqdm(total=len(loader)*2)
    cross_outputs = torch.randn(len(dsets['target_train']), 25, args.class_num)
    with torch.no_grad():
        for i, d in enumerate(loader):
            x, y, idx = d
            x = x.cuda()
            fea = adapter.module.backbone(x)
            bot_fea = adapter.module.bottleneck(fea)
            all_output = adapter.module.fc(bot_fea)
           # all_wei = adapter.module.get_wei(bot_fea)
            #all_features, all_output, all_wei = adapter(x, needs_wei=False)
            banks['domain_fea_bank'][idx] = fea.cpu()
            banks['domain_bot_bank'][idx] = bot_fea.cpu()
            banks['domain_output_bank'][idx] = all_output.cpu()
            #banks['wei_bank'][idx] = all_wei.cpu()
            banks['all_label'][idx] = y.float()

            cross_output_all = None
            for j in range(len(args.source_list)):
                cur_bot_fea = bot_fea[:,j,:]
                cross_output = adapter.module.cross_fc(cur_bot_fea) 
                cross_output_all = cross_output if cross_output_all is None else torch.concat([cross_output_all, cross_output], 1) # BN^2C
            cross_outputs[idx] = cross_output_all.cpu().float()

            q.update()

    adapter.eval()
    with torch.no_grad():
        for i, d in enumerate(loader):
            x, y, idx = d
            x = x.cuda()
            fea = adapter.module.backbone(x)

            banks['domain_fea_bank_eval'][idx] = fea.cpu()
            q.update()

    torch.save(banks, './my/{}_bank.pth'.format(args.target))


def set_weight_decay(path_adapter:nn.Module, config):
    param_group = []
    for k, v in path_adapter.named_parameters():
        if k.startswith(('source_bottlenecks')):
            param_group += [{'params':v, 'lr':config.TRAIN.BASE_LR * config.TRAIN.DECAY1, 'lr_mult':1, 'name':k}]
        elif k.startswith(('attn')):
            param_group += [{'params':v, 'lr':config.TRAIN.BASE_LR * config.TRAIN.DECAY2, 'lr_mult':1, 'name':k}]
        else:
            v.requires_grad = False
    return param_group


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str,
                        default="configs/swin_ens.yaml",
                        metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, default=64, help="batch size for single GPU")
    parser.add_argument('--devices', type=int, default=0, help="device IDs")
    parser.add_argument('--dataset', type=str, default='domainnet',
                        choices=['office31', 'office_home', 'VisDA', 'domainnet','office_caltech_10'], help='dataset used')
    parser.add_argument('--data-root-path', type=str, default='dataset/', help='path to dataset txt files')
    parser.add_argument('--source', type=str, default='clipart', help='source name', )
    parser.add_argument('--target', type=str, default='infograph', help='target name', )
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='my', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--alpha', type=float, default=.1, help='hyper-parameters alpha')
    parser.add_argument('--beta', type=float, default=.2, help='hyper-parameters beta')
    parser.add_argument('--log', default='log/', help='log path')
    parser.add_argument('--head_lr_ratio', type=float, default=3, help='hyper-parameters head lr')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--sourceOnly', action='store_true', default=False, help='Perform source training only')
    # distributed training
    parser.add_argument("--local_rank", nargs="+", type=int,
                        help='local rank for DistributedDataParallel', default=0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--class_num', default=345, type=int)
    parser.add_argument('--file', default='0', type=str)
    parser.add_argument('--lr', default=0, type=float)
    parser.add_argument('--decay1', default=0, type=float)
    parser.add_argument('--decay2', default=0, type=float)
    parser.add_argument('--sem_loss', default=0, type=float)
    parser.add_argument('--seed', default=996, type=int)
    parser.add_argument('--false', default=0, type=int)

    args, unparsed = parser.parse_known_args()

    if args.false and args.file != 'false':
        raise ValueError("args.false={} while args.file={}".format(args.false, args.file))

    # domainnet
    args.output += '/{}'.format(args.target)
    config = get_config(args)
    if args.dataset == 'domainnet':
        args.domain2idx, args.idx2domain = get_domain_idx(args.target)
    elif args.dataset == 'office_home':
        args.domain2idx, args.idx2domain = get_domain_idx_officehome(args.target)
    elif args.dataset == 'office_caltech_10':
        args.domain2idx, args.idx2domain = get_domain_idx_officecal(args.target)
    else:
        raise
    args.model_path = {
        i: './domainnet_src/{}'.format(i) for i in args.domain2idx
    } if args.dataset == 'domainnet' else {
        i: './source_trained/{}.pth'.format(i) for i in args.domain2idx
    }
    args.source_list = [i for i in args.domain2idx if i != args.target]
    return args, config


def main(config, args):
    config.defrost()
    config.TRAIN.MAX_ITER = 100000
    if config.MODEL.TYPE == "swin":
        config.MODEL.NUM_FEATURES = int(config.MODEL.SWIN.EMBED_DIM * 2 ** (len(config.MODEL.SWIN.DEPTHS) - 1))
        model_type = {
            'backbone': SwinTransformer,
            'bottleneck': feat_bottleneck,
            'clf': feat_classifier,
        }
        model_param = {
            'backbone': dict(patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32)),
            'bottleneck': dict(feature_dim=1024),
            'clf': dict(class_num=args.class_num)
        }
        model_load = {
            'backbone': load_swin_backbone,
            'bottleneck': load_swin_bottleneck,
            'clf': load_swin_myclf,
        }
    else:
        raise NotImplementedError()
    config.freeze()
    kwargs = {
        'model_type': model_type,
        'model_param': model_param,
        'model_load': model_load
    }
    logger.info(f"Creating base_model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    adapter = BiATEN(args, config, kwargs).cuda()

    logger.info("======================================")
    logger.info("target: " + config.DATA.TARGET)
    logger.info("======================================")

    parameters = set_weight_decay(adapter, config)
    optimizer = torch.optim.SGD(parameters, weight_decay=0, momentum=0.9)
    [adapter,], optimizer = amp.initialize([adapter,], optimizer, opt_level='O1')

    adapter = torch.nn.parallel.DistributedDataParallel(adapter, device_ids=[torch.cuda.current_device()], broadcast_buffers=False, find_unused_parameters=False)  

    logger.info("================>Start training....................")

    start_time = time.time()

    if not os.path.exists('./my/{}_bank.pth'.format(args.target)):
        init(adapter, args, config)
    logger.info("load bank from ./my/{}_bank.pth".format(args.target))
    banks = torch.load('./my/{}_bank.pth'.format(args.target))

    all_ps, all_dist, all_centroid = [], [], []
    for i in range(len(args.source_list)):
        aff = banks['domain_output_bank'][:,i,:].softmax(1)
        afea = banks['domain_bot_bank'][:,i,:]
        initc = torch.einsum('NC, NH -> CH', aff, afea)
        cent = (initc.T / (1e-8 + aff.sum(axis=0))).T
        all_centroid.append(cent.cpu())
    all_centroid = torch.stack(all_centroid, 1)     # 345, 5, 1024
    banks['domain_centroid'] = all_centroid
    all_centroid = banks['domain_centroid']
    current_bank = {
        'fea': banks['domain_bot_bank'],
        'out': torch.randn(len(banks['all_label']), len(args.source_list), args.class_num)
    }
    #dsets, dset_loaders = build_tar_loader(config, args)
    banks['cluster_label'] = torch.zeros_like(banks['all_label']).long()
    for epoch in range(args.epochs):
        if not args.false:
            flag = True if (epoch%2==0 and epoch>0) else False
        else:
            flag = False
        train_one_epoch(config, args, adapter, epoch, banks, all_centroid, optimizer, current_bank, flag=flag) 
        cluster_acc = (banks['cluster_label']==banks['all_label']).sum() / len(banks['all_label'])
        acc = val(config, args, adapter, banks, current_bank, flag=False)   
        logger.info("Epoch {}: acc={}".format(epoch, acc))
        all_centroid = []
        for i in range(len(args.source_list)):
            aff = current_bank['out'][:,i,:].softmax(1)
            afea = current_bank['fea'][:,i,:]
            initc = torch.einsum('NC, NH -> CH', aff, afea)
            cent = (initc.T / (1e-8 + aff.sum(axis=0))).T

            all_centroid.append(cent)
        all_centroid = torch.stack(all_centroid, 1)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def val(config, args, adapter, banks, current_bank, flag):
    set_eval([adapter, ])
    all_label = banks['all_label']
    fea_bank = banks['domain_fea_bank_eval']
    output_bank = banks['domain_output_bank']
    n = len(all_label)
    idx_gen = index_generator(n, args.batch_size*3, shuffle=False)
    all_wei = None
    domain_outputs_all = None
    with torch.no_grad():
        for i, idx in enumerate(idx_gen):
            fea = fea_bank[idx].cuda()
            bot_fea, origin_pred, wei = adapter(fea, flag)

            origin_pred_deta = origin_pred.detach().cpu()
            current_bank['out'][idx] = origin_pred_deta
            current_bank['fea'][idx] = bot_fea.detach().cpu()
            domain_outputs_all = origin_pred_deta if domain_outputs_all is None else torch.concat([domain_outputs_all, origin_pred_deta], 0)

            all_wei = wei if all_wei is None else torch.cat([all_wei, wei])
            out = torch.einsum('BNC, BN -> BC', origin_pred, wei)
            banks['output_bank'][idx] = out.cpu()
    acc = (banks['output_bank'].max(1)[1]==all_label).sum() / n
    set_train([adapter, ])
    return acc


def cal_dis(aff, all_fea):
    initc = torch.einsum('NC, NH -> CH', aff, all_fea)
    initc = (initc.T / (1e-8 + aff.sum(axis=0))).T  # 345,1024, centroid
    res1 = []
    dist1 = []
    for i in range(all_fea.shape[0]):
        dis = torch.nn.functional.cosine_similarity(all_fea[i], initc)
        pos = dis.argmax(0)
        res1.append(pos)
        dist1.append(dis)
    dist1 = torch.stack(dist1)
    pred1 = torch.stack(res1)
    return dist1, pred1, initc


def clustering(aff, all_fea, K, all_label):
    dist1, pred1, c1 = cal_dis(aff, all_fea)
    aff2 = torch.eye(K)[pred1]
    dist2, pred2, c2 = cal_dis(aff2, all_fea)
    acc1 = (aff.max(-1)[1]==all_label).sum() / len(all_label)
    acc2 = (pred2==all_label).sum() / len(all_label)
    print("{:.4f} -> {:.4f}".format(acc1.item(), acc2.item()))
    return dist2, pred2, c2


def init_bank(num_samples, config, args,):
    logger.info("Pre-computing features for domain '{}', this is a one-time operation.".format(args.target))
    banks = {
        'fea_bank': torch.randn(num_samples, config.MODEL.ATTN.FEATURE_DIM),
        'wei_bank': torch.randn(num_samples, len(args.source_list)),
        'domain_fea_bank': torch.randn(num_samples, len(args.source_list), 1024),
        'domain_fea_bank_eval': torch.randn(num_samples, len(args.source_list), 1024),
        'domain_bot_bank': torch.randn(num_samples, len(args.source_list), config.MODEL.ATTN.FEATURE_DIM),
        'output_bank': torch.randn(num_samples, args.class_num),
        'all_label': torch.randn(num_samples),
        'domain_output_bank': torch.randn(num_samples, len(args.source_list), args.class_num)
    }
    return banks


def index_generator(N, B, shuffle=True):
    lis = list(range(N))
    if shuffle:
        random.shuffle(lis)
    num_steps = N//B + 1 if N%B!=0 else N//B
    start, end = 0, B
    for i in range(num_steps):
        batch = lis[start:end]
        yield batch
        start = end
        end += B
        end = min(end, N)


def train_one_epoch(config, args, adapter, epoch, banks, all_centroid:torch.Tensor, optimizer:torch.optim.Optimizer, current_bank, flag):
    # all_centroid: NCL
    set_train([adapter])
    optimizer.zero_grad()
    loss_meter_ps = AverageMeter()
    num_samples = len(banks['all_label'])
    num_steps = num_samples // args.batch_size
    idx_gen = index_generator(num_samples, args.batch_size)
    fea_bank = banks['domain_fea_bank']
    all_centroid = all_centroid.cuda()
    for idx, img_idx in enumerate(tqdm(idx_gen, total=num_steps, mininterval=.5)):
        idx_step = epoch * num_steps + idx
        optimizer = inv_lr_scheduler(optimizer, idx_step, lr=config.TRAIN.BASE_LR)
        if len(img_idx)==1:
            continue
        fea = fea_bank[img_idx].cuda()
        cluster_fea = current_bank['fea'][img_idx].cuda()   # B,25,256
        loss, cluster_pred = adapter(fea, flag, cluster_fea, all_centroid)
        banks['cluster_label'][img_idx] = cluster_pred.cpu()

        optimizer.zero_grad()

        if config.AMP_OPT_LEVEL != "O0":
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            if config.TRAIN.CLIP_GRAD:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
        else:
            raise

        optimizer.step()

        torch.cuda.synchronize()
        loss_meter_ps.update(loss.item(), fea.size(0))


    if dist.get_rank() == 0:
        lr = optimizer.param_groups[0]['lr']
        logger.info(f'lr: {lr:.7f}\t'
                    f'loss_ps: {loss_meter_ps.avg:.3f}\t')    


if __name__ == '__main__':
    args, config = parse_option()
    import torch.multiprocessing as mp
    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"
        
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
        
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(args.devices)
    dist.init_process_group(backend='nccl',rank=0, world_size=1)
    dist.barrier()
    seed = config.SEED 
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}", file=args.file)
    args.logger = logger
    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config_{}.yaml".format(args.file))
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")
    logger.info(config.dump())
    main(config, args)

