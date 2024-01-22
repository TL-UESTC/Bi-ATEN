import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import math
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weightNorm

resnet_dict = {
    "ResNet18": models.resnet18, 
    "ResNet34": models.resnet34,
    "ResNet50": models.resnet50, 
    "ResNet101": models.resnet101, 
    "ResNet152": models.resnet152
}


class BiATEN(nn.Module):
    def __init__(self, args, config, kwargs) -> None:
        super().__init__()
        self.config = config
        self.hidden_dim = config.MODEL.ATTN.HIDDEN_DIM
        self.num_classes = args.class_num
        self.args = args
        self.heads = config.MODEL.ATTN.HEADS
        self.feature_dim = config.MODEL.ATTN.FEATURE_DIM
        self.source_backbones = [0]*len(args.source_list)
        self.source_clfs = [0]*len(args.source_list)
        self.source_bottlenecks = [0]*len(args.source_list)
        self.model_count = len(args.source_list)
        self.attn_method = config.MODEL.DIS
        self.attn = BiAttn(fea_dim=256, sem_dim=args.class_num, hid_dim=self.hidden_dim, domain_num=self.model_count, n_head=self.heads)
        for i in args.source_list:
            backbone = kwargs['model_type']['backbone']
            backbone_param = kwargs['model_param']['backbone']
            clf = kwargs['model_type']['clf']
            clf_param = kwargs['model_param']['clf']
            bottleneck = kwargs['model_type']['bottleneck']
            bottleneck_param = kwargs['model_param']['bottleneck']
            self.source_backbones[args.domain2idx[i]] = backbone(**backbone_param)
            self.source_clfs[args.domain2idx[i]] = clf(**clf_param)
            self.source_bottlenecks[args.domain2idx[i]] = bottleneck(**bottleneck_param)
        backbone_init_func = kwargs['model_load']['backbone']
        clf_init_func = kwargs['model_load']['clf']
        bottleneck_init_func = kwargs['model_load']['bottleneck']
        for i in range(len(self.source_backbones)):
            if self.source_backbones[i] != 0:
                model_path = args.model_path[args.idx2domain[i]]
                backbone_init_func(self.source_backbones[i], model_path)
                clf_init_func(self.source_clfs[i], model_path)
                bottleneck_init_func(self.source_bottlenecks[i], model_path)
        self.source_backbones = nn.ModuleList(self.source_backbones)
        self.source_clfs = nn.ModuleList(self.source_clfs)
        self.source_bottlenecks = nn.ModuleList(self.source_bottlenecks)

    def cross_fc(self, fea):
        assert fea.shape[1]==256
        outputs = []
        for i in range(self.model_count):
            net = self.source_clfs[i]
            cur = net(fea)
            outputs.append(cur)
        allout = torch.stack(outputs, 1)
        return allout.float()

    def backbone(self, x):
        all_features = []
        for j in range(self.model_count):
            fea = self.source_backbones[j].forward_features(x)
            all_features.append(fea)
        all_features = torch.stack(all_features, 1)    # B, 5, 1024
        return all_features.float()

    def bottleneck(self, fea):
        all_bot = []
        for j in range(self.model_count):
            cur = fea[:,j,:]
            bot_fea = self.source_bottlenecks[j](cur)
            all_bot.append(bot_fea)
        all_bot = torch.stack(all_bot, 1)
        return all_bot.float()

    def fc(self, x):
        allout = []
        for i in range(self.model_count):
            cur = x[:,i,:]
            net = self.source_clfs[i]
            cur = net(cur)
            allout.append(cur)
        allout = torch.stack(allout, 1)
        return allout.float()

    def forward(self, x, use_attn, cluster_fea=None, all_centroid=None):
        bot_fea = self.bottleneck(x)    # B,5,256
        cross_output_all = []         
        for j in range(self.model_count):
            cur_bot_fea = bot_fea[:,j,:]
            cross_output = self.cross_fc(cur_bot_fea) 
            cross_output_all.append(cross_output)
        cross_output_all = torch.stack(cross_output_all, 1)     # B,5,5,345
        self.origin_sem, self.ens_sem, self.wei, self.sem_loss = self.attn(bot_fea, cross_output_all)
        origin_pred = self.origin_sem if not use_attn else self.ens_sem
        if not self.training:
            return bot_fea, origin_pred, self.wei
        else:
            assert (cluster_fea is not None) and (all_centroid is not None) 
            wei = self.wei
            out = torch.einsum('BNC, BN -> BC', origin_pred, wei)
            mix_feas = torch.einsum('BNL, BN -> BL', cluster_fea, wei) 
            initc = torch.einsum('CNL, BN -> BCL', all_centroid, wei)
            pred = []
            for i in range(mix_feas.shape[0]):
                val, dd = torch.cdist(mix_feas[i].unsqueeze(0), initc[i]).min(-1)
                pred.append(dd.long())
            pred = torch.tensor(pred).long().cuda()
            cross_ent = self.args.alpha * nn.CrossEntropyLoss(label_smoothing=0.1)(out, pred)
            imloss = ent_loss(out)
            loss = cross_ent*self.args.alpha + self.sem_loss*self.args.beta + imloss
            return loss, pred 


def compute_attn(q_emb, k_emb, dis='cosine'):
    assert q_emb.shape[-1] == k_emb.shape[-1]
    B, N, L = k_emb.shape
    if dis == "cosine":
        raw_score = F.cosine_similarity(q_emb.view(B, 1, -1), k_emb, dim=-1)
    elif dis == "dotproduct":
        raw_score = torch.sum(q_emb.view(B, 1, -1) * k_emb, dim=-1) / (math.sqrt(L)) 
    else:
        raise ValueError('invalid att type: {}'.format(dis))
    score = raw_score.softmax(1)
    return score


class Head(nn.Module):
    # Head
    def __init__(self, key_dim, query_dim, hid_dim, temp=1, att="cosine"):
        super(Head, self).__init__()
        self.linear_q = nn.Linear(query_dim, hid_dim)    # 256*args.cnt, hid
        self.linear_k = nn.Linear(key_dim, hid_dim)      # 256, hid
        self.temp = temp
        self.att = att
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)

    def forward(self, cat_proto):
        n_class, n_extractors, fea_dim = cat_proto.shape    # B, args.cnt, 256
        q = cat_proto.reshape(n_class, -1)    # B, args.cnt*256
        k = cat_proto                         # B, args.cnt, 256
        q_emb = self.linear_q(q)              # B, hid
        k_emb = self.linear_k(k)              # B, args.cnt, hid
        if self.att == "cosine":
            raw_score = F.cosine_similarity(q_emb.view(n_class, 1, -1), k_emb.view(n_class, n_extractors, -1), dim=-1)
        elif self.att == "dotproduct":
            raw_score = torch.sum(q_emb.view(n_class, 1, -1) * k_emb.view(n_class, n_extractors, -1), dim=-1) / (math.sqrt(fea_dim)) 
        else:
            raise ValueError('invalid att type: {}'.format(self.att))
        score = F.softmax(self.temp * raw_score, dim=1)
        return score


class SemHead(nn.Module):
    # Head
    def __init__(self, key_dim, query_dim, hid_dim, temp=1, att="cosine"):
        super(SemHead, self).__init__()
        self.linear_q = nn.Linear(query_dim, hid_dim)    # 256, hid
        self.linear_k = nn.Linear(key_dim, hid_dim)      # 345, hid
        self.temp = temp
        self.att = att
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.005)

    def forward(self, fea, sem):
        q_emb = self.linear_q(fea)              # B, hid
        k_emb = self.linear_k(sem)              # B, 5, hid
        score = compute_attn(q_emb, k_emb, dis=self.att)
        return score, q_emb


class SemBlock(nn.Module):
    def __init__(self, fea_dim, sem_dim, hid_dim, att="cosine", n_head=4) -> None:
        super().__init__()
        self.dis = att
        layers = []
        for i in range(n_head):
            layer = SemHead(sem_dim, fea_dim, hid_dim, att=att)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)
    
    def forward(self, fea, sem, d):
        # fea: B,256
        # sem: B,5,345
        fea_deta = fea.detach()
        sem_deta = sem.detach()
        attns, feas = [], []
        for i in self.layers:
            attn, fea_emb = i(fea_deta, sem_deta)
            attns.append(attn)
            feas.append(fea_emb)
        attns = torch.stack(attns, -1).mean(-1)     # B, 5
        feas = torch.stack(feas, 1)    # B, nhead, hid
        ens_sem = torch.einsum('BNC, BN -> BC', sem_deta, attns) 
        return attns, feas, ens_sem


class BiAttn(nn.Module):
    def __init__(self, fea_dim, sem_dim, hid_dim, domain_num, sem_att="cosine", fea_att="cosine", n_head=4):
        super(BiAttn, self).__init__()
        self.domain_num = domain_num
        self.n_head = n_head
        self.fea_att = fea_att
        sem_lis, fea_as_q_lis = [], []
        for _ in range(domain_num):
            layer = SemBlock(fea_dim, sem_dim, hid_dim, sem_att, n_head)
            sem_lis.append(layer)
        for _ in range(n_head):
            fea_as_q = nn.Linear(fea_dim*domain_num, hid_dim)
            fea_as_q_lis.append(fea_as_q)
        self.sem_blocks = nn.ModuleList(sem_lis)
        self.fea_as_q_lis = nn.ModuleList(fea_as_q_lis)

    def forward(self, domain_fea, domain_sem):
        fea_embs, ens_sems, origin_sems = [], [], []
        sem_loss = 0
        for d in range(self.domain_num):
            fea = domain_fea[:,d,:]     # B, 256
            sem = domain_sem[:,d,:,:]   # B, 5, 345
            net = self.sem_blocks[d]
            sem_attn, fea_emb, ens_sem = net(fea, sem, d)
            fea_embs.append(fea_emb)
            ens_sems.append(ens_sem)            
            origin_sems.append(sem[:,d,:]) 
            sem_loss += ent_loss(ens_sem)
            sem_loss += ent_loss(sem[:,d,:])
        ens_sems = torch.stack(ens_sems, 1)           # B, 5, 345
        origin_sems = torch.stack(origin_sems, 1)     # B, 5, 345
        fea_keys = torch.stack(fea_embs, 1)           # B, 5, n_head, hid
        
        fea_q_in = domain_fea.detach().reshape(domain_fea.shape[0], -1)
        fea_qs = []
        for i in self.fea_as_q_lis:
            fea_emb = i(fea_q_in)
            fea_qs.append(fea_emb)
        fea_qs = torch.stack(fea_qs, 1)     # B, n_head, hid 
        q_attns = []
        for i in range(self.n_head):
            fea_key = fea_keys[:,:,i,:]     # B, 5, hid
            fea_q = fea_qs[:,i,:]           # B, hid
            attn = compute_attn(fea_q, fea_key, dis=self.fea_att)
            q_attns.append(attn)
        q_attns = torch.stack(q_attns, -1).mean(-1)     # B, 5

        return origin_sems, ens_sems, q_attns, sem_loss


def entropy(input):
    epsilon = 1e-5
    entropy = -input * torch.log(input + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy.mean()


def gentropy(softmax_out):
    epsilon = 1e-5
    msoftmax = softmax_out.mean(dim=0)
    gentropy = -msoftmax * torch.log(msoftmax + epsilon)
    return torch.sum(gentropy)


def ent_loss(out):
    softmax_out = nn.Softmax(dim=1)(out)
    entropy_loss = entropy(softmax_out) - gentropy(softmax_out)
    return entropy_loss


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class feat_bottleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="bn"):
        super(feat_bottleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        return x


class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="wn"):
        super(feat_classifier, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x


if __name__ == '__main__':
    pass