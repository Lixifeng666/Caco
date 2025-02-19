import time
import numpy as np
import torch
import logging
# from dgl.contrib.sampling.sampler import NeighborSampler
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import  average_precision_score, roc_auc_score
import os

def train(args, logger, data, model, path):
    checkpoints_path = path

    # use optimizer AdamW
    logger.info('Start training')
    logger.info(
        f'dropout:{args.dropout},seed:{args.seed},lr:{args.lr},self-loop:{args.self_loop},norm:{args.norm}')

    logger.info(
        f'n-epochs:{args.n_epochs}, n-hidden:{args.n_hidden},n-layers:{args.n_layers},weight-decay:{args.weight_decay}')

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    data_center = init_center(args, data, model)
    if args.gpu < 0:
        adj = data['g'].adjacency_matrix().to_dense().cpu()
        adj_cnn = data['g_cnn'].adjacency_matrix().to_dense().cpu()

    else:
        adj = data['g'].adjacency_matrix().to_dense().cuda()
        adj_cnn = data['g_cnn'].adjacency_matrix().to_dense().cuda()

    model.train()
    # 创立矩阵以存储结果曲线
    arr_epoch = np.arange(args.n_epochs)
    arr_loss = np.zeros(args.n_epochs)
    arr_valauc = np.zeros(args.n_epochs)
    arr_testauc = np.zeros(args.n_epochs)
    savedir = './embeddings/' +args.module+'/'+ args.dataset + '/{}'.format(args.abclass)
    scoredir = './score/' +args.module+'/'+ args.dataset + '/{}'.format(args.abclass)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    if not os.path.exists(scoredir):
        os.makedirs(scoredir)
    max_valauc = 0
    epoch_max = 0
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        t0 = time.time()

        z1, z2, A_mean,S_mean,A_std,S_std  = model(data['g'], data['g_cnn'], data['features'],data['add_features'])

        # loss = loss_func(args,z1,z2,A_mean,S_mean,A_std,S_std,data['train_mask'])
        # loss = hyper_sphere_loss(z1, z2, A_mean, S_mean, A_std, S_std, r=1.0)

        # loss = cosine_sphere_loss(args, z1, z2, A_mean, S_mean, A_std, S_std, data_center,data['train_mask'])
        loss = cosine_kl_loss(args, z1, z2, A_mean, S_mean, A_std, S_std, data['train_mask'])
        # 保存训练loss
        arr_loss[epoch] = loss.item()
        #
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur = time.time() - t0
        auc, ap, val_score = fixed_graph_evaluate(args, model, data,data['val_mask'])
        arr_valauc[epoch] = auc
        if auc > max_valauc:
            max_valauc = auc
            epoch_max = epoch
            torch.save(model.state_dict(), checkpoints_path)
            test_auc, test_ap, test_loss= fixed_graph_evaluate(args, model, data, data['test_mask'])
        print(
            "Epoch {:05d} | Time(s) {:.4f} | Train Loss {:.4f}  | Val AUROC {:.4f}  |test AUROC {:.4f}  | "
            "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
                                          auc, test_auc, data['n_edges'] / np.mean(dur) / 1000))

    if args.early_stop:
        print('loading model before testing.')
        model.load_state_dict(torch.load(checkpoints_path))

        # if epoch%100 == 0:
    model.load_state_dict(torch.load(checkpoints_path))
    auc, ap, scores = fixed_graph_evaluate(args, model, data, data['test_mask'])
    test_dur = 0
    arr_testauc[epoch] = auc
    best_epoch = epoch_max
    print("Test Time {:.4f} | Test AUROC {:.4f} | Test AUPRC {:.4f}".format(test_dur, auc, ap))

    return auc, ap, best_epoch

def init_center(args,data,model, eps=0.001):
    """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    if args.gpu<0 :
        c = torch.zeros(args.n_hidden)
    else:
        c = torch.zeros(args.n_hidden, device=f'cuda:{args.gpu}')

    model.eval()
    with torch.no_grad():
        Z = model.get_embeddings(data['g'],data['features'])
        n_samples = Z.shape[0]
        c =torch.sum(Z, dim=0)

    c /= n_samples

    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps
    return c
def cosine_sphere_loss(args, z1, z2, A_mean, S_mean, A_std, S_std, data_center,mask):
    z1 = z1[mask]
    z2 = z2[mask]

    # 计算 z1 和 z2 在超球面上的欧氏距离
    r = data_center  # 可根据实际需要调整超球面半径
    dist = torch.norm(r * torch.nn.functional.normalize(z1) - r * torch.nn.functional.normalize(z2), dim=1)
    dist_loss = dist.mean()

    # 计算余弦相似度
    cos_similarity = torch.nn.functional.cosine_similarity(z1, z2)

    # 余弦相似度负向的均值作为损失，即最大化相似度（最小化负相似度）
    loss_sim = -cos_similarity.mean()

    # 计算KL散度
    kl_a = kl(A_mean, A_std)
    kl_s = kl(S_mean, S_std)

    # 最终的损失是 dist_loss，loss_sim 和 KL散度的线性组合
    loss = dist_loss + args.lamda * loss_sim + kl_a + kl_s

    return loss
def hyper_sphere_loss(z1, z2, A_mean, S_mean, A_std, S_std, r=1.0):
    # 计算 z1 和 z2 在超球面上的欧氏距离
    dist = torch.norm(r*torch.nn.functional.normalize(z1) - r*torch.nn.functional.normalize(z2), dim=1)

    # 对 z1 和 z2 使用损失函数得到的结果求平均值
    sphere_loss = dist.mean()

    # 计算 KL 散度
    kl_a = kl(A_mean, A_std)
    kl_s = kl(S_mean, S_std)

    # 最终的损失是 sphere_loss 和 KL 散度的线性组合
    loss = sphere_loss + kl_a + kl_s

    return loss

def loss_func(args,z1, z2, A_mean,S_mean,A_std,S_std, mask):
    z1 = z1[mask]
    z2 = z2[mask]
    c_ = torch.mm(z1.T, z2)
    c1 = torch.mm(z1.T, z1)
    c2 = torch.mm(z2.T, z2)
    c = -torch.diagonal(c_)
    loss_inv = c.mean()
    if z1.is_cuda:
        iden = torch.as_tensor(torch.eye(c.shape[0])).cuda()
    else:
        iden = torch.as_tensor(torch.eye(c.shape[0]))
    loss_dec1 = (iden - c1).pow(2).mean()
    loss_dec2 = (iden - c2).pow(2).mean()
    kl_a = kl(A_mean,A_std)
    kl_s = kl(S_mean,S_std)
    loss = loss_inv + args.lamda*(loss_dec1 + loss_dec2) + (kl_a+kl_s)
    return loss
def anomaly_score(z1,z2,mask):

    scores = F.mse_loss(z1[mask], z2[mask], reduction='none')
    return scores.mean(1)
def kl(mean,std):

    kl = (0.5 / mean.size(0)) * torch.mean(torch.sum(1 + 2 * std - torch.pow(mean,2) - torch.pow(torch.exp(std),2), 1))

    return -kl

def fixed_graph_evaluate(args, model, data, mask):
    model.eval()
    with torch.no_grad():
        labels = data['labels'][mask]
        z1, z2, A_mean,S_mean,A_std,S_std,= model(data['g'], data['g_cnn'], data['features'],data['add_features'])
        loss = loss_func(args,z1, z2,A_mean,S_mean,A_std,S_std,mask)
        scores = anomaly_score(z1,z2,mask)
        labels = labels.cpu().numpy()
        scores = scores.cpu().numpy()
        auc = roc_auc_score(labels, scores)
        ap = average_precision_score(labels, scores)

    return auc, ap, scores

def cosine_kl_loss(args, z1, z2, A_mean, S_mean, A_std, S_std, mask):
    z1 = z1[mask]
    z2 = z2[mask]

    # calculate cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(z1, z2, dim=1)
    similarity_loss = 1 - cos_sim.mean() # if you want to minimize 1- cosine similarity
    # similarity_loss = - cos_sim.mean() # if you want to maximize cosine similarity

    # calculate Euclidean distance
    dist = torch.nn.functional.pairwise_distance(z1, z2)
    dist_loss = dist.mean()

    # calculate KL divergence
    kl_a = kl(A_mean, A_std)
    kl_s = kl(S_mean, S_std)

    # the final loss is a combination of similarity_loss, dist_loss and KL divergences
    loss = similarity_loss + args.lamda2 * dist_loss + args.lamda3 * (kl_a + kl_s)
    # loss = args.lamda2 * dist_loss + args.lamda3 * (kl_a + kl_s)
    # loss = similarity_loss

    return loss
