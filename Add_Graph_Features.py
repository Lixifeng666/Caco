import torch
import dgl
import dgl.function as fn


def add_graph_features(g, features):
    """
    在DGL图和特征基础上添加新的图特征。
    Args:
        g: DGLGraph对象，图结构
        features: tensor，原始节点特征
    Returns:
        new_features: tensor，增强的节点特很
    """
    g = g.local_var()  # 使用local_var确保不修改原图

    # 1. 在每个节点的邻居节点集合中，将每个邻居节点及其邻居节点特征聚合平均
    g.ndata['h'] = features
    g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h_neigh'))
    neigh_features = g.ndata['h_neigh']

    # 2. 在每个节点的邻居节点集合N(v)中，将更新后的特征聚合平均
    g.ndata['h'] = neigh_features
    g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h_neigh_neigh'))
    h_neigh_neigh = g.ndata['h_neigh_neigh']

    # 3. 结合原始特征和新计算的图特征
    new_features = torch.cat((features, h_neigh_neigh), dim=1)

    return new_features