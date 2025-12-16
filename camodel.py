# -*- coding: utf-8 -*-
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv, GATConv, SAGEConv, GATv2Conv
import pandas as pd


class GenerativeGraph(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.2,types =""):
        super(GenerativeGraph, self).__init__()
        self.types = types
        if types == "drug":
            self.l = nn.Linear(768,input_size)
        self.input_size = input_size
        self.Gencoder = nn.ModuleList()
        self.Gencoder.append(GATConv(input_size, output_size, attn_drop=dropout, residual=False, feat_drop=dropout,
                                     num_heads=1, allow_zero_in_degree=True, activation=F.elu))

    def forward(self, embedding,text_emb =None):
        if self.types == "drug":
            tep = self.l(text_emb.to(embedding.device)).unsqueeze(1)
            embedding = torch.cat((embedding,tep),dim=1)
        _graph = self.constructure_graph(embedding)
        _shape = embedding.shape
        embedding = embedding.reshape(-1, self.input_size)
        for layer in self.Gencoder:
            embedding = layer(_graph, embedding).squeeze()
        embedding = embedding.reshape(_shape[0], _shape[1], -1)
        embedding = torch.mean(embedding, dim=1)

        return embedding

    def constructure_graph(self, embedding):
        adjacency = embedding @ embedding.permute(0, 2, 1)
        edge = []
        for i, d in enumerate(adjacency):
            d = (d > torch.mean(d)).to(dtype=torch.int) + torch.eye(d.shape[0]).to(d.device)
            edge.append(d.nonzero() + i * len(d))
        edge = torch.cat(edge, dim=0).permute(1, 0)
        _graph = dgl.graph(data=(edge[0], edge[1]))
        del edge, adjacency
        return _graph


class HGenerativeGraphFusion(nn.Module):
    def __init__(self, meta_paths, inpsize, oupsize, layer_num_heads=1, dropout=0.2,types = ""):
        super(HGenerativeGraphFusion, self).__init__()
        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            self.gat_layers.append(GraphConv(inpsize, oupsize, activation=F.elu, norm="both"))
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)
        self.graph_type = "heterogeneous" if len(meta_paths) > 1 else "homogeneous"
        if self.graph_type == "heterogeneous":
            self.generative_graph = GenerativeGraph(oupsize * layer_num_heads, oupsize,types = types)
        else:
            self.trans = nn.Sequential(nn.Linear(oupsize * layer_num_heads, oupsize), )

    def forward(self, g, h,text_emb=None):
        semantic_embeddings = []
        for i, paths in enumerate(self.meta_paths):
            semantic_embeddings.append(self.gat_layers[i](g[paths], h))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1).flatten(2)
        if self.graph_type == "heterogeneous":
            return self.generative_graph(semantic_embeddings,text_emb = text_emb)
        else:
            return  self.trans(semantic_embeddings).squeeze()



class Classifier(nn.Module):
    def __init__(self, nfeat):
        super(Classifier, self).__init__()
        self.L1 = nn.Linear(nfeat, nfeat * 2)
        self.L2 =  nn.Linear(nfeat * 2, 2)

    def forward(self, x,e = 0):
        out = nn.ELU()(self.L1(x))
        out = self.L2(out)
        return nn.Softmax(dim = 1)(out)


class CE_DTI(nn.Module):
    def __init__(self, all_meta_paths, in_size, hidden_size, out_size, dropout=0.2, layersnums=1, att_heads=1):
        super(CE_DTI, self).__init__()
        self.sum_layers = nn.ModuleList()
        for i in range(len(all_meta_paths)):
            if i == 0:
                types= "0"
            else:
                types = ""
            self.sum_layers.append(
                HGenerativeGraphFusion(all_meta_paths[i], in_size, hidden_size, 1, dropout,types=types))
        self.encoder = nn.ModuleList()
        if layersnums >= 2:
            self.encoder.append(
                GATConv(hidden_size * 2, hidden_size * 2, attn_drop=dropout, residual=True, feat_drop=dropout,
                        num_heads=1, activation=F.elu,
                        allow_zero_in_degree=True))
            for i in range(layersnums - 2):
                self.encoder.append(
                    GATConv(hidden_size * 2, hidden_size * 2, attn_drop=dropout, residual=True,
                            feat_drop=dropout,
                            num_heads=att_heads, activation=F.elu,
                            allow_zero_in_degree=True))
            self.encoder.append(
                GATConv(hidden_size * 2, out_size * 2, attn_drop=dropout, residual=True, feat_drop=dropout,
                        num_heads=1, activation=None,
                        allow_zero_in_degree=True))
        else:
            self.encoder.append(
                GATConv(hidden_size * 2, out_size * 2, attn_drop=dropout, residual=True, feat_drop=dropout,
                        num_heads=1, activation=F.elu,
                        allow_zero_in_degree=True))
        self.layer_nums = layersnums
        self.predict = Classifier(out_size * 2)

    def forward(self, s_g, embedList, data, ind, iftrain=True,e=1,text_emb =None):
        drug_emb = self.sum_layers[0](s_g[0], embedList[0],text_emb)
        protein_emb = self.sum_layers[1](s_g[1], embedList[1])
        DPP_emb = torch.cat((drug_emb[data[:, 0]], protein_emb[data[:, 1]]), dim=-1)
        # edge = self.DPP_feature_Graph_Generation(DPP_emb).to(DPP_emb.device)
        edge = self.DPP_toplogy_Graph_Generation(data[:,:2]).to(DPP_emb.device)
        prob= 0.3
        graph1_feat, graph1 = self.random_agument(DPP_emb, edge, prob)
        graph2_feat, graph2 = self.random_agument(DPP_emb, edge, prob)
        edge = edge.permute(1, 0)
        self._old_graph = dgl.graph(data=(edge[0], edge[1]))
        self._old_emb = DPP_emb
        for l in range(self.layer_nums):
            graph1_feat = self.encoder[l](graph1, graph1_feat).squeeze()
            graph2_feat = self.encoder[l](graph2, graph2_feat).squeeze()
        loss = self.caculated_loss(graph1_feat, graph2_feat)
        emb = self.get_emb()
        out = self.predict(emb,e)
        return out, loss

    def DPP_toplogy_Graph_Generation(self, data):
        (A_indices, B_indices) = data.reshape(2, -1)
        num_c_nodes = len(data)
        G_new_adjacency_matrix = np.zeros((num_c_nodes, num_c_nodes), dtype=int)
        for i in range(num_c_nodes):
            G_new_adjacency_matrix[i] = np.logical_or(A_indices == A_indices[i], B_indices == B_indices[i])
        edge = torch.tensor(G_new_adjacency_matrix.astype(int).nonzero()).permute(1, 0)
        return edge

    def DPP_feature_Graph_Generation(self, x):
        adj = x @ x.T
        adj = (adj > torch.mean(adj) ).to(dtype=torch.int)
        adj = adj + torch.eye(adj.shape[0]).to(adj.device)
        edge = adj.nonzero()

        return edge

    def random_agument(self, x, edge, aug_prob=0.1):
        drop_mask = torch.empty((x.size(1),),
                                dtype=torch.float32,
                                device=x.device).uniform_(0, 1) < aug_prob
        x = x.clone()
        x[:, drop_mask] = 0
        mask_rates = torch.FloatTensor(np.ones(len(edge)) * aug_prob)
        masks = torch.bernoulli(1 - mask_rates)
        mask_idx = masks.nonzero().squeeze(1)
        edge = edge[mask_idx].permute(1, 0)
        _graph = dgl.graph([]).to(x.device)
        _graph.add_nodes(x.shape[0])
        _graph.add_edges(edge[0], edge[1])
        return x, _graph

    def get_emb(self):
        for l in range(self.layer_nums):
            self._old_emb = self.encoder[l](self._old_graph, self._old_emb).squeeze()
        return self._old_emb

    def caculated_loss(self, h1, h2):

        z1 = ((h1 - h1.mean(0)) / (h1.std(0)))
        z2 = ((h2 - h2.mean(0)) / (h2.std(0)))

        std_x = h1.var(dim=0)
        std_y = h2.var(dim=0)
        std_loss = torch.sum(torch.sqrt((0.5 - std_x) ** 2)) / \
                   2 + torch.sum(torch.sqrt((0.5 - std_y) ** 2)) / 2
        N = z1.shape[0]
        c = torch.mm(z1, z2.T)
        c = c / (N ** 2)
        loss_inv = torch.diagonal(c).sum()
        return loss_inv + std_loss * 0.001

