# import dgl
from utils import *
from camodel import *
import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import dgl.nn.pytorch as dglnn
from sklearn.metrics import roc_auc_score, f1_score
import warnings
import os
from sklearn.metrics.pairwise import cosine_similarity as cos
import pandas as pd
import argparse




warnings.filterwarnings("ignore")
set_random_seed( 52)
args = argparse.ArgumentParser()

args.add_argument("--learning_rate",default=0.001)
args.add_argument("--inp_size",default=128)
args.add_argument("--hidden_size",default=128)
args.add_argument("--out_size",default=128)
args.add_argument("--dropout",default=0.2)
args.add_argument("--ptwd",default=0)
args.add_argument("--epochs",default=200)
args.add_argument("--dataname",default="heter")
args.add_argument("--device",default="cuda:2" )
args = args.parse_args()


inp_size = int(args.inp_size)
hidden_size = int(args.hidden_size)
out_size = int(args.out_size)
dropout = float(args.dropout)
ptlr = float(args.learning_rate)
ptwd = float(args.ptwd)
epochs = int(args.epochs)


def generation_graph(pretrain_epoch):
    for name in [args.dataname]:
        # for name in ["heter","zheng"]:
        data, graph, num, all_meta_paths, index,text_emb = load_dataset(name)
        graph = [i.to(args.device) for i in graph]
        label = torch.tensor(data[:, 2:3]).to(args.device)
        hd = torch.randn((num[0], inp_size))
        hp = torch.randn((num[1], inp_size))
        features_d = hd.to(args.device)
        features_p = hp.to(args.device)
        node_feature = [features_d, features_p]
        train_index = index[0]
        vail_index = index[1]
        test_index = index[2]
        model = CE_DTI(
            all_meta_paths=all_meta_paths,
            in_size=inp_size,
            hidden_size=hidden_size,
            out_size=out_size,
            dropout=dropout,
        ).to(args.device)
        optim = torch.optim.AdamW(lr=ptlr, weight_decay=ptwd, params=model.parameters())
        l = torch.nn.CrossEntropyLoss()
        bestacc = 0
        broc = 0
        bpr = 0
        for e in tqdm.tqdm(range(pretrain_epoch)):
            model.train()
            out, loss = model(graph, node_feature, data, train_index,text_emb =text_emb)
            out = out[train_index]
            optim.zero_grad()
            loss = l(out, label[train_index].reshape(-1)) + loss
            loss.backward()
            optim.step()

            trainACC = (out.argmax(dim=1) == label[train_index].reshape(-1)).sum(
                dtype=float) / len(train_index)
            trainROC = get_roc(out, label[train_index])
            trainPR = get_pr(out, label[train_index])

            model.eval()
            with torch.no_grad():
                out, loss = model(graph, node_feature, data, vail_index, False,e,text_emb =text_emb)
                out = out[vail_index]

                testACC = (out.argmax(dim=1) == label[vail_index].reshape(-1)).sum(
                    dtype=float) / len(vail_index)
                testROC = get_roc(out, label[vail_index])
                testPR = get_pr(out, label[vail_index])
                if e % 1 == 0:
                    print(f"vail:  acc:{testACC.item():.4f}, roc:{testROC.item():.4f},pr:{testPR.item():.4f}" )
                bestacc = max(bestacc, testACC)
                broc = max(broc, testROC)
                bpr = max(bpr, testPR)

        predict(model,graph, node_feature, data, test_index,text_emb,label)

def predict(model,graph, node_feature, data, test_index,text_embm,label):
    with torch.no_grad():
        out, _ = model(graph, node_feature, data,test_index, False, text_emb=text_emb)
        out = out[test_index]
        testACC = (out.argmax(dim=1) == label[test_index].reshape(-1)).sum(
            dtype=float) / len(test_index)
        testROC = get_roc(out, label[test_index])
        testPR = get_pr(out, label[test_index])
        print(f"test acc:{testACC.item():.4f}, roc:{testROC.item():.4f},pr:{testPR.item():.4f}" )


generation_graph(epochs)
