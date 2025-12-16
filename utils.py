import datetime
import dgl
import errno
import numpy as np
import pandas as pd
import os
import pickle
import random
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
from sklearn.metrics import auc as auc3
from dgl.data.utils import download, get_download_dir, _get_dgl_url
from pprint import pprint
from scipy import sparse
from scipy import io as sio
from sklearn.metrics.pairwise import cosine_similarity as cos
import time
import scipy.spatial.distance as dist
from torch.utils.data import random_split


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)




def load_hetero(network_path):
    drug_drug = np.loadtxt(network_path + 'mat_drug_drug.txt')
    drug_eye = torch.eye(drug_drug.shape[0])
    drug_chemical = np.loadtxt(network_path + 'Similarity_Matrix_Drugs.txt')
    drug_disease = np.loadtxt(network_path + 'mat_drug_disease.txt')
    drug_sideeffect = np.loadtxt(network_path + 'mat_drug_se.txt')
    protein_protein = np.loadtxt(network_path + 'mat_protein_protein.txt')
    protein_sequence = np.loadtxt(network_path + 'Similarity_Matrix_Proteins.txt') / 100
    protein_disease = np.loadtxt(network_path + 'mat_protein_disease.txt')

    d_d_sideeffect = torch.tensor(np.dot(drug_sideeffect, drug_sideeffect.T))
    d_d_sideeffect = (d_d_sideeffect > torch.mean(d_d_sideeffect)).type(dtype=torch.int) + drug_eye
    d_d_sideeffect = sparse.csr_matrix(d_d_sideeffect).tocoo()


    d_d_disease = torch.tensor(np.dot(drug_disease, drug_disease.T))
    d_d_disease = (d_d_disease > torch.mean(d_d_disease)).type(dtype=torch.int) + drug_eye
    d_d_disease = sparse.csr_matrix(d_d_disease).tocoo()

    drug_drug = sparse.csr_matrix(torch.tensor(drug_drug) + drug_eye).tocoo()
    drug_chemical[drug_chemical >= np.mean(drug_chemical)] = 1
    drug_chemical[drug_chemical < np.mean(drug_chemical)] = 0
    drug_chemical = sparse.csr_matrix(drug_chemical).tocoo()

    eye = np.eye(protein_protein.shape[0])
    protein_protein = sparse.csr_matrix(protein_protein + eye).tocoo()

    protein_sequence[protein_sequence >= np.mean(protein_sequence)] = 1
    protein_sequence[protein_sequence < np.mean(protein_sequence)] = 0
    protein_sequence = sparse.csr_matrix(protein_sequence).tocoo()

    p_p_di = torch.tensor(np.dot(protein_disease, protein_disease.T))
    p_p_di = (p_p_di > torch.mean(p_p_di)).type(dtype=torch.int) + eye
    p_p_di = sparse.csr_matrix(p_p_di).tocoo()

    d_d = dgl.heterograph({("drug", "similarity", "drug"): (drug_drug.row, drug_drug.col),
                           ("drug", "chemical", "drug"): (drug_chemical.row, drug_chemical.col),
                           ("drug", "ddi", "drug"): (torch.tensor(d_d_disease.row), torch.tensor(d_d_disease.col)),
                           ("drug", "dse", "drug"): (d_d_sideeffect.row, d_d_sideeffect.col)
                           })
    p_p = dgl.heterograph({("protein", "similarity", "protein"): (protein_protein.row, protein_protein.col),
                           ("protein", "sequence", "protein"): (protein_sequence.row, protein_sequence.col),
                           ("protein", "pdi", "protein"): (p_p_di.row, p_p_di.col),
                           })

    dti_o = torch.tensor(np.loadtxt(network_path + 'mat_drug_protein.txt'))

    true_index = torch.where(dti_o == 1)
    tep_label = torch.ones(len(true_index[0]), dtype=torch.int).reshape(-1, 1)
    train_positive_index = torch.cat((true_index[0].reshape(-1, 1), true_index[1].reshape(-1, 1), tep_label), dim=1)
    false_index = torch.where(dti_o == 0)
    tep_label = torch.zeros(len(false_index[0]), dtype=torch.int).reshape(-1, 1)
    whole_negative_index = torch.cat((false_index[0].reshape(-1, 1), false_index[1].reshape(-1, 1), tep_label), dim=1)
    negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),
                                             size=len(train_positive_index),
                                             replace=False)

    dataset = torch.tensor(np.concatenate((train_positive_index, whole_negative_index[negative_sample_index]), axis=0),
                           dtype=torch.long)
    num_drug = d_d.num_nodes()
    num_protein = p_p.num_nodes()
    graph = [d_d, p_p]
    node_num = [num_drug, num_protein]
    all_meta_paths = [d_d.canonical_etypes, p_p.canonical_etypes]

    text_emb = pd.read_pickle(network_path+"drug_description_emb.pkl")


    trp = torch.tensor(np.loadtxt(network_path + "train_ind.txt"), dtype=torch.long)
    vap = torch.tensor(np.loadtxt(network_path + "vail_ind.txt"), dtype=torch.long)
    tep = torch.tensor(np.loadtxt(network_path + "test_ind.txt"), dtype=torch.long)

    return dataset, graph, node_num, all_meta_paths, (trp, vap, tep),text_emb



def load_zheng(network_path):
    """
    meta_path of drug
    """

    drug_sideeffect = np.loadtxt(network_path + 'mat_drug_sideeffects.txt')
    drug_drug = np.loadtxt(network_path + 'mat_drug_chemical_sim.txt')
    drug_substituent = np.loadtxt(network_path + 'mat_drug_sub_stituent.txt')
    drug_chemical = np.loadtxt(network_path + "mat_drug_chemical_substructures.txt")

    drug_drug = sparse.csr_matrix(torch.tensor(drug_drug) + torch.eye(drug_drug.shape[0])).tocoo()


    d_d_ch = torch.tensor(np.dot(drug_chemical, drug_chemical.T))
    d_d_ch = (d_d_ch > torch.mean(d_d_ch)).type(dtype=torch.int) + torch.eye(d_d_ch.shape[0])
    d_d_ch = sparse.csr_matrix(d_d_ch).tocoo()

    d_d_si = torch.tensor(np.dot(drug_sideeffect, drug_sideeffect.T))
    d_d_si = (d_d_si > torch.mean(d_d_si)).type(dtype=torch.int) + torch.eye(d_d_si.shape[0])
    d_d_si = sparse.csr_matrix(d_d_si).tocoo()

    d_d_sub = torch.tensor(np.dot(drug_substituent, drug_substituent.T))
    d_d_sub = (d_d_sub > torch.mean(d_d_sub)).type(dtype=torch.int) + torch.eye(d_d_sub.shape[0])
    d_d_sub = sparse.csr_matrix(d_d_sub).tocoo()

    protein_protein = np.loadtxt(network_path + 'mat_target_GO_sim.txt')
    protein_GO = np.loadtxt(network_path + 'mat_target_GO.txt')

    p_p_GO = torch.tensor(np.dot(protein_GO, protein_GO.T))
    p_p_GO = (p_p_GO > torch.mean(p_p_GO)).type(dtype=torch.int) + torch.eye(p_p_GO.shape[0])
    p_p_GO = sparse.csr_matrix(p_p_GO).tocoo()

    protein_protein = sparse.csr_matrix(torch.tensor(protein_protein) + torch.eye(protein_protein.shape[0])).tocoo()

    d_d = dgl.heterograph({("drug", "similarity", "drug"): (drug_drug.row, drug_drug.col),
                           ("drug", "chemical", "drug"): (d_d_ch.row, d_d_ch.col),
                           ("drug", "sideeffect", "drug"): (d_d_si.row, d_d_si.col),
                           ("drug", "substituent", "drug"): (d_d_sub.row, d_d_sub.col)
                           })
    p_p = dgl.heterograph({("protein", "similarity", "protein"): (protein_protein.row, protein_protein.col),
                           ("protein", "Go", "protein"): (p_p_GO.row, p_p_GO.col), })

    dti_o = torch.tensor(np.loadtxt(network_path + 'mat_drug_target.txt'))

    true_index = torch.where(dti_o == 1)
    tep_label = torch.ones(len(true_index[0]), dtype=torch.int).reshape(-1, 1)
    whole_positive_index = torch.cat((true_index[0].reshape(-1, 1), true_index[1].reshape(-1, 1), tep_label), dim=1)

    train_positive_index = np.random.choice(np.arange(len(whole_positive_index)),
                                             size=2000,
                                             replace=False)

    false_index = torch.where(dti_o == 0)
    tep_label = torch.zeros(len(false_index[0]), dtype=torch.int).reshape(-1, 1)
    whole_negative_index = torch.cat((false_index[0].reshape(-1, 1), false_index[1].reshape(-1, 1), tep_label), dim=1)
    negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),
                                             size=len(train_positive_index),
                                             replace=False)

    dataset = torch.tensor(np.concatenate((whole_positive_index[train_positive_index], whole_negative_index[negative_sample_index]), axis=0),
                           dtype=torch.long)
    num_drug = d_d.num_nodes()
    num_protein = p_p.num_nodes()
    graph = [d_d, p_p]
    node_num = [num_drug, num_protein]
    all_meta_paths = [d_d.canonical_etypes, p_p.canonical_etypes]

    text_emb = pd.read_pickle(network_path+"drug_description_emb.pkl")

    trp = torch.tensor(np.loadtxt( network_path+"train_ind.txt"), dtype=torch.long)
    vap = torch.tensor(np.loadtxt(network_path+"vail_ind.txt"), dtype=torch.long)
    tep = torch.tensor(np.loadtxt(network_path+"test_ind.txt"), dtype=torch.long)

    return dataset, graph, node_num, all_meta_paths, (trp, vap, tep),text_emb


def get_roc(out, label):
    return roc_auc_score(label.cpu(), out[:, 1:].cpu().detach().numpy())


def get_pr(out, label):
    precision, recall, thresholds = precision_recall_curve(label.cpu(), out[:, 1:].cpu().detach().numpy())

    return auc3(recall, precision)


def get_f1score(out, label):
    return f1_score(label.cpu(), out.argmax(dim=1).cpu().detach().numpy())


def load_dataset(dateName):
    if dateName == "heter":
        return load_hetero("./data/heter/")
    elif dateName == "zheng":
        return load_zheng("./data/zheng/")
    else:
        print(f"no found dataname {dateName}")

