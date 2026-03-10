import math
import os

import numpy as np
from Bio import SeqIO
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, RobertaTokenizer
import torch

import pandas as pd

from sklearn.metrics import mean_squared_error, roc_curve, auc, precision_recall_curve
from lifelines.utils import concordance_index
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

import torch.nn.functional as F

root_path = "/root/drug-gene"


class DNA_drug_dataset(Dataset):
    def __init__(self, seq_tokens, smiles_input_ids, smiles_attention_mask, labels):
        self.seq_tokens = seq_tokens
        # print(f'input_ids: {smiles_tokens["input_ids"].shape}; attention_mask: {smiles_tokens["attention_mask"].shape}')
        self.smiles_input_ids = smiles_input_ids
        self.smiles_attention_mask = smiles_attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.seq_tokens[idx], self.smiles_input_ids[idx], self.smiles_attention_mask[idx], self.labels[idx]


def token_seqs_smiles(seqs, seq_seg_len, seq_token_len, smiles):
    seq_tokenizer = AutoTokenizer.from_pretrained(root_path + "/LLM/GPN")
    smiles_tokenizer = RobertaTokenizer.from_pretrained(root_path + "/LLM/ChemBERTa-77M-MLM/", trust_remote_code=True)
    max_smiles_len = 100
    gene_seq_token, drug_smiles_token_input, drug_smiles_token_attention_mask = [], [], []
    for idx, (seq, smile) in enumerate(tqdm(zip(seqs, smiles), total=len(seqs), desc="token seqs and smiles...")):
        part_len = len(seq) // seq_seg_len

        if part_len == 0:
            part_len = len(seq)

        token_seq = [seq[i * part_len:(i + 1) * part_len] for i in range(seq_seg_len)]
        seq_tokens_tmp = seq_tokenizer(token_seq, return_tensors="pt")["input_ids"]

        target_shape = (seq_seg_len, seq_token_len)

        # Get the current shape of seq_tokens_tmp
        current_shape = seq_tokens_tmp.shape

        # Pad or truncate the tensor to match the target shape
        if current_shape[0] < target_shape[0] or current_shape[1] < target_shape[1]:
            seq_tokens = seq_tokens_tmp
        else:
            seq_tokens = seq_tokens_tmp[:target_shape[0], :target_shape[1]]
        seq_tokens = seq_tokens.long()

        gene_seq_token.append(seq_tokens)

        smile_tokens_input, smiles_token_mask = smiles_tokenizer(smile)["input_ids"], smiles_tokenizer(smile)["attention_mask"]
        if len(smile_tokens_input) > max_smiles_len:
            smile_tokens_input = smile_tokens_input[:max_smiles_len]
            smiles_token_mask = smiles_token_mask[:max_smiles_len]
        else:
            smile_tokens_input = smile_tokens_input + [0] * (max_smiles_len - len(smile_tokens_input))
            smiles_token_mask = smiles_token_mask + [0] * (max_smiles_len - len(smiles_token_mask))
        drug_smiles_token_input.append(smile_tokens_input)
        drug_smiles_token_attention_mask.append(smiles_token_mask)

    return gene_seq_token, drug_smiles_token_input, drug_smiles_token_attention_mask


def collate_fn(batch):
    seqs, smiles_input_ids, smiles_attention_mask, labels = zip(*batch)
    seq_tokens = torch.stack(seqs, dim=0)
    smiles_input_ids, smiles_attention_mask = torch.tensor(smiles_input_ids), torch.tensor(smiles_attention_mask)
    labels = torch.tensor(labels, dtype=torch.float32)
    return seq_tokens, smiles_input_ids, smiles_attention_mask, labels


def get_DNA_drug_data_loader(path="GDI", args=None, shuffle=False):
    GDI_file = pd.read_csv(f"{root_path}/data/{path}.csv")
    gene_seqs, drug_smiles, labels = GDI_file["seq"].values.tolist(), GDI_file["smiles"].values.tolist(), GDI_file["label"].values

    gene_seq_token, drug_smiles_token_input, drug_smiles_token_attention_mask = token_seqs_smiles(gene_seqs, args.seq_seg_len, args.seq_token_len, drug_smiles)
    data_set = DNA_drug_dataset(seq_tokens=gene_seq_token, smiles_input_ids=drug_smiles_token_input, smiles_attention_mask=drug_smiles_token_attention_mask, labels=labels)
    data_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return data_loader


def eval(test_targets, test_preds):
    test_targets = np.array(test_targets).flatten()
    test_preds = np.array(test_preds).flatten()
    mse = mean_squared_error(test_targets, test_preds)
    rmse = np.sqrt(mse)

    pcc, p_value = pearsonr(test_targets, test_preds)
    scc, s_p_value = spearmanr(test_targets, test_preds)

    conindex = concordance_index(test_targets, test_preds)
    return {
        "RMSE": rmse,
        'PCC': pcc,
        'SCC': scc,
        'conindex': conindex
    }