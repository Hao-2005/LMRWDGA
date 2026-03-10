import torch
import torch.nn as nn
from transformers import AutoModel, RobertaModel, AutoModelForMaskedLM
import LLM.GPN
import gpn.gpn.model
import argparse
from gpn_model import load_gpn_model

root_path = "/root/drug-gene"

class GDI_model(nn.Module):
    def __init__(self, args):
        super(GDI_model, self).__init__()
        gene_language_model_file = root_path + "/LLM/GPN"
        drug_language_model_file = root_path + "/LLM/ChemBERTa-77M-MLM/"

        print("Initializing GDI model...")
        self.gene_language_model = AutoModel.from_pretrained(gene_language_model_file).cuda()
        self.drug_language_model = RobertaModel.from_pretrained(drug_language_model_file).cuda()

        self.args = args
        self.gene_emb_W = nn.Parameter(torch.randn(args.seq_seg_len, args.hidden_size))
        self.gene_emb_B = nn.Parameter(torch.randn(args.seq_seg_len, args.hidden_size))

        self.predictor = nn.Sequential(
            nn.Linear(896, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        ).cuda()

        self.seq_ln = nn.LayerNorm(args.hidden_size).cuda()  # 512
        self.drug_ln = nn.LayerNorm(384).cuda()

    def forward(self, seq_tokens, smiles_input_ids, smiles_attention_mask):
        smiles_input_ids, smiles_attention_mask = smiles_input_ids.cuda(), smiles_attention_mask.cuda()
        seq_tokens = seq_tokens.reshape(-1, int(seq_tokens.numel() / self.args.seq_seg_len))

        seq_emb = self.gene_language_model(input_ids=seq_tokens.cuda())['last_hidden_state']
        seq_emb = torch.mean(seq_emb, dim=1)
        seq_emb = seq_emb * self.gene_emb_W + self.gene_emb_B
        seq_emb = seq_emb.reshape(-1, self.args.seq_seg_len, self.args.hidden_size)
        seq_emb = torch.mean(seq_emb, dim=1)

        drug_emb = self.drug_language_model(smiles_input_ids, attention_mask=smiles_attention_mask)['pooler_output']

        seq_emb = self.seq_ln(seq_emb)
        drug_emb = self.drug_ln(drug_emb)

        emb = torch.cat([seq_emb, drug_emb], dim=1)
        output = self.predictor(emb)

        if output.dim() == 2 and output.size(1) == 1:
            output = output.squeeze(1)
        elif output.dim() == 0:
            output = output.unsqueeze(0)

        return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='GDI', help='dataset name: platinum/gdsc')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--learn_rate', type=float, default=0.00005, help='learning rate')
    parser.add_argument('--hidden_size', type=int, default=512, help='the number of layer used')
    parser.add_argument('--seq_seg_len', type=int, default=128, help='the length of seq token')
    parser.add_argument('--seq_token_len', type=int, default=256, help='the length of seq token')
    parser.add_argument('--save_model', type=bool, default=False)
    parser.add_argument('--load_model', type=bool, default=False)
    # CUDA_VISIBLE_DEVICES=3 nohup python -u main.py --save_model True > log/IC50_0.0001_34.log 2>&1 &
    args = parser.parse_args()

    model = GDI_model(args=args).cuda()