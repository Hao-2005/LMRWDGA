from tqdm import tqdm
from util import get_DNA_drug_data_loader, eval
from model import GDI_model
import torch.nn as nn
import torch
import argparse
import pandas as pd
import time


def compute_label_stats(data_loader):
    ys = []
    for _, _, _, labels in data_loader:
        ys.append(labels.float().view(-1))
    y = torch.cat(ys, dim=0)
    mean = y.mean().item()
    std = y.std(unbiased=False).item()
    if std < 1e-6:
        std = 1.0
    return mean, std


def train_test(train_loader, test_loader, val_loader, args):
    model = GDI_model(args=args).cuda()

    label_mean, label_std = compute_label_stats(train_loader)
    print(f"[Label stats] mean={label_mean:.4f}, std={label_std:.4f}")

    if args.load_model:
        model_path = f'model/{args.dataset}_{args.seq_seg_len}_{args.seq_token_len}_{args.learn_rate}_{args.idx}.pth'
        model.load_state_dict(torch.load(model_path), strict=True)
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': args.learn_rate}])

    mse_loss = nn.MSELoss()
    best_pcc, best_scc, best_ci = 0, 0, 0
    best_result = {}

    for epoch in range(args.epochs):
        model.train()

        print("\n" + "=" * 80)
        print(f"Epoch {epoch} / {args.epochs}")
        print("=" * 80)

        for idx, (seq_tokens, smiles_input_ids, smiles_attention_mask, labels) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            pred = model(seq_tokens, smiles_input_ids, smiles_attention_mask)

            labels = labels.cuda().float()
            labels_norm = (labels - label_mean) / label_std
            loss = mse_loss(pred, labels_norm)

            loss.backward()
            optimizer.step()

        test_result = test_model(model, test_loader, label_mean, label_std)
        val_result = test_model(model, val_loader, label_mean, label_std)

        if val_result["conindex"] > best_ci:
            best_ci = val_result["conindex"]
            best_result = val_result

            print(f'*' * 50)
            print(f'epoch: {epoch}')
            print(f'test_result: {test_result}\nval_result: {val_result}')
            print(f'*' * 50)

            if args.save_model:
                timestamp = time.time()
                local_time = time.localtime(timestamp)
                formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', local_time)

                torch.save(model.state_dict(), f'model/{args.dataset}_{args.seq_seg_len}_{args.seq_token_len}_{args.learn_rate}_{formatted_time}.pth')
                print(f'save model to model/{args.dataset}_{args.seq_seg_len}_{args.seq_token_len}_{args.learn_rate}_{formatted_time}.pth')
            else:
                print(f'do not save model!!!')
    return best_result


def test_model(model, data_loader, label_mean, label_std):
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for idx, (seq_tokens, smiles_input_ids, smiles_attention_mask, labels) in enumerate(tqdm(data_loader)):
            pred = model(seq_tokens, smiles_input_ids, smiles_attention_mask)
            pred = pred * label_std + label_mean

            preds.extend(pred.detach().cpu().numpy().tolist())
            actuals.extend(labels.numpy().tolist())

    result = eval(actuals, preds)
    return result


def cv_5(args, log_file=None):
    best_result = []
    for i in range(5):
        args.idx = i
        train_path, test_path, val_path = args.dataset + "_train_" + str(i), args.dataset + "_test_" + str(i), args.dataset + "_indepent"
        train_data_loader = get_DNA_drug_data_loader(path=train_path, args=args, shuffle=True)
        test_data_loader = get_DNA_drug_data_loader(path=test_path, args=args, shuffle=True)
        val_data_loader = get_DNA_drug_data_loader(path=val_path, args=args, shuffle=False)

        result = train_test(train_data_loader, test_data_loader, val_data_loader, args=args)
        best_result.append(result)

    result_msg = f'args.learn_rate: {args.learn_rate}: best_result: {best_result}'
    print(result_msg)

    return best_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='GDI', help='dataset name: platinum/gdsc')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--learn_rate', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--hidden_size', type=int, default=512, help='the number of layer used')
    parser.add_argument('--seq_seg_len', type=int, default=64, help='the number of whole-genome sequence segmentation segment')
    parser.add_argument('--seq_token_len', type=int, default=512, help='the length of seq token')
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--load_model', type=bool, default=False)

    # CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --save_model True > log/train.log 2>&1 &
    args = parser.parse_args()
    torch.cuda.empty_cache()
    cv_5(args=args)


