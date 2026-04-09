import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from tokenizer_support.model import SASRec
from tokenizer_support.utils import WarpSampler, build_index, data_partition, evaluate, evaluate_valid


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TOKENIZER_CKPT = os.path.normpath(
    os.path.join(
        SCRIPT_DIR,
        "..",
        "artifacts",
        "tokenizer",
        "ml-1m",
        "SASRec.epoch=1000.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth",
    )
)
DEFAULT_RUN_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "artifacts", "tokenizer_runs"))


def str2bool(s):
    if s not in {"false", "true"}:
        raise ValueError("Not a valid boolean string")
    return s == "true"


def entropy(prob):
    return -prob * np.log2(prob) if prob > 0 else 0


def calculate_token_entropies(sequences, item_idx_seq, model, args):
    model.eval()
    entropy_dict = {}
    with torch.no_grad():
        for seq_key, next_items in tqdm(sequences.items(), desc="Calculating token entropies", leave=False):
            seq_tensor = np.array([np.array(seq_key)])
            user_id = np.array([1])
            predictions = -model.predict(user_id, seq_tensor, item_idx_seq)
            probs = torch.softmax(predictions, dim=-1)
            entropies = [entropy(probs[0, item - 1].item()) for item in next_items]
            entropy_dict[seq_key] = entropies
    return entropy_dict


def count_sequences_occurrences(seq_key, user_train):
    count = 0
    seq_len = len(seq_key)
    for _, items in user_train.items():
        for i in range(len(items) - seq_len + 1):
            if tuple(items[i : i + seq_len]) == seq_key:
                count += 1
    return count


def build_successor_dict(user_train, seq_len):
    successor_dict = {}
    for _, items in user_train.items():
        for i in range(len(items) - seq_len):
            current_seq = tuple(items[i : i + seq_len])
            next_item = items[i + seq_len]
            if current_seq not in successor_dict:
                successor_dict[current_seq] = []
            if next_item not in successor_dict[current_seq]:
                successor_dict[current_seq].append(next_item)
    return successor_dict


def build_count_dict(user_train, seq_len):
    count_dict = {}
    for _, items in user_train.items():
        for i in range(len(items) - seq_len + 1):
            current_seq = tuple(items[i : i + seq_len])
            if current_seq not in count_dict:
                count_dict[current_seq] = 0
            count_dict[current_seq] += 1
    return count_dict


def calculate_item_entropy(item_counts, total_count):
    probabilities = item_counts / total_count
    return -np.sum(probabilities * np.log2(probabilities + 1e-9))


def tokenizer(model, dataset, args, run_dir):
    user_train, _, _, _, itemnum = dataset
    token_set = []
    item_idx_seq = list(range(1, itemnum + 1))

    total_items_count = sum(len(items) for items in user_train.values())
    item_counts = np.zeros(itemnum + 1)
    for items in user_train.values():
        for item in items:
            item_counts[item] += 1

    print("Calculating entropy for each item...")
    item_entropies = {
        item: calculate_item_entropy(item_counts[item], total_items_count)
        for item in range(1, itemnum + 1)
    }

    average_ent = np.mean(list(item_entropies.values()))
    print(f"Global average entropy: {average_ent}")
    average_ent = args.tokenizer_threshold * 5 * average_ent

    to_analyze = {}
    for item, ent in item_entropies.items():
        if ent < average_ent and item_counts[item] > 2:
            to_analyze[(item,)] = ent

    iteration = 1
    while to_analyze:
        seq_len = len(list(to_analyze.keys())[0])
        print(f"Iteration {iteration}: analyzing sequences of length {seq_len + 1}")
        successor_dict = build_successor_dict(user_train, seq_len)

        top_sequences_stats = []
        for seq_key, seq_entropy in list(to_analyze.items())[:5]:
            count = count_sequences_occurrences(seq_key, user_train)
            top_sequences_stats.append((seq_key, seq_entropy, count))
        for seq_key, seq_entropy, count in top_sequences_stats:
            print(f"  Sequence={seq_key}, entropy={seq_entropy:.6f}, count={count}")

        new_sequences = {}
        for seq_key in to_analyze.keys():
            if seq_key in successor_dict:
                new_sequences[seq_key] = successor_dict[seq_key]
        if not new_sequences:
            break

        new_entropy_dict = calculate_token_entropies(new_sequences, item_idx_seq, model, args)
        updated_to_analyze = {}
        count_dict = build_count_dict(user_train, seq_len + 1)
        count_dict_prev = build_count_dict(user_train, seq_len)

        for seq_key, next_entropy in new_entropy_dict.items():
            for idx, each_nxt_ent in enumerate(next_entropy):
                combined_entropy = to_analyze[seq_key] + each_nxt_ent
                extended_key = seq_key + (new_sequences[seq_key][idx],)
                if combined_entropy < average_ent and len(extended_key) < 5:
                    if count_dict[extended_key] >= 2:
                        updated_to_analyze[extended_key] = combined_entropy
                    elif len(seq_key) > 1 and seq_key not in token_set:
                        token_set.append(seq_key)
                elif count_dict_prev[seq_key] >= 2 and len(seq_key) > 1 and seq_key not in token_set:
                    token_set.append(seq_key)

        print(f"  remaining={len(updated_to_analyze)}, collected_tokens={len(token_set)}")
        to_analyze = updated_to_analyze
        iteration += 1

    if args.target_name:
        output_path = os.path.join(run_dir, f"{args.output_dataset_name}.json")
    else:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            run_dir,
            f"{args.dataset}_token_T_{args.tokenizer_threshold}_T2_{args.tokenizer_threshold_p2}_{current_time}.json",
        )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(token_set, f, indent=1)
    print(f"Tokenization completed. Output written to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ml-1m")
    parser.add_argument("--train_dir", default="release_run")
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--maxlen", default=200, type=int)
    parser.add_argument("--hidden_units", default=50, type=int)
    parser.add_argument("--num_blocks", default=2, type=int)
    parser.add_argument("--num_epochs", default=1000, type=int)
    parser.add_argument("--num_heads", default=1, type=int)
    parser.add_argument("--dropout_rate", default=0.2, type=float)
    parser.add_argument("--l2_emb", default=0.0, type=float)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--inference_only", default=False, type=str2bool)
    parser.add_argument("--tokenize_only", default=False, type=str2bool)
    parser.add_argument("--state_dict_path", default=DEFAULT_TOKENIZER_CKPT, type=str)
    parser.add_argument("--norm_first", action="store_true", default=False)
    parser.add_argument("--tokenizer_threshold", default=0.2, type=float)
    parser.add_argument("--tokenizer_threshold_p2", default=0.2, type=float)
    parser.add_argument("--output_dataset_name", default="ml-1m", type=str)
    parser.add_argument("--target_name", default=False, type=bool)
    parser.add_argument("--savename", default="best_save.pth", type=str)
    parser.add_argument("--eval_epoch", default=100, type=int)
    args = parser.parse_args()

    run_dir = os.path.join(DEFAULT_RUN_ROOT, args.dataset + "_" + args.train_dir)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "args.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join([str(k) + "," + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

    u2i_index, i2u_index = build_index(args.dataset)
    _ = (u2i_index, i2u_index)

    dataset = data_partition(args.dataset)
    user_train, user_valid, user_test, usernum, itemnum = dataset
    num_batch = (len(user_train) - 1) // args.batch_size + 1
    avg_len = sum(len(user_train[u]) for u in user_train) / len(user_train)
    print(f"batch_size={args.batch_size}, num_batch={num_batch}")
    print(f"average sequence length: {avg_len:.2f}")

    f = open(os.path.join(run_dir, "log.txt"), "w", encoding="utf-8")
    f.write("epoch (val_ndcg, val_hr) (test_ndcg, test_hr)\n")

    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    model = SASRec(usernum, itemnum, args).to(args.device)

    for _, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass

    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0
    model.train()

    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find("epoch=") + 6 :]
            epoch_start_idx = int(tail[: tail.find(".")]) + 1
        except Exception:
            print("failed loading state_dicts, please check:", args.state_dict_path)
            raise

    if args.tokenize_only:
        model.eval()
        tokenizer(model, dataset, args, run_dir)
        f.close()
        sampler.close()
        return

    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args)
        print("test (NDCG@10: %.4f, HR@10: %.4f)" % (t_test[0], t_test[1]))
        f.close()
        sampler.close()
        return

    bce_criterion = torch.nn.BCEWithLogitsLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    total_eval_time = 0.0
    t0 = time.time()

    for epoch in tqdm(range(epoch_start_idx, args.num_epochs + 1)):
        for _ in range(num_batch):
            u, seq, pos, neg = sampler.next_batch()
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels = torch.ones(pos_logits.shape, device=args.device)
            neg_labels = torch.zeros(neg_logits.shape, device=args.device)
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()

        if epoch % args.eval_epoch == 0:
            model.eval()
            eval_elapsed = time.time() - t0
            total_eval_time += eval_elapsed
            print("Evaluating", end="")
            t_test = evaluate(model, dataset, args)
            t_valid = evaluate_valid(model, dataset, args)
            print(
                "epoch:%d, time:%f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)"
                % (epoch, total_eval_time, t_valid[0], t_valid[1], t_test[0], t_test[1])
            )

            if t_valid[0] > best_val_ndcg or t_valid[1] > best_val_hr or t_test[0] > best_test_ndcg or t_test[1] > best_test_hr:
                best_val_ndcg = max(t_valid[0], best_val_ndcg)
                best_val_hr = max(t_valid[1], best_val_hr)
                best_test_ndcg = max(t_test[0], best_test_ndcg)
                best_test_hr = max(t_test[1], best_test_hr)
                fname = "SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth".format(
                    epoch, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen
                )
                torch.save(model.state_dict(), os.path.join(run_dir, fname))

            f.write(str(epoch) + " " + str(t_valid) + " " + str(t_test) + "\n")
            f.flush()
            t0 = time.time()
            model.train()

        if epoch == args.num_epochs:
            fname = "SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth".format(
                args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen
            )
            torch.save(model.state_dict(), os.path.join(run_dir, fname))
            torch.save(model.state_dict(), os.path.join(run_dir, args.savename))

    f.close()
    sampler.close()
    print("Done")


if __name__ == "__main__":
    main()
