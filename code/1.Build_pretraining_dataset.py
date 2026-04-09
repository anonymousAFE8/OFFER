import argparse
import os
from multiprocessing import Pool, cpu_count
from random import shuffle

import torch
from tqdm import tqdm


def is_sublist(sublst, lst):
    for element in sublst:
        try:
            ind = lst.index(element)
        except ValueError:
            return False
        lst = lst[ind + 1 :]
    return True


def find_subsequence_indices(pattern, seq_ori):
    indices = []
    iter_seq_ori = iter(enumerate(seq_ori))

    try:
        for elem in pattern:
            for index, value in iter_seq_ori:
                if value == elem:
                    indices.append(index)
                    break
        return indices
    except StopIteration:
        return None


def process_sequence(seq_ori, patterns_value, ori_domain):
    local_data_generation_pair = []
    shuffle(patterns_value)
    cnt = 0
    for pattern in patterns_value:
        if cnt >= 10:
            break
        indices = find_subsequence_indices(pattern, seq_ori)
        if indices is not None and len(indices) == len(pattern):
            pattern_domain = [ori_domain[i] for i in indices]
            local_data_generation_pair.append([seq_ori, pattern, ori_domain, pattern_domain])
            cnt += 1
    return local_data_generation_pair


def extract_seq_domain_info(original_train):
    seq_list_ori_domain = []
    for item in original_train:
        seq = item[1][: item[3]] + [item[2][item[3] - 1]]
        domain = item[5]
        seq_list_ori_domain.append((seq, domain))
    return seq_list_ori_domain


def truncate_or_pad(seq, max_seq_len):
    cur_seq_len = len(seq)
    if cur_seq_len > max_seq_len:
        return seq[-max_seq_len:]
    return seq + [0] * (max_seq_len - cur_seq_len)


def build_patterns_train_list(patterns_value, original_train, max_seq_len):
    train_set = set()
    for pattern in patterns_value:
        seq = pattern
        train_set.add(tuple(truncate_or_pad(seq[:-1], max_seq_len) + truncate_or_pad(seq[1:], max_seq_len)))

    train_list = []
    for packed in list(train_set):
        train_item_seq = list(packed[:max_seq_len])
        target_item_seq = list(packed[max_seq_len:])
        seq_len = sum(1 for item in train_item_seq if item != 0)
        train_list.append(
            [
                0,
                train_item_seq,
                target_item_seq,
                seq_len,
                [1] * seq_len + [0] * (max_seq_len - seq_len),
                [0] * max_seq_len,
            ]
        )
    return train_list + original_train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="./workdir/ml-1m", help="Directory containing intermediate tensors such as seq2pat_data.pth and train_ori.pth.")
    parser.add_argument("--alpha", type=int, default=5, help="The sliding window size for pre-training dataset construction.")
    parser.add_argument("--beta", type=int, default=4, help="The threshold for pre-training dataset construction.")
    parser.add_argument("--n_jobs", type=int, default=20, help="The job number for Seq2Pat pattern mining.")
    parser.add_argument("--max_seq_len", type=int, default=50, help="The job number for Seq2Pat pattern mining.")
    args = parser.parse_args()

    max_seq_len = args.max_seq_len

    seq2pat_data_path = os.path.join(args.root_path, "seq2pat_data.pth")
    seq2pat_data = torch.load(seq2pat_data_path)
    print(f"Original dataset loaded with size {len(seq2pat_data)}")

    try:
        from sequential.seq2pat import Seq2Pat
    except ImportError:
        from seq2pat import Seq2Pat

    seq2pat = Seq2Pat(sequences=seq2pat_data, n_jobs=args.n_jobs, max_span=args.alpha)
    print("Performing rule-based pattern-mining!")
    patterns = seq2pat.get_patterns(min_frequency=args.beta)
    patterns_value = [_[:-1] for _ in patterns]
    patterns_value = [lst_ for lst_ in patterns_value if len(lst_) >= 3]
    print(patterns_value[0:5])
    print(f"Rule-based patterns mined with size {len(patterns_value)}")

    original_train_path = os.path.join(args.root_path, "train_ori.pth")
    original_train = torch.load(original_train_path)

    patterns_output_path = os.path.join(args.root_path, "patterns.pth")
    torch.save(build_patterns_train_list(patterns_value, original_train, max_seq_len), patterns_output_path)
    print(f"Saved refine-only patterns dataset to {patterns_output_path}")

    seq_list_ori_domain = extract_seq_domain_info(original_train)

    with Pool(cpu_count()) as pool:
        results = []
        for seq_ori, ori_domain in tqdm(seq_list_ori_domain):
            result = pool.apply_async(process_sequence, (seq_ori, patterns_value, ori_domain))
            results.append(result)

        data_generation_pair = []
        for result in tqdm(results):
            data_generation_pair.extend(result.get())

    print(f"Building sequence-pattern pair dataset with size {len(data_generation_pair)}.")
    print(data_generation_pair[0:5])
    output_path = os.path.join(args.root_path, "seq-pat-pair.pth")
    torch.save(data_generation_pair, output_path)
