import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from copy import deepcopy
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from utils.reparam_module import ReparamModule
import numpy as np
import random
from tqdm import tqdm
from argparse import ArgumentParser
import os

from utils.minimal import normal_initialization
from module.layers import SeqPoolingLayer

Main_domain_count=0
K = 5
RUNTIME_DEVICE = torch.device('cpu')
MASK_CACHE = {}
ZERO_MASK_CACHE = {}
PIN_MEMORY = False
AMP_DTYPE = None

class ConditionEncoder(nn.Module):
    def __init__(self, K) -> None:
        super().__init__()
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=2,
            dim_feedforward=256,
            dropout=0.5,
            activation='gelu',
            layer_norm_eps=1e-12,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=transformer_layer,
            num_layers=2,
        )
        self.condition_layer = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, K),
        )
        self.pooling_layer = SeqPoolingLayer('mean')
        self.tau = 1

    def forward(self, trm_input, src_mask, memory_key_padding_mask, src_seqlen):
        trm_out = self.encoder(
            src=trm_input,
            mask=src_mask,  # BxLxD
            src_key_padding_mask=memory_key_padding_mask,
        )
        trm_out = self.pooling_layer(trm_out, src_seqlen) # BD
        condition = self.condition_layer(trm_out) # BK
        condition = F.gumbel_softmax(condition, tau=self.tau, dim=-1) # BK
        self.condition4loss = condition
        self.tau = max(self.tau * 0.995, 0.1)
        return condition

class Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # self.item_embedding = nn.Embedding(num_item + 2, 64, padding_idx=0)
        # self.item_embedding_decoder = nn.Embedding(num_item + 2, 64, padding_idx=0)
        self.transformer = nn.Transformer(
            d_model=64,
            nhead=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=256,
            dropout=0.5,
            activation='gelu',
            layer_norm_eps=1e-12,
            batch_first=True,
        )
        self.condition_linear = nn.Sequential(
            nn.Linear(64, 64 * K),
            nn.ReLU(),
            nn.Linear(64 * K, 64 * K)
        )
        self.dropout = nn.Dropout(0.5)
        self.position_embedding = torch.nn.Embedding(args.max_seq_len, 64)
        self.condition_encoder = ConditionEncoder(K)
        self.device = str(RUNTIME_DEVICE)
        self.apply(normal_initialization)
        self.load_pretrained()

    def load_pretrained(self):
        path = os.path.join(args.root_path, 'pre-trained_embedding-new.ckpt')
        # path = path_dict[dataset_name]
        saved = torch.load(path, map_location='cpu')
        pretrained = saved['parameters']['item_embedding.weight']
        """
        pretrained = torch.cat([
            pretrained,
            nn.init.normal_(torch.zeros(2, 64), std=0.02),
            nn.init.normal_(torch.zeros(Main_domain_count, 64), std=0.02)
        ])
        """
        self.item_embedding = nn.Embedding.from_pretrained(pretrained, padding_idx=0, freeze=False)
        self.item_embedding_decoder = self.item_embedding

    def condition_mask(self, logits, src):
        mask = torch.zeros_like(logits, device=logits.device, dtype=torch.bool)
        mask = mask.scatter(-1, src.unsqueeze(-2).repeat(1, mask.shape[1], 1), 1)
        logits = torch.masked_fill(logits, ~mask, -torch.inf)
        return logits

    def forward(self, src, tgt, src_mask, tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            memory_key_padding_mask,
            src_seqlen,
            tgt_seqlen,
        ):
        position_ids = torch.arange(src.size(1), dtype=torch.long, device=self.device)
        position_ids = position_ids.reshape(1, -1)
        src_position_embedding = self.position_embedding(position_ids)
        src_emb = self.dropout(self.item_embedding(src) + src_position_embedding)

        memory = self.transformer.encoder(src_emb, src_mask, src_padding_mask)
        B, L, D = memory.shape
        memory = self.condition_linear(memory).reshape(B, L, K, D)

        position_ids = torch.arange(tgt.size(1), dtype=torch.long, device=self.device)
        position_ids = position_ids.reshape(1, -1)
        tgt_position_embedding = self.position_embedding(position_ids)
        tgt_emb = self.dropout(self.item_embedding(tgt) + tgt_position_embedding)

        condition = self.condition_encoder(tgt_emb, tgt_mask, tgt_padding_mask, tgt_seqlen) # BK
        condition = condition.reshape(B, 1, K, 1)
        memory_cond = (memory * condition).sum(-2)

        outs = self.transformer.decoder(tgt_emb, memory_cond, tgt_mask, None, tgt_padding_mask, memory_key_padding_mask)

        logits = outs @ self.item_embedding_decoder.weight.T
        logits = self.condition_mask(logits, src)

        return logits
    
    def encode(self, src, src_mask):
        position_ids = torch.arange(src.size(1), dtype=torch.long, device=self.device)
        position_ids = position_ids.reshape(1, -1)
        src_position_embedding = self.position_embedding(position_ids)
        src_emb = self.dropout(self.item_embedding(src) + src_position_embedding)

        return self.transformer.encoder(src_emb, src_mask)

    def set_condition(self, condition):
        self.condition = condition

    def decode(self, tgt, memory, tgt_mask):
        B, L, D = memory.shape
        memory = self.condition_linear(memory).reshape(B, L, K, D)[:, :, self.condition]
        position_ids = torch.arange(tgt.size(1), dtype=torch.long, device=self.device)
        position_ids = position_ids.reshape(1, -1)
        tgt_position_embedding = self.position_embedding(position_ids)
        tgt_emb = self.dropout(self.item_embedding(tgt) + tgt_position_embedding)
        
        return self.transformer.decoder(tgt_emb, memory, tgt_mask)

def generate_square_subsequent_mask(sz, device='cuda'):
    device = torch.device(device)
    key = (str(device), sz)
    if key not in MASK_CACHE:
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, -100000).masked_fill(mask == 1, float(0.0))
        MASK_CACHE[key] = mask
    return MASK_CACHE[key]


def get_zero_src_mask(sz, device):
    device = torch.device(device)
    key = (str(device), sz)
    if key not in ZERO_MASK_CACHE:
        ZERO_MASK_CACHE[key] = torch.zeros(sz, sz, device=device, dtype=torch.bool)
    return ZERO_MASK_CACHE[key]


def maybe_autocast():
    if RUNTIME_DEVICE.type != 'cuda' or AMP_DTYPE is None:
        return nullcontext()
    return torch.autocast(device_type='cuda', dtype=AMP_DTYPE)

def create_mask(src, tgt):
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    # src_mask = torch.zeros((src_seq_len, src_seq_len),device='cuda').type(torch.bool)
    src_mask = generate_square_subsequent_mask(src_seq_len)

    src_padding_mask = (src == 0)
    tgt_padding_mask = (tgt == 0)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def inference_mask(logits, src, ys):
    mask = torch.zeros_like(logits, device=logits.device, dtype=torch.bool)
    mask = mask.scatter(-1, src, 1)
    mask = mask.scatter(-1, ys, 0)
    mask[:, 0] = 0
    logits = torch.masked_fill(logits, ~mask, -torch.inf)
    return logits

def inference_mask_generative(logits, src, ys):
    mask = torch.ones_like(logits, device=logits.device, dtype=torch.bool)
    mask = mask.scatter(-1, ys, 0)
    mask[:, 0] = 0
    logits = torch.masked_fill(logits, ~mask, -torch.inf)
    return logits

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(RUNTIME_DEVICE)
    src_mask = src_mask.to(RUNTIME_DEVICE)

    with maybe_autocast():
        memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(RUNTIME_DEVICE)
    for i in range(max_len-1):
        memory = memory.to(RUNTIME_DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(1))
                    .type(torch.bool)).to(RUNTIME_DEVICE)
        with maybe_autocast():
            out = model.decode(ys, memory, tgt_mask)
            prob = out[:, -1] @ model.item_embedding_decoder.weight.T
        if random.random() > 1 or i <= 1:
            prob = inference_mask(prob, src, ys)
        else:
            prob = inference_mask_generative(prob, src, ys)
        _, next_word = torch.max(prob, dim=-1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        if next_word == EOS:
            break
    return ys

def translate(model: torch.nn.Module, src):
    model.eval()
    src = src.reshape(1, -1)
    num_tokens = src.shape[1]
    src_mask = torch.zeros(num_tokens, num_tokens, device=RUNTIME_DEVICE, dtype=torch.bool)
    tgt_tokens = greedy_decode(
        model, src, src_mask, max_len=25, start_symbol=SOS).flatten()
    return tgt_tokens


@torch.inference_mode()
def greedy_decode_batch(model, src, max_len, start_symbol, eos_symbol):
    model.eval()
    src = src.to(RUNTIME_DEVICE, non_blocking=PIN_MEMORY)
    num_tokens = src.shape[1]
    src_mask = get_zero_src_mask(num_tokens, src.device)
    with maybe_autocast():
        memory = model.encode(src, src_mask)
    ys = torch.full((src.size(0), 1), start_symbol, dtype=torch.long, device=src.device)
    finished = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)

    for i in range(max_len - 1):
        tgt_mask = generate_square_subsequent_mask(ys.size(1), device=src.device).type(torch.bool)
        with maybe_autocast():
            out = model.decode(ys, memory, tgt_mask)
            prob = out[:, -1] @ model.item_embedding_decoder.weight.T
        if i <= 1:
            prob = inference_mask(prob, src, ys)
        else:
            prob = inference_mask_generative(prob, src, ys)

        next_word = torch.argmax(prob, dim=-1)
        next_word = torch.where(
            finished,
            torch.full_like(next_word, eos_symbol),
            next_word,
        )
        ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)
        finished = finished | (next_word == eos_symbol)
        if finished.all():
            break

    return ys


def trim_generated_batch(batch_tokens, eos_symbol):
    trimmed = []
    for row in batch_tokens:
        seq = row.tolist()
        if eos_symbol in seq[1:]:
            eos_idx = seq[1:].index(eos_symbol) + 1
            seq = seq[: eos_idx + 1]
        trimmed.append(torch.tensor(seq, dtype=row.dtype, device='cpu'))
    return trimmed


class SequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def collate_sequences(batch):
    return pad_sequence(batch, batch_first=True, padding_value=0)


def translate_in_batches(
    model,
    sequences,
    batch_size,
    max_len,
    start_symbol,
    eos_symbol,
    num_workers,
    prefetch_factor,
    pin_memory,
):
    outputs = []
    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': False,
        'collate_fn': collate_sequences,
        'num_workers': num_workers,
    }
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = prefetch_factor
        loader_kwargs['persistent_workers'] = True
    if pin_memory and RUNTIME_DEVICE.type == 'cuda':
        loader_kwargs['pin_memory'] = True

    data_loader = DataLoader(SequenceDataset(sequences), **loader_kwargs)
    for src in tqdm(data_loader, desc='batch inference'):
        batch_tokens = greedy_decode_batch(
            model=model,
            src=src,
            max_len=max_len,
            start_symbol=start_symbol,
            eos_symbol=eos_symbol,
        )
        outputs.extend(trim_generated_batch(batch_tokens, eos_symbol))
    return outputs


if __name__ == '__main__':


    parser = ArgumentParser()
    parser.add_argument('--root_path', type=str, default='./workdir/ml-1m', help='Directory containing train.pth, a regenerator checkpoint, and the intermediate augmentation files.')
    parser.add_argument('--ckpt_name', type=str, default="regenerator.pth", help='The name of pretrained regenerator')
    parser.add_argument('--begin', '-b', type=int, default=0, help='Used for multi-processing. Beginning of the inference.')
    parser.add_argument('--end', '-e', type=int, default=1000000, help='Used for multi-processing. End of the inference.')
    parser.add_argument('--max_seq_len', type=int, default=50 , help='The job number for Seq2Pat pattern mining.')
    parser.add_argument('--loops', type=int, default=5 , help='loops')
    parser.add_argument('--domain_count', type=int, default=2 , help='The domain_count.')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda', help='Runtime device: cuda or cpu.')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for hybrid inference.')
    parser.add_argument('--num_workers', type=int, default=8, help='DataLoader workers for hybrid inference.')
    parser.add_argument('--prefetch_factor', type=int, default=4, help='Prefetch factor when num_workers > 0.')
    parser.add_argument('--disable_pin_memory', action='store_true', help='Disable pin_memory for host-to-device transfer.')
    parser.add_argument('--amp_dtype', type=str, default='bf16', choices=['bf16', 'fp16', 'none'], help='Autocast dtype on CUDA.')
    parser.add_argument('--max_samples', type=int, default=None, help='Optional cap on the number of source sequences for quick validation.')
    parser.add_argument('--output_name', type=str, default='train_regen.pth', help='Output file name under root_path.')
    args = parser.parse_args()
    begin = args.begin * 5000
    end = args.end * 5000
    Main_domain_count = args.domain_count
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    def resolve_runtime_device(requested_device):
        if requested_device == 'cpu':
            return torch.device('cpu')
        try:
            test_device = torch.device(requested_device)
            torch.zeros(1, device=test_device)
            return test_device
        except Exception as exc:
            print(f'Falling back to CPU because device {requested_device} is unavailable: {exc}')
            return torch.device('cpu')

    RUNTIME_DEVICE = resolve_runtime_device(args.device)
    print(f'Using runtime device: {RUNTIME_DEVICE}')
    if RUNTIME_DEVICE.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
    PIN_MEMORY = (not args.disable_pin_memory) and RUNTIME_DEVICE.type == 'cuda'
    AMP_DTYPE = {
        'bf16': torch.bfloat16,
        'fp16': torch.float16,
        'none': None,
    }[args.amp_dtype]

    dataset_name = os.path.basename(os.path.normpath(args.root_path))
    tmp_path = os.path.join(args.root_path, 'pre-trained_embedding.ckpt')
    # path = path_dict[dataset_name]
    tmp_saved = torch.load(tmp_path, map_location='cpu')
    print(tmp_saved['parameters']['item_embedding.weight'].shape)
    num_item_dict = {
        'toy': tmp_saved['parameters']['item_embedding.weight'].shape[0],
        'sport': tmp_saved['parameters']['item_embedding.weight'].shape[0],
        'beauty':tmp_saved['parameters']['item_embedding.weight'].shape[0],
        'yelp': tmp_saved['parameters']['item_embedding.weight'].shape[0],
        '1m': tmp_saved['parameters']['item_embedding.weight'].shape[0],
        'kr': tmp_saved['parameters']['item_embedding.weight'].shape[0],
        '20m': tmp_saved['parameters']['item_embedding.weight'].shape[0]
    }
    #20m :18346!
    num_item = tmp_saved['parameters']['item_embedding.weight'].shape[0]
    SOS = num_item
    EOS = num_item + 1
    first_domain_token = EOS + 1

    model = Generator().to(RUNTIME_DEVICE)
    model.load_state_dict(torch.load(os.path.join(args.root_path, args.ckpt_name), map_location=RUNTIME_DEVICE))

    def preprocess(seq):
        return torch.tensor([SOS] + seq + [EOS], dtype=torch.long)
    original_data = torch.load(os.path.join(args.root_path, 'train.pth'))
    seqlist = [_[1][:_[3]]  for _ in original_data]
    domain_seqlist=[_[5][:_[3]] for _ in original_data]
    new_nested_list = []
    for sublist in domain_seqlist:
        new_sublist = []
        for item in sublist:
            new_sublist.append(item + first_domain_token)
        new_nested_list.append(new_sublist)
    domain_seqlist=new_nested_list
    final_seqlist = []

    # 遍历每个子列表中的元素
    for seq, domain_seq in zip(seqlist, domain_seqlist):
        combined = [item for pair in zip(seq, domain_seq) for item in pair]
        final_seqlist.append(combined)
    print(f'Loaded {len(seqlist)} sequences for inference.')
    seqlist = [preprocess(_) for _ in final_seqlist]
    target_sequences = seqlist[begin:end]
    if args.max_samples is not None:
        target_sequences = target_sequences[:args.max_samples]
    target_sequences.sort(key=len, reverse=True)
    print(f'Running inference on {len(target_sequences)} sequences.')

    #ori_pattern = torch.load(os.path.join(args.root_path, 'patterns.pth'))

    filtered_sequences = []
    for i in range(args.loops):
        model.set_condition(i)
        filtered_sequences.extend(
            translate_in_batches(
                model=model,
                sequences=target_sequences,
                batch_size=args.batch_size,
                max_len=25,
                start_symbol=SOS,
                eos_symbol=EOS,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
                pin_memory=PIN_MEMORY,
            )
        )
    
    train_set = set()

    for pattern in filtered_sequences:
        seq = pattern.tolist()[1:-1]
        train_set.add(tuple(seq))

    max_seq_len = args.max_seq_len
    def truncate_or_pad(seq):
        cur_seq_len = len(seq)
        if cur_seq_len > max_seq_len:
            return seq[-max_seq_len:]
        else:
            return seq + [0] * (max_seq_len - cur_seq_len)
    train_list = []
    for _ in train_set:
        seq_len = sum([a != 0 for a in list(_)[:-1]])
        if seq_len == 0:
            continue
        train_list.append([
            1,
            truncate_or_pad(list(_)[:-1]),
            truncate_or_pad(list(_)[1:]),
            seq_len,
            [1] * max_seq_len,
            [0] * max_seq_len,
        ])

    out_path = os.path.join(args.root_path, args.output_name)
    torch.save(original_data + train_list, out_path)
    #torch.save(original_data + ori_pattern + train_list, out_path)

