import torch
import random

import numpy as np
import pandas as pd
import pytorch_lightning as pl

from completeformer.data import MaskedDataset
from completeformer.models import Completeformer
from pathlib import Path
from tqdm.auto import tqdm
from x_transformers.autoregressive_wrapper import top_p

# This code was taken from https://gist.github.com/kylebgorman/1081951/bce3de986e4b05fc0b63d4d9e0cfa4bde6664365
def _dist(A, B, insertion, deletion, substitution):
    D = np.zeros((len(A) + 1, len(B) + 1))
    for i in range(len(A)): 
        D[i + 1][0] = D[i][0] + deletion
    for j in range(len(B)): 
        D[0][j + 1] = D[0][j] + insertion
    for i in range(len(A)): # fill out middle of matrix
        for j in range(len(B)):
            if A[i] == B[j]:
                D[i + 1][j + 1] = D[i][j] # aka, it's free. 
            else:
                D[i + 1][j + 1] = min(D[i + 1][j] + insertion,
                                      D[i][j + 1] + deletion,
                                      D[i][j]     + substitution)
    return D

def levenshtein_distance(l1, l2, normalize=False):
    dist = _dist(l1, l2, 1, 1, 1)[-1][-1]
    if normalize:
        return 1. - dist / max(len(l1), len(l2))
    else:
        return dist

CONFIG = {
    "dec_heads": 32,
    "dec_layers": 8,
    "dec_max_len": 128,
    "dim": 1024,
    "enc_heads": 16,
    "enc_layers": 12,
    "enc_max_len": 1024,
    "lr": 0.0001,
    "max_epochs": 5, # This will need to be tweaked based on our compute constraints
    "num_warmup_steps": 2000,
    "grad_accum": 2,
}
NAME = "completeformer-training-w-alibi"
NUM_WORKERS = 4
BATCH_SIZE = 32
MAX_LEN = 1024
OUTPUT_DIR = Path("/home/jovyan/data/output")
N_TEST_SET=7_243 # 10% of test set

SIMPLE_LEN_AVG = 11
random.seed(115)
SIMPLE_DS = MaskedDataset(
    max_seq_len=MAX_LEN,
    batch_size=BATCH_SIZE,
    complexity_type="simple",
    complexity_avg_len=SIMPLE_LEN_AVG,
    tokenizer="ncoop57/completeformer-tokenizer",
    num_workers=NUM_WORKERS,
    n=N_TEST_SET # 10% of test set
)

MEDIUM_LEN_AVG = 19
random.seed(115)
MEDIUM_DS = MaskedDataset(
    max_seq_len=MAX_LEN,
    batch_size=BATCH_SIZE,
    complexity_type="medium",
    complexity_avg_len=MEDIUM_LEN_AVG,
    tokenizer="ncoop57/completeformer-tokenizer",
    num_workers=NUM_WORKERS,
    n=N_TEST_SET # 10% of test set
)

COMPLEX_LEN_AVG = 45
random.seed(115)
COMPLEX_DS = MaskedDataset(
    max_seq_len=MAX_LEN,
    batch_size=BATCH_SIZE,
    complexity_type="complex",
    complexity_avg_len=COMPLEX_LEN_AVG,
    tokenizer="ncoop57/completeformer-tokenizer",
    num_workers=NUM_WORKERS,
    n=N_TEST_SET # 10% of test set
)

SIMPLE_DS.prepare_data()
MEDIUM_DS.prepare_data()
COMPLEX_DS.prepare_data()

dfs = []

def run_experiment(train_type, checkpoint_path):
    model = Completeformer.load_from_checkpoint(checkpoint_path).to("cuda")
    model.eval()

    def gen_results(dataloader, train_type, test_type):
        inputs, references, predictions, losses, exact_matches, leven_dists = [], [], [], [], [], []
        for batch in tqdm(dataloader, total=len(dataloader)):
            srcs = batch["input_ids"].to(model.device)
            tgts = batch["labels"].to(model.device)

            srcs_mask = batch["attention_mask"].bool().to(model.device)
            tgts_mask = tgts != model.tokenizer.pad_token_id

            num_tokens = tgts_mask.sum(dim=1)
            max_tokens = torch.max(num_tokens).item() - 1 # -1 for the BOS token
            start_tokens = (torch.ones((batch["input_ids"].shape[0], 1)) * model.tokenizer.bos_token_id).long().to(model.device)

            samples = model.model.generate(srcs, start_tokens, max_tokens, src_mask=srcs_mask, filter_logits_fn=top_p)
            # Remove excess tokens so generated samples and targets are the same length
            new_samples, new_tgts = [], []
            for i in range(0, num_tokens.shape[0]):
                new_samples.append(samples[i][:num_tokens[i].item() - 1].tolist()) # -1 for the BOS token
                new_tgts.append(tgts[i][:num_tokens[i].item()].tolist()) # No need for -1 because BOS token will be removed during decoding
            decoded_preds = model.tokenizer.batch_decode(new_samples, skip_special_tokens=True)
            decoded_labels = model.tokenizer.batch_decode(new_tgts, skip_special_tokens=True)
            exact_match = [a == b for a, b in zip(decoded_preds, decoded_labels)]
            decoded_preds = [model.tokenizer.tokenize(p) for p in decoded_preds]
            decoded_labels = [model.tokenizer.tokenize(l) for l in decoded_labels]

            decoded_inputs = model.tokenizer.batch_decode(new_samples, skip_special_tokens=True)
            decoded_inputs = [model.tokenizer.tokenize(inpt) for inpt in decoded_inputs]
            inputs.extend(decoded_inputs)
            predictions.extend(decoded_preds)
            references.extend(decoded_labels)

            for i in range(srcs.shape[0]):
                loss = model(
                    srcs[i].unsqueeze(0),
                    tgts[i].unsqueeze(0),
                    srcs_mask[i].unsqueeze(0),
                    tgts_mask[i].unsqueeze(0)
                ).cpu().item()
                losses.append(loss)

            exact_matches.extend(exact_match)
            leven_dist = [levenshtein_distance(a, b, normalize=True) for a, b in zip(decoded_preds, decoded_labels)]
            leven_dists.extend(leven_dist)

        train_types = [train_type] * len(inputs)
        test_types = [test_type] * len(inputs)
        df = pd.DataFrame(
            list(zip(train_types, test_types, inputs, references, predictions, losses, exact_matches, leven_dists)),
            columns=["train_type", "test_type", "input", "reference", "prediction", "loss", "exact_match", "levenshtein_distance"]
        )
        dfs.append(df)

    gen_results(
        SIMPLE_DS.test_dataloader(), train_type, "simple"
    )
    gen_results(
        MEDIUM_DS.test_dataloader(), train_type, "medium"
    )
    gen_results(
        COMPLEX_DS.test_dataloader(), train_type, "complex"
    )

checkpoint_path = "/home/jovyan/data/output/simple-wo-alibi-final/checkpoints/final_checkpoint.ckpt"
run_experiment("simple", checkpoint_path)

checkpoint_path = "/home/jovyan/data/output/medium-wo-alibi/checkpoints/final_checkpoint.ckpt"
run_experiment("medium", checkpoint_path)

checkpoint_path = "/home/jovyan/data/output/complex-wo-alibi/checkpoints/final_checkpoint.ckpt"
run_experiment("complex", checkpoint_path)

complete_df = pd.concat(dfs)
complete_df.to_csv("/home/jovyan/data/output/t5_complete_results.csv", index=False)