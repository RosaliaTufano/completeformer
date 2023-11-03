import argparse
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pytorch_lightning as pl

from completeformer.data import MaskedDataset
from completeformer.models import Completeformer
from pathlib import Path

# setup arg parser for checking complexity based checkpoints
parser = argparse.ArgumentParser(description='Test complexity based checkpoints')
parser.add_argument('--checkpoint_path', type=str, help='path to checkpoint')
parser.add_argument('--output_dir', type=str, help='path to output directory')

args = parser.parse_args()

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
OUTPUT_DIR = Path(args.output_dir)
    # "/home/jovyan/data/output")

SIMPLE_LEN_AVG = 11
SIMPLE_DS = MaskedDataset(
    max_seq_len=MAX_LEN,
    batch_size=BATCH_SIZE,
    complexity_type="simple",
    complexity_avg_len=SIMPLE_LEN_AVG,
    tokenizer="ncoop57/completeformer-tokenizer",
    num_workers=NUM_WORKERS
)

MEDIUM_LEN_AVG = 19
MEDIUM_DS = MaskedDataset(
    max_seq_len=MAX_LEN,
    batch_size=BATCH_SIZE,
    complexity_type="medium",
    complexity_avg_len=MEDIUM_LEN_AVG,
    tokenizer="ncoop57/completeformer-tokenizer",
    num_workers=NUM_WORKERS
)

COMPLEX_LEN_AVG = 45
COMPLEX_DS = MaskedDataset(
    max_seq_len=MAX_LEN,
    batch_size=BATCH_SIZE,
    complexity_type="complex",
    complexity_avg_len=COMPLEX_LEN_AVG,
    tokenizer="ncoop57/completeformer-tokenizer",
    num_workers=NUM_WORKERS
)

SIMPLE_DS.prepare_data()
MEDIUM_DS.prepare_data()
COMPLEX_DS.prepare_data()

# checkpoint_path = "/home/jovyan/data/output/alibi_checkpoints/medium-w-alibi/final_checkpoint_m.ckpt"
model = Completeformer.load_from_checkpoint(args.checkpoint_path)

trainer = pl.Trainer(
    gpus=1,
    precision=16,
)

# evaluating using the different test sets
trainer.test(model=model, dataloaders=SIMPLE_DS.test_dataloader())
trainer.test(model=model, dataloaders=MEDIUM_DS.test_dataloader())
trainer.test(model=model, dataloaders=COMPLEX_DS.test_dataloader())