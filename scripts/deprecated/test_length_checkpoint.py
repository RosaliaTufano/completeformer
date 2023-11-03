import argparse
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pytorch_lightning as pl

from completeformer.data import MaskedDataset
from completeformer.models import Completeformer
from pathlib import Path

# setup arg parser for checking length based checkpoints
parser = argparse.ArgumentParser(description='Test length based checkpoints')
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
    "grad_accum": 1,
}
NAME = "completeformer-training-t5-short"
NUM_WORKERS = 4
BATCH_SIZE = 32
MAX_LEN = 1024
OUTPUT_DIR = Path(args.output_dir)
    # "/home/jovyan/data/output")

SHORT_LEN_AVG = 11
SHORT_DS = MaskedDataset(
    max_seq_len=MAX_LEN,
    batch_size=BATCH_SIZE,
    complexity_type="short",
    complexity_avg_len=SHORT_LEN_AVG,
    dataset_config="length_short",
    tokenizer="ncoop57/completeformer-tokenizer",
    num_workers=NUM_WORKERS
)

MED_LEN_AVG = 11
MED_DS = MaskedDataset(
    max_seq_len=MAX_LEN,
    batch_size=BATCH_SIZE,
    complexity_type="medium",
    complexity_avg_len=MED_LEN_AVG,
    dataset_config="length_medium",
    tokenizer="ncoop57/completeformer-tokenizer",
    num_workers=NUM_WORKERS
)

LONG_LEN_AVG = 11
LONG_DS = MaskedDataset(
    max_seq_len=MAX_LEN,
    batch_size=BATCH_SIZE,
    complexity_type="long",
    complexity_avg_len=SHORT_LEN_AVG,
    dataset_config="length_long",
    tokenizer="ncoop57/completeformer-tokenizer",
    num_workers=NUM_WORKERS
)

SHORT_DS.prepare_data()
MED_DS.prepare_data()
LONG_DS.prepare_data()


# checkpoint_path = "/home/jovyan/data/output/medium-t5/checkpoints/final_checkpoint.ckpt"
model = Completeformer.load_from_checkpoint(args.checkpoint_path)

trainer = pl.Trainer(
    gpus=1,
    precision=16,
)

# evaluating using the different test sets
trainer.test(model=model, dataloaders=SHORT_DS.test_dataloader())
trainer.test(model=model, dataloaders=MED_DS.test_dataloader())
trainer.test(model=model, dataloaders=LONG_DS.test_dataloader())