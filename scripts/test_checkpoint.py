import argparse
import json
import os
import torch
import wandb

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pytorch_lightning as pl

from completeformer.data import CompleteformerDataset
from completeformer.models import Completeformer
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--length", type=str, default="long")
    parser.add_argument("--language", type=str, default="python")
    parser.add_argument("--position_type", type=str, default="alibi")
    parser.add_argument("--output_dir", type=str, default="/semeru/completeformer/models")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=os.cpu_count())

    return parser.parse_args()

args = parse_args()


NAME = f"completeformer_{args.position_type}_{args.length}_test"
wandb.init(project="Completeformer", name=NAME)
ENC_MAX_LEN = 1024
DEC_MAX_LEN = 128
OUTPUT_DIR = Path(args.output_dir) / f"{args.position_type}_{args.length}"

# short_ds = CompleteformerDataset(
#     length="short",
    # language=args.language,
#     tokenizer_name="semeru/completeformer_tokenizer",
#     batch_size=args.batch_size,
#     enc_max_len=ENC_MAX_LEN,
#     dec_max_len=DEC_MAX_LEN,
#     num_workers=args.num_workers,
# )
# short_ds.prepare_data()
# medium_ds = CompleteformerDataset(
#     length="medium",
    # language=args.language,
#     tokenizer_name="semeru/completeformer_tokenizer",
#     batch_size=args.batch_size,
#     enc_max_len=ENC_MAX_LEN,
#     dec_max_len=DEC_MAX_LEN,
#     num_workers=args.num_workers,
# )
# medium_ds.prepare_data()
long_ds = CompleteformerDataset(
    length="long",
    language=args.language,
    tokenizer_name="semeru/completeformer_tokenizer",
    batch_size=args.batch_size,
    enc_max_len=ENC_MAX_LEN,
    dec_max_len=DEC_MAX_LEN,
    num_workers=args.num_workers,
)
long_ds.prepare_data()

model = Completeformer.load_from_checkpoint(args.checkpoint_path)

trainer = pl.Trainer(
    gpus=torch.cuda.device_count(),
    precision=16,
)

# evaluating using the different test sets
# model.length = "short"
# short_results = trainer.test(
#     model=model,
#     dataloaders=short_ds.test_dataloader()
# )[0]

# with open(OUTPUT_DIR / "short_results.json", "w") as f:
#     json.dump(short_results, f)
# wandb.save(str(OUTPUT_DIR / "short_results.json"))

# model.length = "medium"
# medium_results = trainer.test(
#     model=model,
#     dataloaders=medium_ds.test_dataloader()
# )[0]

# with open(OUTPUT_DIR / "medium_results.json", "w") as f:
#     json.dump(medium_results, f)
# wandb.save(str(OUTPUT_DIR / "medium_results.json"))

model.length = "long"
long_results = trainer.test(
    model=model,
    dataloaders=long_ds.test_dataloader()
)[0]

with open(OUTPUT_DIR / "long_results.json", "w") as f:
    json.dump(long_results, f)
wandb.save(str(OUTPUT_DIR / "long_results.json"))