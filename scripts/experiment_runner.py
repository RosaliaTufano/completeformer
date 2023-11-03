import argparse
import json
import os
import wandb

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from completeformer.data import CompleteformerDataset
from completeformer.models import Completeformer
from completeformer.train import train as train_model
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=str, default="long")
    parser.add_argument("--language", type=str, default="python")
    parser.add_argument("--position_type", type=str, default="alibi")
    parser.add_argument("--output_dir", type=str, default="/semeru/completeformer/models")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=os.cpu_count())

    return parser.parse_args()

args = parse_args()

NAME = f"completeformer_{args.position_type}_{args.length}_{args.language}"
EFFECTIVE_BS = 256
grad_accum = EFFECTIVE_BS // args.batch_size
ENC_MAX_LEN = 1024
DEC_MAX_LEN = 128
OUTPUT_DIR = Path(args.output_dir) / f"{args.position_type}_{args.length}_{args.language}"
CONFIG = {
    "position_type": args.position_type,
    "length": args.length,
    "dec_heads": 8,
    "dec_layers": 6,
    "dec_max_len": DEC_MAX_LEN,
    "dim": 512,
    "enc_heads": 8,
    "enc_layers": 6,
    "enc_max_len": ENC_MAX_LEN,
    "lr": 0.0001,
    "max_epochs": 5, # This will need to be tweaked based on our compute constraints
    "num_warmup_steps": 2000,
    "grad_accum": grad_accum,
}

tokenizer_name = "semeru/completeformer_tokenizer" if args.language == "python" else "semeru/completeformer_java"

short_ds = CompleteformerDataset(
    length="short",
    language=args.language,
    tokenizer_name=tokenizer_name,
    batch_size=args.batch_size,
    enc_max_len=ENC_MAX_LEN,
    dec_max_len=DEC_MAX_LEN,
    num_workers=args.num_workers,
)
short_ds.prepare_data()
medium_ds = CompleteformerDataset(
    length="medium",
    language=args.language,
    tokenizer_name=tokenizer_name,
    batch_size=args.batch_size,
    enc_max_len=ENC_MAX_LEN,
    dec_max_len=DEC_MAX_LEN,
    num_workers=args.num_workers,
)
medium_ds.prepare_data()
long_ds = CompleteformerDataset(
    length="long",
    language=args.language,
    tokenizer_name=tokenizer_name,
    batch_size=args.batch_size,
    enc_max_len=ENC_MAX_LEN,
    dec_max_len=DEC_MAX_LEN,
    num_workers=args.num_workers,
)
long_ds.prepare_data()
mix_ds = CompleteformerDataset(
    length="mix",
    language=args.language,
    tokenizer_name=tokenizer_name,
    batch_size=args.batch_size,
    enc_max_len=ENC_MAX_LEN,
    dec_max_len=DEC_MAX_LEN,
    num_workers=args.num_workers,
)
mix_ds.prepare_data()


if args.length == "short":
    completeformer_ds = short_ds
elif args.length == "medium":
    completeformer_ds = medium_ds
elif args.length == "long":
    completeformer_ds = long_ds
elif args.length == "mix":
    completeformer_ds = mix_ds

model = Completeformer(
    completeformer_ds.tokenizer,
    **CONFIG
)
model, best_model_path, trainer = train_model(
    model,
    completeformer_ds,
    num_epochs=CONFIG["max_epochs"],
    output_dir=OUTPUT_DIR,
    name=NAME
)

# evaluating using the different test sets
model.length = "short"
short_results = trainer.test(
    model=model,
    dataloaders=short_ds.test_dataloader()
)[0]

with open(OUTPUT_DIR / "short_results.json", "w") as f:
    json.dump(short_results, f)
wandb.save(str(OUTPUT_DIR / "short_results.json"))

model.length = "medium"
medium_results = trainer.test(
    model=model,
    dataloaders=medium_ds.test_dataloader()
)[0]

with open(OUTPUT_DIR / "medium_results.json", "w") as f:
    json.dump(medium_results, f)
wandb.save(str(OUTPUT_DIR / "medium_results.json"))

model.length = "long"
long_results = trainer.test(
    model=model,
    dataloaders=long_ds.test_dataloader()
)[0]

with open(OUTPUT_DIR / "long_results.json", "w") as f:
    json.dump(long_results, f)
wandb.save(str(OUTPUT_DIR / "long_results.json"))

model.length = "mix"
mix_results = trainer.test(
    model=model,
    dataloaders=mix_ds.test_dataloader()
)[0]

with open(OUTPUT_DIR / "mix_results.json", "w") as f:
    json.dump(mix_results, f)
wandb.save(str(OUTPUT_DIR / "mix_results.json"))