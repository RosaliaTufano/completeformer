import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from completeformer.data import MaskedDataset
from completeformer.models import Completeformer
from completeformer.train import train as train_model
from pathlib import Path


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
    "grad_accum": 8,
}
NAME = "completeformer-training-t5-short"
NUM_WORKERS = 4
BATCH_SIZE = 4
MAX_LEN = 1024
OUTPUT_DIR = Path("/home/jovyan/data/output")

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

SHORT_DS.prepare_data()

short_model = Completeformer(
    SHORT_DS.tokenizer,
    use_alibi = False,
    **CONFIG
)

short_model, best_short_model_path, short_trainer = train_model(
    short_model,
    SHORT_DS,
    num_epochs=CONFIG["max_epochs"],
    output_dir=OUTPUT_DIR / "short-t5",
    name=NAME + "-short",
)

# evaluating using the different test sets
short_trainer.test(model=short_model, dataloaders=SHORT_DS.test_dataloader())