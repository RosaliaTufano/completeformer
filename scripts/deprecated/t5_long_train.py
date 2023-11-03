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
    "grad_accum": 2,
}
NAME = "completeformer-training-t5-long"
NUM_WORKERS = 4
BATCH_SIZE = 16
MAX_LEN = 1024
OUTPUT_DIR = Path("/home/jovyan/data/output")

LONG_LEN_AVG = 11
LONG_DS = MaskedDataset(
    max_seq_len=MAX_LEN,
    batch_size=BATCH_SIZE,
    complexity_type="long",
    complexity_avg_len=LONG_LEN_AVG,
    dataset_config="length_long",
    tokenizer="ncoop57/completeformer-tokenizer",
    num_workers=NUM_WORKERS
)

LONG_DS.prepare_data()

long_model = Completeformer(
    LONG_DS.tokenizer,
    use_alibi = False,
    **CONFIG
)

long_model, best_long_model_path, long_trainer = train_model(
    long_model,
    LONG_DS,
    num_epochs=CONFIG["max_epochs"],
    output_dir=OUTPUT_DIR / "long-t5",
    name=NAME + "-long",
)

# evaluating using the different test sets
long_trainer.test(model=long_model, dataloaders=LONG_DS.test_dataloader())