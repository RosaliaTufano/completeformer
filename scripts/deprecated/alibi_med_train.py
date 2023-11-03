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
NAME = "completeformer-training-alibi-medium"
NUM_WORKERS = 4
BATCH_SIZE = 4
MAX_LEN = 1024
OUTPUT_DIR = Path("/home/jovyan/data/output")

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

MED_DS.prepare_data()

med_model = Completeformer(
    MED_DS.tokenizer,
    use_alibi = True,
    **CONFIG
)

med_model, best_med_model_path, med_trainer = train_model(
    med_model,
    MED_DS,
    num_epochs=CONFIG["max_epochs"],
    output_dir=OUTPUT_DIR / "medium-alibi",
    name=NAME + "-medium",
)

# evaluating using the different test sets
med_trainer.test(model=med_model, dataloaders=MED_DS.test_dataloader())