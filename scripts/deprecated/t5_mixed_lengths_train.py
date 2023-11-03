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
NAME = "completeformer-training-t5-mixed-lengths"
NUM_WORKERS = 4
BATCH_SIZE = 4
MAX_LEN = 1024
OUTPUT_DIR = Path("/home/jovyan/data/output")

MIXED_LEN_AVG = 19
MIXED_DS = MaskedDataset(
    max_seq_len=MAX_LEN,
    batch_size=BATCH_SIZE,
    complexity_type="length_mix",
    complexity_avg_len=MIXED_LEN_AVG,
    dataset_config="length_mix",
    tokenizer="ncoop57/completeformer-tokenizer",
    num_workers=NUM_WORKERS
)

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

MIXED_DS.prepare_data()
SHORT_DS.prepare_data()
MED_DS.prepare_data()
LONG_DS.prepare_data()

mixed_model = Completeformer(
    MIXED_DS.tokenizer,
    use_alibi = False,
    **CONFIG
)

mixed_model, best_mixed_model_path, mixed_trainer = train_model(
    mixed_model,
    MIXED_DS,
    num_epochs=CONFIG["max_epochs"],
    output_dir=OUTPUT_DIR / "mixed-length-t5",
    name=NAME + "-mixed-length-t5",
)

# evaluating using the different test sets
mixed_trainer.test(model=mixed_model, dataloaders=MIXED_DS.test_dataloader())
mixed_trainer.test(model=mixed_model, dataloaders=SHORT_DS.test_dataloader())
mixed_trainer.test(model=mixed_model, dataloaders=MED_DS.test_dataloader())
mixed_trainer.test(model=mixed_model, dataloaders=LONG_DS.test_dataloader())
