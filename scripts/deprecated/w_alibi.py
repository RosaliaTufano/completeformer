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
NAME = "completeformer-training-w-alibi"
NUM_WORKERS = 4
BATCH_SIZE = 16
MAX_LEN = 1024
OUTPUT_DIR = Path("/home/jovyan/data/output")

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

simple_model = Completeformer(SIMPLE_DS.tokenizer, use_alibi = True, **CONFIG)

simple_model, best_simple_model_path, simple_trainer = train_model(
    simple_model,
    SIMPLE_DS,
    num_epochs=CONFIG["max_epochs"],
    output_dir=OUTPUT_DIR / "simple-w-alibi",
    name=NAME + "-simple",
)

# evaluating using the different test sets
simple_trainer.test(model=simple_model, dataloaders=SIMPLE_DS.test_dataloader())
simple_trainer.test(model=simple_model, dataloaders=MEDIUM_DS.test_dataloader())
simple_trainer.test(model=simple_model, dataloaders=COMPLEX_DS.test_dataloader())

# medium_model = Completeformer(MEDIUM_DS.tokenizer, use_alibi = True, **CONFIG)

# medium_model, best_medium_model_path, medium_trainer = train_model(
#     medium_model,
#     MEDIUM_DS,
#     num_epochs=CONFIG["max_epochs"],
#     output_dir=OUTPUT_DIR / "medium-w-alibi",
#     name=NAME + "-medium",
# )

# # evaluating using the different test sets
# medium_trainer.test(model=medium_model, dataloaders=SIMPLE_DS.test_dataloader())
# medium_trainer.test(model=medium_model, dataloaders=MEDIUM_DS.test_dataloader())
# medium_trainer.test(model=medium_model, dataloaders=COMPLEX_DS.test_dataloader())

# complex_model = Completeformer(COMPLEX_DS.tokenizer, use_alibi = True, **CONFIG)

# complex_model, best_complex_model_path, complex_trainer = train_model(
#     complex_model,
#     COMPLEX_DS,
#     num_epochs=CONFIG["max_epochs"],
#     output_dir=OUTPUT_DIR / "complex-w-alibi",
#     name=NAME + "-complex",
# )

# # evaluating using the different test sets
# complex_trainer.test(model=complex_model, dataloaders=SIMPLE_DS.test_dataloader())
# complex_trainer.test(model=complex_model, dataloaders=MEDIUM_DS.test_dataloader())
# complex_trainer.test(model=complex_model, dataloaders=COMPLEX_DS.test_dataloader())