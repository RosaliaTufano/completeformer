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
NAME = "completeformer-training-alibi-mixed"
NUM_WORKERS = 4
BATCH_SIZE = 4
MAX_LEN = 1024
OUTPUT_DIR = Path("/home/jovyan/data/output/lr_fix")

MIXED_LEN_AVG = 19
MIXED_DS = MaskedDataset(
    max_seq_len=MAX_LEN,
    batch_size=BATCH_SIZE,
    complexity_type="mix",
    complexity_avg_len=MIXED_LEN_AVG,
    dataset_config="mix",
    tokenizer="ncoop57/completeformer-tokenizer",
    num_workers=NUM_WORKERS
)

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

MIXED_DS.prepare_data()
SIMPLE_DS.prepare_data()
MEDIUM_DS.prepare_data()
COMPLEX_DS.prepare_data()

# checkpoint_path = "/home/jovyan/data/output/simple-wo-alibi/checkpoints/completeformer-epoch=04-val_loss=0.88.ckpt"
# simple_model = Completeformer.load_from_checkpoint(checkpoint_path)

# # evaluating using the different test sets
# simple_trainer.test(model=simple_model, dataloaders=SIMPLE_DS.test_dataloader())
# simple_trainer.test(model=simple_model, dataloaders=MEDIUM_DS.test_dataloader())
# simple_trainer.test(model=simple_model, dataloaders=COMPLEX_DS.test_dataloader())

mixed_model = Completeformer(
    SIMPLE_DS.tokenizer,
    use_alibi = True,
    **CONFIG
)

mixed_model, best_mixed_model_path, mixed_trainer = train_model(
    mixed_model,
    MIXED_DS,
    num_epochs=CONFIG["max_epochs"],
    output_dir=OUTPUT_DIR / "mixed-alibi",
    name=NAME + "-mixed",
)

# evaluating using the different test sets
mixed_trainer.test(model=mixed_model, dataloaders=SIMPLE_DS.test_dataloader())
mixed_trainer.test(model=mixed_model, dataloaders=MEDIUM_DS.test_dataloader())
mixed_trainer.test(model=mixed_model, dataloaders=COMPLEX_DS.test_dataloader())

# medium_model = Completeformer(MEDIUM_DS.tokenizer, use_alibi = False, **CONFIG)

# medium_model, best_medium_model_path, medium_trainer = train_model(
#     medium_model,
#     MEDIUM_DS,
#     num_epochs=CONFIG["max_epochs"],
#     output_dir=OUTPUT_DIR / "medium-wo-alibi",
#     name=NAME + "-medium",
# )

# # evaluating using the different test sets
# medium_trainer.test(model=medium_model, dataloaders=SIMPLE_DS.test_dataloader())
# medium_trainer.test(model=medium_model, dataloaders=MEDIUM_DS.test_dataloader())
# medium_trainer.test(model=medium_model, dataloaders=COMPLEX_DS.test_dataloader())

# complex_model = Completeformer(COMPLEX_DS.tokenizer, use_alibi = False, **CONFIG)

# complex_model, best_complex_model_path, complex_trainer = train_model(
#     complex_model,
#     COMPLEX_DS,
#     num_epochs=CONFIG["max_epochs"],
#     output_dir=OUTPUT_DIR / "complex-wo-alibi",
#     name=NAME + "-complex",
# )

# # evaluating using the different test sets
# complex_trainer.test(model=complex_model, dataloaders=SIMPLE_DS.test_dataloader())
# complex_trainer.test(model=complex_model, dataloaders=MEDIUM_DS.test_dataloader())
# complex_trainer.test(model=complex_model, dataloaders=COMPLEX_DS.test_dataloader())