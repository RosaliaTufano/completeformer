# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/04_cli.ipynb.

# %% auto 0
__all__ = ['logger', 'URLs', 'MAX_EPOCHS', 'EFFECTIVE_BS', 'LENGTHs', 'POSITION_TYPES', 'download', 'reproduce']

# %% ../nbs/04_cli.ipynb 5
import io
import json
import logging
import requests
import zipfile

from .models import Completeformer
from .train import train as train_model
from fastcore.script import call_parse, Param
from git import Repo
from pathlib import Path
from transformers import AutoTokenizer

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# %% ../nbs/04_cli.ipynb 6
URLs = {
    "completeformer_reproduction_package": "https://zenodo.org/record/4453765/files/tango_reproduction_package.zip",
}

# %% ../nbs/04_cli.ipynb 7
# @call_parse
def _download(
    out_path
):
    """Function for downloading all data and results related to this tool's paper"""
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Downloading and extracting datasets and models to {str(out_path)}.")
    r = requests.get(URLs["completeformer_reproduction_package"])
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(out_path)

# %% ../nbs/04_cli.ipynb 8
@call_parse
def download(
    out_path: Param("The output path to save and unzip all files.", str)
):
    _download(out_path)

# %% ../nbs/04_cli.ipynb 9
def _prep_data():
    short_ds = CompleteformerDataset(
        length="short",
        tokenizer_name="semeru/completeformer_tokenizer",
        batch_size=args.batch_size,
        enc_max_len=ENC_MAX_LEN,
        dec_max_len=DEC_MAX_LEN,
        num_workers=args.num_workers,
    )
    medium_ds = CompleteformerDataset(
        length="medium",
        tokenizer_name="semeru/completeformer_tokenizer",
        batch_size=args.batch_size,
        enc_max_len=ENC_MAX_LEN,
        dec_max_len=DEC_MAX_LEN,
        num_workers=args.num_workers,
    )
    long_ds = CompleteformerDataset(
        length="long",
        tokenizer_name="semeru/completeformer_tokenizer",
        batch_size=args.batch_size,
        enc_max_len=ENC_MAX_LEN,
        dec_max_len=DEC_MAX_LEN,
        num_workers=args.num_workers,
    )
    short_ds.prepare_data()
    medium_ds.prepare_data()
    long_ds.prepare_data()

    return short_ds, medium_ds, long_ds

# %% ../nbs/04_cli.ipynb 10
MAX_EPOCHS = 5
def _train(dataset, length, position_type, grad_accum, output_dir):
    model = Completeformer(
        tokenizer=dataset.tokenizer,
        max_epochs=MAX_EPOCHS,
        length=length,
        position_type=position_type,
        grad_accum=grad_accum
    )
    return train_model(
        model,
        dataset,
        num_epochs=max_epochs,
        output_dir=output_dir,
        name=f"completeformer_{position_type}_{length}"
    )

# %% ../nbs/04_cli.ipynb 11
# python experiment_runner.py \
#         --length long \
#         --position_type rotary \
#         --output_dir /data/models \
#         --batch_size 16
EFFECTIVE_BS = 256
LENGTHs = ["short", "medium", "long"]
POSITION_TYPES = ["sinusoidal", "rotary", "alibi", "relative"]

@call_parse
def reproduce(
    length: str = "short", # The length of the model to train.
    position_type: str = "sinusoidal", # The position encoding type to use.
    batch_size: int = 16, # The batch size to use. The effective batch size will be 256.
):
    """
    Reproduce the results of our paper.
    """

    assert length in LENGTHs + ["all"], f"length must be one of {LENGTHs + ['all']}"
    assert position_type in POSITION_TYPES + ["all"], f"position_type must be one of {POSITION_TYPES + ['all']}"

    # Prepare data
    short_ds, medium_ds, long_ds = _prep_data()

    if length == "short":
        completeformer_ds = short_ds
    elif length == "medium":
        completeformer_ds = medium_ds
    elif length == "long":
        completeformer_ds = long_ds

    # Train model
    grad_accum = EFFECTIVE_BS // batch_size
    model, best_model_path, trainer = _train(
        completeformer_ds,
        length,
        position_type,
        grad_accum,
        output_dir
    )
    ...
