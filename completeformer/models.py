# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/01_models.ipynb.

# %% auto 0
__all__ = ['get_grouped_params', 'Completeformer']

# %% ../nbs/01_models.ipynb 5
import evaluate
import torch
import wandb

import pandas as pd
import pytorch_lightning as pl

# from datasets import load_metric
from pathlib import Path
from transformers import AdamW, get_cosine_schedule_with_warmup
from x_transformers import XTransformer
from x_transformers.autoregressive_wrapper import top_p

# %% ../nbs/01_models.ipynb 7
# Code from: https://github.com/huggingface/transformers/blob/master/examples/research_projects/codeparrot/scripts/codeparrot_training.py#L113
def get_grouped_params(model, weight_decay, no_decay=["bias", "LayerNorm.weight"]):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]

# %% ../nbs/01_models.ipynb 8
class Completeformer(pl.LightningModule):
    """
        Completeformer is a T5 based encoder-decoder model that uses the Alibi positional embedding heuristic for the decoder.
    """
    def __init__(
        self,
        tokenizer,
        max_epochs,
        length,
        position_type="sinusoidal",
        dim=512,
        enc_max_len=1024,
        enc_layers=6,
        enc_heads=8,
        dec_max_len=128,
        dec_layers=6,
        dec_heads=8,
        lr=1e-4,
        num_warmup_steps=2_000,
        weight_decay=0.1,
        grad_accum=1,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.length = length
        self.position_type = position_type
        assert self.position_type in ["sinusoidal", "rotary", "alibi", "relative", "dynamic"], f"Position type {self.position_type} not supported."
        if self.position_type == "sinusoidal":
            self.model = XTransformer(
                dim = dim,
                tie_token_embeds = True,
                return_tgt_loss = True,
                use_abs_pos_emb = False,
                scaled_sinu_pos_emb = True,
                enc_scaled_sinu_pos_emb = True,
                enc_num_tokens = len(tokenizer),
                enc_depth = enc_layers,
                enc_heads = enc_heads,
                enc_max_seq_len = enc_max_len,
                dec_num_tokens = len(tokenizer),
                dec_depth = dec_layers,
                dec_heads = dec_heads,
                dec_max_seq_len = dec_max_len,
                dec_scaled_sinu_pos_emb = True
            )
        elif self.position_type == "rotary":
            self.model = XTransformer(
                dim = dim,
                tie_token_embeds = True,
                return_tgt_loss = True,
                enc_rotary_pos_emb = True,
                enc_num_tokens = len(tokenizer),
                enc_depth = enc_layers,
                enc_heads = enc_heads,
                enc_max_seq_len = enc_max_len,
                dec_num_tokens = len(tokenizer),
                dec_depth = dec_layers,
                dec_heads = dec_heads,
                dec_max_seq_len = dec_max_len,
                dec_rotary_xpos = True
            )
        elif self.position_type == "alibi":
            self.model = XTransformer(
                dim = dim,
                tie_token_embeds = True,
                return_tgt_loss = True,
                enc_alibi_pos_emb = True,
                enc_num_tokens = len(tokenizer),
                enc_depth = enc_layers,
                enc_heads = enc_heads,
                enc_max_seq_len = enc_max_len,
                dec_num_tokens = len(tokenizer),
                dec_depth = dec_layers,
                dec_heads = dec_heads,
                dec_max_seq_len = dec_max_len,
                dec_alibi_pos_emb = True
            )
        elif self.position_type == "relative":
            self.model = XTransformer(
                dim = dim,
                tie_token_embeds = True,
                return_tgt_loss = True,
                enc_rel_pos_bias = True,
                enc_num_tokens = len(tokenizer),
                enc_depth = enc_layers,
                enc_heads = enc_heads,
                enc_max_seq_len = enc_max_len,
                dec_num_tokens = len(tokenizer),
                dec_depth = dec_layers,
                dec_heads = dec_heads,
                dec_max_seq_len = dec_max_len,
                dec_rel_pos_bias = True
            )
        elif self.position_type == "dynamic":
            self.model = XTransformer(
                dim = dim,
                tie_token_embeds = True,
                return_tgt_loss = True,
                enc_dynamic_pos_bias = True,
                enc_dynamic_pos_bias_log_distance = False,
                enc_num_tokens = len(tokenizer),
                enc_depth = enc_layers,
                enc_heads = enc_heads,
                enc_max_seq_len = enc_max_len,
                dec_num_tokens = len(tokenizer),
                dec_depth = dec_layers,
                dec_heads = dec_heads,
                dec_max_seq_len = dec_max_len,
                dec_dynamic_pos_bias = True,
                dec_dynamic_pos_bias_log_distance = False
            )
        self.lr = lr
        self.max_epochs = max_epochs
        self.num_warmup_steps = num_warmup_steps
        self.weight_decay = weight_decay
        self.grad_accum = grad_accum

        # Get metrics for testing
        self.bleu_metric = evaluate.load("bleu")
        self.chrf_metric = evaluate.load("chrf")
        self.em_metric = evaluate.load("exact_match")
        self.leven_dist_metric = evaluate.load("ncoop57/levenshtein_distance")
        self.meteor_metric = evaluate.load("meteor")
        self.rouge_metric = evaluate.load("rouge")

        # Ignore padding token in loss calculation
        self.model.decoder.ignore_index = tokenizer.pad_token_id

        self.save_hyperparameters()

    def forward(self, srcs, tgts, srcs_mask):
        return self.model(srcs, tgts, mask=srcs_mask)

    def on_train_start(self):
        # Create a table to store the generated samples
        self.table = wandb.Table(data=[], columns=["input", "completion", "step"])

    def on_train_end(self):
        # Save the generated samples
        self.logger.experiment.log({"generated_samples": self.table})

    def training_step(self, batch, batch_idx):
        srcs = batch["input_ids"]
        tgts = batch["labels"]

        srcs_mask = batch["attention_mask"].bool()

        loss = self(srcs, tgts, srcs_mask)
        self.log(
            "trn_loss", 
            loss,
            on_step=True,
            on_epoch=True,
            logger=True
        )
        opt = self.optimizers()
        lr = opt.param_groups[0]["lr"]
        self.log("lr", lr, on_step=True, on_epoch=True, logger=True)

        return loss

    def training_epoch_end(self, training_step_outputs):
        # Generate a sample and add it to the table
        text = "def bubbleSort(arr): <TAB>n = len(arr) <TAB><MASK>"
        prediction = self.generate(text, self.tokenizer)
        completion = text.replace("<MASK>", prediction)
        self.table.add_data(text, completion, self.global_step)

    def validation_step(self, batch, batch_idx):
        srcs = batch["input_ids"]
        tgts = batch["labels"]

        srcs_mask = batch["attention_mask"].bool()
        loss = self(srcs, tgts, srcs_mask)
        self.log("val_loss", loss, on_epoch=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        srcs = batch["input_ids"].to(self.device)
        tgts = batch["labels"].to(self.device)

        srcs_mask = batch["attention_mask"].bool().to(self.device)
        loss = self(srcs, tgts, srcs_mask)
        self.log("tst_loss", loss, on_epoch=True, logger=True)

        num_tokens = torch.sum(tgts != self.tokenizer.pad_token_id, dim=1)
        max_tokens = torch.max(num_tokens).item() - 1 # -1 for the BOS token
        start_tokens = (torch.ones((batch["input_ids"].shape[0], 1)) * self.tokenizer.bos_token_id).long().to(self.device)

        samples = self.model.generate(srcs, start_tokens, max_tokens, mask=srcs_mask, filter_logits_fn=top_p)
        new_samples, new_tgts = [], []
        for i in range(0, num_tokens.shape[0]):
            # Trim samples and targets to EOS token
            sample_eos = torch.where(samples[i] == self.tokenizer.eos_token_id)[0]
            tgt_eos = torch.where(tgts[i] == self.tokenizer.eos_token_id)[0]
            if sample_eos.shape[0] > 0:
                new_sample = samples[i][:sample_eos[0]]
            else:
                new_sample = samples[i]
            
            if tgt_eos.shape[0] > 0:
                new_tgt = tgts[i][:tgt_eos[0]]
            else:
                new_tgt = tgts[i]
            
            new_samples.append(new_sample.tolist())
            new_tgts.append(new_tgt.tolist())

        decoded_preds = self.tokenizer.batch_decode(new_samples, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(new_tgts, skip_special_tokens=True)
        self.bleu_metric.add_batch(
            predictions=decoded_preds,
            references=decoded_labels,
        )
        self.chrf_metric.add_batch(
            predictions=decoded_preds,
            references=decoded_labels
        )
        self.em_metric.add_batch(
            predictions=decoded_preds,
            references=decoded_labels
        )
        self.leven_dist_metric.add_batch(
            predictions=decoded_preds,
            references=decoded_labels,
        )
        self.meteor_metric.add_batch(
            predictions=decoded_preds,
            references=decoded_labels
        )
        self.rouge_metric.add_batch(
            predictions=decoded_preds,
            references=decoded_labels,
        )

    def test_epoch_end(self, training_step_outputs):
        bleu_score = self.bleu_metric.compute(tokenizer=lambda x: self.tokenizer.tokenize(x))
        self.log(f"bleu_{self.length}", bleu_score["bleu"], on_epoch=True, logger=True)

        chrf_score = self.chrf_metric.compute()
        self.log(f"chrf_{self.length}", chrf_score["score"], on_epoch=True, logger=True)

        em_score = self.em_metric.compute()
        self.log(f"exact_match_{self.length}", em_score["exact_match"], on_epoch=True, logger=True)

        leven_dist_score = self.leven_dist_metric.compute(tokenizer=lambda x: self.tokenizer.tokenize(x), normalize=True)
        self.log(f"leven_dist_{self.length}",  leven_dist_score["levenshtein_distance"], on_epoch=True, logger=True)

        meteor_score = self.meteor_metric.compute()
        self.log(f"meteor_{self.length}", meteor_score["meteor"], on_epoch=True, logger=True)

        rouge_score = self.rouge_metric.compute(tokenizer=lambda x: self.tokenizer.tokenize(x))
        self.log(f"rouge_{self.length}", rouge_score["rougeL"], on_epoch=True, logger=True)

    def configure_optimizers(self):
        # Setup the Adam optimizer with a Cosine LR scheduler with warm restarts
        optimizer = AdamW(get_grouped_params(self.model, self.weight_decay), lr=self.lr)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.total_steps(),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1
            },
        }
    
    def total_steps(self) -> int:
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        dataset_size = len(self.trainer.datamodule.train_dataloader())
        return dataset_size * self.hparams.max_epochs

    def generate(self, text, tokenizer, num_tokens=32, decode=True):
        """
            Generate a sample from the model.

            Args:
                text (str): The text to generate from.
                tokenizer (Tokenizer): The tokenizer to use.
                num_tokens (int): The number of tokens to generate.
                decode (bool): Whether to decode the output.
            Returns:
                sample (str or torch.Tensor): The generated sample as a string or list of tokens (if decode=True).
        """
        self.eval()
        t = tokenizer(text, return_tensors="pt")
        src = t["input_ids"].to(self.device)
        src_mask = t["attention_mask"].bool().to(self.device)
        start_tokens = (torch.ones((1, 1)) * tokenizer.bos_token_id).long().to(self.device)
        sample = self.model.generate(src, start_tokens, num_tokens, mask=src_mask, filter_logits_fn=top_p)

        return tokenizer.decode(sample[0], skip_special_tokens=True) if decode else sample[0]
