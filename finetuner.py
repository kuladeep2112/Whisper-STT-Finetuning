#!/usr/bin/env python3
"""
trainer.py
----------
Whisper large-v3-turbo LoRA fine-tune. No arguments - edit CONFIG and run:

    tmux new -s whisper
    conda activate agentic_env
    cd ~/cloudfiles/code/Users/kuladeep.a/STT_finetuning
    mkdir -p logs
    python -u trainer.py 2>&1 | tee logs/train_$(date +%m%d_%H%M).log

    detach   Ctrl-b then d
    reattach tmux attach -t whisper

Loads the dataset already built by the notebook (whisper_ds/).
Resumable: re-run the same command and it continues from the last checkpoint.

Sized for a 16 GB T4:
  fp16 base + gradient checkpointing (use_reentrant=False) + batch 2 x accum 8.
"""

import os
# must precede the torch import
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import (EarlyStoppingCallback, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments,
                          WhisperForConditionalGeneration, WhisperProcessor)

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None


# =============================== CONFIG ====================================
CONFIG = {
    "base":    "/home/azureuser/cloudfiles/code/Users/kuladeep.a/"
               "STT_finetuning/stt_audio",
    "ds_dir":  "whisper_ds",
    "out_dir": "whisper_lora",
    "model":   "openai/whisper-large-v3-turbo",

    # ---- LoRA ------------------------------------------------------------
    "rank":           64,
    "lora_alpha":     128,
    "lora_dropout":   0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],

    # ---- training (batch 2 x accum 8 = effective 16, fits 16 GB) ----------
    "batch_size":  2,
    "accum":       8,
    "lr":          1e-4,
    "epochs":      5,
    "warmup":      60,
    "log_steps":   5,
    "eval_steps":  20,
    "patience":    3,          # stop after N evals with no improvement
    "seed":        3407,
}
# ===========================================================================


P    = Path(CONFIG["base"])
DS   = P / CONFIG["ds_dir"]
OUT  = P / CONFIG["out_dir"]


def show(rows, headers):
    if tabulate:
        print(tabulate(rows, headers=headers, tablefmt="github"), flush=True)
    else:
        print("  ".join(headers), flush=True)
        for r in rows:
            print("  ".join(str(x) for x in r), flush=True)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], Any]]]):
        inputs = [{"input_features": np.asarray(f["input_features"],
                                                dtype=np.float32)}
                  for f in features]
        batch = self.processor.feature_extractor.pad(inputs, return_tensors="pt")
        lbl = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features], return_tensors="pt")
        labels = lbl["input_ids"].masked_fill(lbl.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("dataset : {}".format(DS))
    print("output  : {}".format(OUT))
    print("gpu     : {}  ({:.1f} GB)".format(
        torch.cuda.get_device_name(0),
        torch.cuda.get_device_properties(0).total_memory / 1e9))
    print("=" * 70, flush=True)

    # ------------------------------------------------------------- dataset
    if not DS.exists():
        raise FileNotFoundError(
            "dataset not found: {}\nBuild it in the notebook first.".format(DS))
    ds = load_from_disk(str(DS))
    n_tr, n_va = len(ds["train"]), len(ds["validation"])

    eff = CONFIG["batch_size"] * CONFIG["accum"]
    steps_per_epoch = max(n_tr // eff, 1)
    total_steps = steps_per_epoch * CONFIG["epochs"]

    # --------------------------------------------------------------- model
    processor = WhisperProcessor.from_pretrained(
        CONFIG["model"], language="English", task="transcribe")
    tk = processor.tokenizer
    if tk.pad_token is None:                  # Whisper pads with <|endoftext|>
        tk.pad_token = tk.eos_token
        tk.pad_token_id = tk.convert_tokens_to_ids(tk.eos_token)

    model = WhisperForConditionalGeneration.from_pretrained(
        CONFIG["model"],
        dtype               = torch.float16,   # T4 is Turing: fp16, never bf16
        device_map          = {"": 0},
        attn_implementation = "sdpa",          # pre-Ampere, no flash-attn 2
    )
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.pad_token_id = tk.pad_token_id
    model.config.use_cache = False
    model.generation_config.language = "en"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    decoder_start = model.config.decoder_start_token_id   # capture before peft
    assert processor.feature_extractor.feature_size == 128, "expected 128 mel bins"

    # ---------------------------------------------------------------- LoRA
    for p in model.parameters():              # fp16 base: freeze manually
        p.requires_grad_(False)

    # use_reentrant=False is REQUIRED. Whisper's encoder receives no
    # grad-requiring input, so reentrant checkpointing silently yields a
    # detached loss ("element 0 of tensors does not require grad").
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    model = get_peft_model(model, LoraConfig(
        r              = CONFIG["rank"],
        lora_alpha     = CONFIG["lora_alpha"],
        lora_dropout   = CONFIG["lora_dropout"],
        bias           = "none",
        target_modules = CONFIG["target_modules"],
    ))
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all   = sum(p.numel() for p in model.parameters())

    show([["train examples",  "{:,}".format(n_tr)],
          ["val examples",    "{:,}".format(n_va)],
          ["rank / alpha",    "{} / {}".format(CONFIG["rank"], CONFIG["lora_alpha"])],
          ["trainable",       "{:,}  ({:.2f}%)".format(n_train, 100*n_train/n_all)],
          ["params/example",  "{:,.0f}".format(n_train / max(n_tr, 1))],
          ["effective batch", eff],
          ["steps/epoch",     steps_per_epoch],
          ["total steps",     total_steps],
          ["train logs",      "{} lines (every {} steps)".format(
              total_steps // CONFIG["log_steps"], CONFIG["log_steps"])],
          ["evals",           "{} (every {} steps)".format(
              total_steps // CONFIG["eval_steps"], CONFIG["eval_steps"])],
          ["learning rate",   CONFIG["lr"]]],
         ["setup", "value"])

    collator = DataCollatorSpeechSeq2SeqWithPadding(processor, decoder_start)

    # ------------------------------------------------- smoke: grads + vram
    # Catches the two failures that otherwise waste a full trainer startup:
    # a detached loss, and OOM.
    b = {k: v.to("cuda") for k, v in next(iter(DataLoader(
            ds["train"], batch_size=CONFIG["batch_size"],
            collate_fn=collator))).items()}

    # autocast mirrors what Seq2SeqTrainer does with fp16=True; without it the
    # fp32 input_features hit fp16 conv weights and raise a dtype error
    with torch.autocast("cuda", dtype=torch.float16):
        loss = model(**b).loss

    if not loss.requires_grad:
        raise RuntimeError("loss has no grad_fn - LoRA or checkpointing is "
                           "misconfigured. Aborting before wasting GPU hours.")
    loss.backward()
    ngrad = sum(1 for _, p in model.named_parameters()
                if p.requires_grad and p.grad is not None)
    peak = torch.cuda.max_memory_allocated() / 1e9
    model.zero_grad(set_to_none=True)
    del b, loss
    torch.cuda.empty_cache()
    print("\nsmoke ok: {} tensors have gradients | peak {:.2f} GB of {:.1f}"
          .format(ngrad, peak,
                  torch.cuda.get_device_properties(0).total_memory / 1e9),
          flush=True)

    # --------------------------------------------------------------- train
    args = Seq2SeqTrainingArguments(
        output_dir                    = str(OUT),
        per_device_train_batch_size   = CONFIG["batch_size"],
        gradient_accumulation_steps   = CONFIG["accum"],
        per_device_eval_batch_size    = CONFIG["batch_size"],
        learning_rate                 = CONFIG["lr"],
        warmup_steps                  = CONFIG["warmup"],
        num_train_epochs              = CONFIG["epochs"],
        fp16                          = True,
        bf16                          = False,
        max_grad_norm                 = 1.0,   # fp16 Whisper is spike-prone
        optim                         = "adamw_8bit",
        gradient_checkpointing        = True,
        gradient_checkpointing_kwargs = {"use_reentrant": False},
        logging_steps                 = CONFIG["log_steps"],
        eval_strategy                 = "steps",
        eval_steps                    = CONFIG["eval_steps"],
        save_strategy                 = "steps",
        save_steps                    = CONFIG["eval_steps"],
        save_total_limit              = 3,
        load_best_model_at_end        = True,
        metric_for_best_model         = "eval_loss",
        greater_is_better             = False,
        predict_with_generate         = False,
        report_to                     = "none",
        remove_unused_columns         = False,  # input_features is custom
        label_names                   = ["labels"],
        dataloader_num_workers        = 2,
        seed                          = CONFIG["seed"],
    )

    trainer = Seq2SeqTrainer(
        model         = model,
        args          = args,
        train_dataset = ds["train"],
        eval_dataset  = ds["validation"],
        data_collator = collator,
        callbacks     = [EarlyStoppingCallback(
            early_stopping_patience=CONFIG["patience"])],
    )
    trainer.model_accepts_loss_kwargs = False   # Whisper.forward has no **kwargs

    resume = any(OUT.glob("checkpoint-*"))
    if resume:
        print("\nresuming from last checkpoint in {}".format(OUT), flush=True)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    stats = trainer.train(resume_from_checkpoint=resume)

    # -------------------------------------------------------------- report
    hist = trainer.state.log_history
    curve = []
    for h in hist:
        if "eval_loss" in h:
            tr = next((x["loss"] for x in reversed(hist)
                       if "loss" in x and x["step"] <= h["step"]), None)
            curve.append([h["step"],
                          round(tr, 4) if tr is not None else "",
                          round(h["eval_loss"], 4)])
    if curve:
        print()
        show(curve, ["step", "train loss", "eval loss"])
        best = min(curve, key=lambda r: r[2])
        print("\nbest eval loss {} at step {} of {}".format(
            best[2], best[0], total_steps))
        if best[0] < total_steps * 0.5:
            print("NOTE: best eval was in the first half of training - the model "
                  "started overfitting early. A lower rank would likely do better.")

    print("\nruntime   : {:.1f} min".format((time.time() - t0) / 60))
    print("final loss: {:.4f}".format(stats.metrics["train_loss"]))
    print("peak vram : {:.2f} GB".format(torch.cuda.max_memory_allocated() / 1e9))

    final = OUT / "final"
    model.save_pretrained(str(final))
    processor.save_pretrained(str(final))
    (OUT / "run_config.json").write_text(json.dumps(CONFIG, indent=2))
    print("\nadapter saved -> {}".format(final), flush=True)


if __name__ == "__main__":
    main()