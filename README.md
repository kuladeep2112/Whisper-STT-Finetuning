# Domain-Adaptive Speech-to-Text (STT) for Vehicle Damage Assessment

Fine-tuning **Whisper large-v3-turbo** to transcribe vehicle damage-assessment and
insurance-inspection speech while preserving domain terminology.

---

## Objective

General ASR mis-recognizes automotive and insurance terms (`quarter panel`,
`salvage title`, `total loss`, `control arm`, `rear-ended`), which corrupts any
downstream use of the transcript. This project fine-tunes Whisper large-v3-turbo
on domain audio so those terms survive transcription, and builds the pipeline that
turns inspection audio into **timestamped transcripts usable as weak labels** for
the parts/damage vision and VLM stacks (the data flywheel).

This repo is the **English proof-of-concept** that establishes the end-to-end
pipeline. The production target is a different acoustic/accent domain, reached
later by swapping the training corpus — not the pipeline.

**Design rule:** the transcript is a *label for the audio* and must match what was
spoken. LLM cleaning only fixes mis-recognized domain terms (`quarter pale →
quarter panel`); it never fixes grammar or removes disfluencies, since that would
produce labels that no longer match the waveform.

---

## Dataset

| Artifact | What | Location |
|---|---|---|
| Source audio | 86 sourced YouTube videos, ~12.8 h speech | Azure Blob `stt_audio/audio/` *(internal)* |
| Cleaned transcripts | Whisper output + LLM domain filter/correction | `stt_audio/transcripts_clean/` |
| Fine-tune dataset | ≤30 s chunks, 128-mel features + labels | `stt_audio/whisper_ds/` (train + validation) |
| Test set | 6 manually-verified clips | `stt_audio/test_set/` |
| Base model | Whisper large-v3-turbo | [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) |

> Source audio is scraped from public YouTube for internal research only; it is
> **not redistributed**. Provenance is recorded per item. Update the Blob paths /
> internal registry link for your environment.

---
## End to End workflow
![alt text](<Untitled diagram-2026-08-06-013846.png>)

---


## Setup

```bash
# torch from the CUDA-matched index first
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Set an Anthropic key for the LLM cleaning stage (`CLAUDE` in Colab secrets, or
`ANTHROPIC_API_KEY` env var).

---

## Pipeline

| Phase | Notebook / script | Produces |
|---|---|---|
| 1 · Sourcing | `data_sourcing.ipynb` | `stt_audio/audio/*.mp3` |
| 2 · Generation | `data_generation.ipynb` | filtered + corrected transcript JSONs |
| 3 · Dataset build | `STT_finetuning.ipynb` | `whisper_ds/` Arrow dataset |
| 4 · Fine-tune | `trainer.py` | LoRA adapters / checkpoints |
| 5 · Evaluation | `Evaluation_testing.ipynb` | `eval_out/REPORT.md`, diffs |

Run `trainer.py` headless:

```bash
tmux new -s whisper
conda activate agentic_env
python -u trainer.py 2>&1 | tee logs/train_$(date +%m%d_%H%M).log
```

---

## Results

Best checkpoint **r64:80** vs baseline (full tables in `eval_out/REPORT.md`):

| metric | baseline | r64:80 | change |
|---|---|---|---|
| TEST micro WER | 0.0331 | **0.0259** | **−21.8%** |
| TEST substitutions | 56 | 59 | +5.4% |
| TEST insertions | 110 | 65 | −40.9% |
| VAL micro WER | 0.0736 | **0.0629** | **−14.5%** |
| VAL deletions | 484 | 314 | −35.1% |

**Read the result honestly:** WER fell, but **substitutions — the only error type
that measures recognition — rose on both sets.** The gain is *decoder stability*
(fewer repetition loops on test, less early truncation on val), not better hearing.
Baseline entity recall on the domain glossary was already **1.00**, i.e. no lexical
headroom on clean English audio. Rank 64 matched rank 512 at 1/8 the trainable
parameters; rank 16 underfit.

---

## Next steps

- **Acoustic/accent domain** — the real target; needs in-domain (noisy, accented)
  audio where the baseline is not already perfect.
- **Inference-time loop suppression** (`repetition_penalty`) may capture the same
  gain with no training — test before investing in more fine-tuning.
- **Human-corrected labels** — current labels are Whisper + LLM correction
  (near self-distillation); a small human-verified set is the highest-leverage
  data investment.

See [`STT_domain_adaptation_wiki.md`](STT_domain_adaptation_wiki.md) for the full
workflow, notebook-by-notebook detail, folder structure, and schema.
