# DS-GA 1011 Assignment 4: Fine-tuning Language Models

Starter code for Assignment 4. See the assignment writeup (`main.tex`/`main.pdf` in the course
handout) for the full problem statement.

Part 1 (BERT fine-tuning for sentiment classification) and Part 2 (T5 fine-tuning for text-to-SQL)
pin different, incompatible package versions, so set up a **separate environment for each part** —
see `part1/README.md` and `part2/README.md` for setup instructions specific to each part.

You will require GPUs for this homework. You can either use the HPC server (see documentation
[here](https://github.com/Athul-R/NYU-HPC-Tutorials)) or use an API service like
[lightning.ai](https://lightning.ai/) or other cloud providers.

## Layout

- `part1/`: starter code for Part 1 (`main.py`, `utils.py`). You will modify `main.py` and `utils.py`.
- `part2/`: starter code for Part 2 (`train_t5.py`, `t5_utils.py`, `utils.py`, `load_data.py`,
  `evaluate.py`), the text-to-SQL `data/`, the ground-truth dev records in `records/`, and a
  `results/` directory where you should save your model's output SQL/records for submission.

## Reproducing the reported runs

Use a fresh checkout and the environment for the part you are running. The commands below are relative to the indicated part directory.

### Part 1 (IMDB)

Run these commands from `part1/`. Each trained model is evaluated on both the original and the transformed test set.

```bash
# Q1 trains and evaluates on original reviews. Q2 evaluates that model on reviews with typos.
python main.py --train --eval --learning_rate 5e-5 --batch_size 8 --num_epochs 3
python main.py --eval_transformed --model_dir out

# Q3 trains on augmented reviews and evaluates the same checkpoint on both test sets.
python main.py --train_augmented --learning_rate 2e-5 --batch_size 8 --num_epochs 3
python main.py --eval --model_dir out_augmented
python main.py --eval_transformed --model_dir out_augmented
```

Both models start from `bert-base-cased` and use seed 0, AdamW and linear decay with zero warmup steps. The typo probability is `p=0.4` for alphabetic words of at least four letters. Q3 adds 5,000 reviews sampled with seed 42 to the 25,000 original training reviews. The Q3 learning-rate comparison reused saved tokenized inputs for identical typos in both runs. The commands above regenerate the inputs in a fresh checkout. The `5e-5` augmented run scored 50.00% on both tests. The final `2e-5` run scored 93.760% on the original test and 90.904% on the typo test.

### Part 2 (Q7 fine-tuning)

From `part2/`, the training configuration can be passed to the existing entry point. The original job set seed 42 before starting training.

```bash
python - --finetune --experiment_name experiment \
  --optimizer_type AdamW --learning_rate 1e-4 --weight_decay 0 \
  --scheduler_type linear --num_warmup_epochs 1 --max_n_epochs 30 \
  --patience_epochs 5 --batch_size 16 --test_batch_size 16 <<'PY'
import train_t5
from utils import set_random_seeds

set_random_seeds(42)
train_t5.main()
PY
```

The learning-rate schedule spans 30 epochs. Dev Record F1 peaked at epoch 25 (71.36%), and this checkpoint generated the 432 Q7 test queries. Epochs 26–30 resumed from epoch 25 with a fresh AdamW state after the job reached its time limit. Dev and test generation use four beams, a limit of 512 new tokens, and no sampling. `decode_sql_batch` restores the SQL `<` operator when decoding.

The Q7 Google Drive folder contains the epoch-25 checkpoint in four `.safetensors` weight files, plus `model.safetensors.index.json`, `config.json`, and `generation_config.json`. Download all seven files into one directory and load it with `T5ForConditionalGeneration.from_pretrained(checkpoint_dir)`. All 131 saved tensors match the original checkpoint byte for byte. The tokenizer is `google-t5/t5-small`.
