# LN vs DyT Numerical Diagnostics

Spring 2026 Final project for 18.S995 by Kyle Luo.

A controlled numerical case study comparing core LayerNorm and core Dynamic Tanh maps applied post-hoc to `distilbert-base-uncased` hidden states. See `main.tex` for the writeup.

By default, the input text is `data/alice_120.txt`: 120 sentence-like segments extracted from the public-domain Project Gutenberg text of Lewis Carroll's *Alice's Adventures in Wonderland*. Use `--text_file` to run the same diagnostics on another one-sentence-per-line text file.

## Run

Requires Python 3.12 or newer and `uv`.

```bash
uv sync
uv run python diagnostics.py                                                                          # full run -> results/
uv run python diagnostics.py --num_texts 40 --max_tokens 256 --skip_attention --output_dir results_fast  # fast smoke -> results_fast/
```

The fast command uses a separate output directory so it cannot overwrite the canonical paper outputs in `results/`.

Outputs land in `--output_dir` (`results/` by default): `metrics_summary.csv`, `low_rank_errors.csv`, `gram_diagonal_summary.csv`, and `fig_*.png`. With the default attention diagnostics enabled, the run also writes `attention_perturbation.csv`, `fig_stable_rank_S.png`, and `fig_kappa_eff_S.png`; `--skip_attention` omits the attention-only outputs.

## CLI flags

`--model_name`, `--num_texts`, `--batch_size`, `--max_length`, `--max_tokens`, `--alphas` (e.g. `0.25,0.5,1.0`), `--eps`, `--rel_tol`, `--output_dir`, `--device` (`auto`/`cpu`/`cuda`/`mps`), `--skip_attention`, `--text_file`, `--perturb_scales`. Defaults match the paper. Run `--help` for full descriptions.

## Repo

```
diagnostics.py    experiment script
data/             fixed text input used by the paper defaults
main.tex          paper source
results/          canonical CSVs + figures from the default output path
```
