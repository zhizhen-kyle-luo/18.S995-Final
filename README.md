# LN vs DyT Numerical Diagnostics

Final project for 18.S995, Spring 2026 — Kyle Luo.

A controlled numerical case study comparing core LayerNorm and core Dynamic Tanh maps applied post-hoc to DistilBERT hidden states. See `main.tex` for the writeup.

## Run

```bash
uv sync
uv run python diagnostics.py                                                                          # full paper run -> results/
uv run python diagnostics.py --num_texts 40 --max_tokens 256 --skip_attention --output_dir results_fast  # fast smoke -> results_fast/
```

The fast command uses a separate output directory so it cannot overwrite the canonical paper outputs in `results/`.

Outputs land in `results/`: `metrics_summary.csv`, `low_rank_errors.csv`, `gram_diagonal_summary.csv`, and `fig_*.png`. When attention diagnostics are enabled (default), `attention_perturbation.csv`, `fig_stable_rank_S.png`, and `fig_kappa_eff_S.png` are also written; `--skip_attention` omits them.

## CLI flags

`--model_name`, `--num_texts`, `--batch_size`, `--max_length`, `--max_tokens`, `--alphas` (e.g. `0.25,0.5,1.0`), `--eps`, `--rel_tol`, `--output_dir`, `--device` (`auto`/`cpu`/`cuda`/`mps`), `--skip_attention`, `--text_file`, `--perturb_scales`. Defaults match the paper. Run `--help` for full descriptions.

## Repo

```
diagnostics.py    experiment script
main.tex          paper source
results/          CSVs + figures (auto-created)
```
