# Pruning Qwen3.5-4B directly in GGUF

I removed 1,024 of 9,216 MLP channels per layer from a Qwen3.5-4B Q4_K_M GGUF. Each cut removes the corresponding rows from the gate and up projections and the matching columns from the down projection. Groups contain 256 consecutive channels, so the exporter copies the surviving quantized blocks directly. It updates the MLP width in the GGUF metadata and preserves the original tensor types. Both artifacts load in stock llama.cpp at revision `cf659f240`.

## Results

| Variant | MLP width | GGUF size | Exact match | Median decode speed |
| --- | ---: | ---: | ---: | ---: |
| Original | 9,216 | 2,614.0 MiB | 22/24 | 16.28 tok/s |
| Lowest down-projection norm | 8,192 | 2,468.6 MiB | 0/24 | 20.87 tok/s |
| Seeded random | 8,192 | 2,468.6 MiB | 21/24 | 21.05 tok/s |

Each cut removed 251,658,240 weight entries and saved 152,371,200 bytes (145.3 MiB, 5.56%). All 330 tensors outside the cut were byte-identical. Exact match ignores case, surrounding whitespace, and final periods. For example, the original model's “Pacific Ocean” scores as a mismatch against “Pacific”. The random cut returned an incorrect Python result and two multilingual strings that differed from the expected answer. The norm cut frequently emitted empty content or stray thinking tokens.

The norm ranking repeatedly selected the earliest blocks. It removed block 0 in 29 of 32 layers, block 1 in 23, block 2 in 22, and block 3 in 20. The result shows that raw down-projection norm was a poor pruning criterion in this run. It does not isolate whether early block placement, cross-layer correlations, or another property caused the collapse.

The speed figures are exploratory medians of three 96-token runs on four CPU threads on mach-1, in fixed model order. Model outputs can differ even on the same speed prompt. The quality suite is a 24-item smoke test with a 12-token answer cap and one seeded random mask. The norm model's empty and thinking-token responses are failures under that inference budget. These numbers do not establish general quality retention or a production speedup. The five-worker baseline passed its health check separately.

## Reproduce

Use Python with NumPy and the gguf-py package from the same llama.cpp checkout:

```bash
export PYTHONPATH="$HOME/llama.cpp/gguf-py"
python3 prune_gguf.py "$MODEL" norm.gguf --strategy norm
python3 prune_gguf.py "$MODEL" random.gguf --strategy random
```

Each export writes a JSON manifest of the kept blocks. The benchmark script uses the mach-1 paths and writes a fresh JSONL file. On another host, update the source-model, output-directory, and server-binary paths in `main()`. Run it with the three GGUFs in place.

## Resume bullet

Built a block-aligned GGUF pruning pipeline for Qwen3.5-4B that removed 251.7M weight entries (145 MiB) without retraining or re-quantizing retained weights. Benchmarked magnitude-based and randomized cuts in llama.cpp and found that the norm heuristic collapsed on a 24-item evaluation while the random cut retained 21/24 exact matches against a 22/24 baseline.
