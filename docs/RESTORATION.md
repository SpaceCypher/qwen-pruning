# Qwen3.5-4B block ablation and restoration

This study follows up on the 24-question pruning smoke test. It uses the same instruction-tuned Q4_K_M GGUF and llama.cpp revision `cf659f240`. The runner uses CPU inference, four threads, and a 256-token context on mach-1. It disables weight repacking.

Each runtime mask zeros complete 256-channel blocks at the output of the gated MLP activation, before the down projection. The evaluator checks the number of intercepted layers for each prompt. Restoring a layer means enabling all its previously masked channels. The original GGUF is shared by the runtime variants.

The arithmetic probe asks for the tens digit of a two-digit sum. It scores the highest-logit token among the ten digit tokens and logs the correct digit's log probability over the full vocabulary. The result is digit-choice accuracy. Development and test use disjoint operand ranges. This split also changes the distribution of sums, so compare conditions within each split. Two preliminary multiple-choice prompt formats had weak baseline scores and were discarded before the tens-digit test.

The screening procedure compares lowest down-projection norm, earliest block index, and five random seeds at one, two, and four blocks removed per layer. A 12-prompt development subset ranks 32 individual layer restorations by mean correct-digit log probability. The top four are frozen for a 40-prompt test against five equal-size random layer restorations. The block inspection computes the norm of each dequantized down-projection block multiplied by its recorded activation vector at the final prompt token.

## Reproduce

Build the evaluator against the existing llama.cpp checkout and run from a fresh run directory containing the norm manifest and the norm ranking. Use the absolute paths for `REPO`, `LLAMA_CPP`, `RUN`, and `MODEL` from the [README](../README.md).

```bash
mkdir -p "$RUN"
cp "$REPO/data/norm.json" "$REPO/data/norm-ranking.json" "$RUN/"
cd "$LLAMA_CPP"
c++ -std=c++17 -O2 -Iinclude -Iggml/include "$REPO/src/ablate.cpp" \
  -Lbuild/bin -Wl,-rpath,"$PWD/build/bin" -lllama -lggml-base \
  -o "$RUN/ablate"
cd "$RUN"
cp ablate ablate-trace
python3 "$REPO/scripts/restoration.py" "$MODEL" --dev 16 --test 40
PYTHONPATH="$LLAMA_CPP/gguf-py" OPENBLAS_NUM_THREADS=1 python3 "$REPO/scripts/inspect_blocks.py" "$MODEL"
python3 "$REPO/scripts/check_compact.py" "$MODEL"
python3 "$REPO/scripts/restoration.py" "$MODEL" --dev 16 --test 40 --followup
python3 "$REPO/scripts/restoration.py" "$MODEL" --dev 16 --test 40 --matched-test
python3 "$REPO/scripts/plot_restoration.py" --data-dir "$RUN" --output "$RUN/restoration.png"
```

The compact check expects the original smoke test's `norm.gguf` and `random.gguf`. The inspector requires NumPy and the plotter requires Matplotlib. The runners create fresh JSONL files and refuse to overwrite prior results.

## Results

The original scored 13/16 on development. Lowest-norm masks scored 2/16, 0/16, and 0/16 at one, two, and four blocks removed per layer. The five random masks ranged from 10–15, 8–13, and 4–14 correct, respectively. Earliest-position masks scored 1/16, 11/16, and 2/16. Accuracy is nonmonotonic for the earliest-position masks in this sample.

The development screen selected layers 12, 20, 29, and 2. On the larger-operand test, restoring them improved mean correct-digit log probability from −3.919 to −3.193. Digit accuracy fell from 5/40 to 2/40, within the 1–7/40 range of the five random restorations. The original scored 12/40. This test did not establish an accuracy rescue.

At the final prompt position in layer 29, the removed blocks had 0.84× the mean weight norm, 4.13× the mean activation norm, and 3.34× the mean projected contribution norm of the retained blocks. The corresponding contribution ratios were 1.98× at layer 2, 1.39× at layer 20, and 0.85× at layer 12. Each projection norm describes the size of a block's residual-stream contribution, averaged across the 16 development prompts. It does not measure whether that contribution points toward the correct answer or is indispensable when other blocks are active.

The original and physically compact lowest-norm GGUF matched their corresponding runtime conditions on all 40 digit predictions and printed log probabilities. The compact random GGUF scored 10/40 on those same prompts. Both compact files saved 145.3 MiB and reduced peak process RSS from 2.705 GiB to 2.561 GiB in the one-pass scoring harness. Median request time was 0.332 s for the original, 0.320 s for compact lowest-norm, and 0.287 s for compact random. This measures prompt scoring latency, not generation throughput.

The isolated knockout of the removed blocks at layers 2, 12, 20, and 29 scored 11/40, 19/40, 14/40, and 13/40, respectively, on the larger-operand set. The same-layer random cuts scored 14/40, 15/40, 13/40, and 13/40. None of the four single-layer interventions reproduced the full low-norm mask's 5/40 result on this set.

A second held-out split used unseen operand pairs from the development operand range. We added it after seeing the original model's weak baseline on larger operands and reused the frozen restoration layers and random controls. The original scored 35/40, the low-norm mask scored 0/40, and targeted restoration recovered 10/40. Random restorations scored 8, 5, 5, 3, and 12 out of 40. Targeted restoration had the best mean correct-digit log probability among restorations (−2.579 versus −2.884 to −3.921) and the second-best accuracy. The rescue was partial and depended on the operand range.

The study used an arithmetic digit probe. The earlier multi-domain smoke test remains separate. These results do not identify individual neurons or show that the selected blocks implement carry. The restored variants were tested as runtime masks. The size and RSS measurements apply to the physically compact original pruning variants.
