# Comparing v1 and v2

`benchmarks/compare.py` builds the v1 revision in an isolated temporary checkout using JJ, builds v2 in release mode, and runs both against the same compiled Core ML models. Compilation, model loading, input generation, warmup, validation and CSV writing are outside the timed sections. The driver retains raw samples and output vectors for inspection.

## Prepare and run

The tiny ReLU case needs no external model. The MobileNetV2 case uses Apple's downloaded FP16 model, converted to accept a normalized float array and return its 1,001-element softmax vector. Conversion preserves the network layers and weights while removing image preprocessing and the classifier dictionary wrapper.

```sh
./scripts/download-example-models.sh
python3 -m venv demo/benchmarks/venv
demo/benchmarks/venv/bin/pip install coremltools==9.0
demo/benchmarks/venv/bin/python benchmarks/make_mobilenet_array.py \
  demo/models/MobileNetV2FP16.mlmodel demo/benchmarks/MobileNetV2Array.mlmodel
python3 benchmarks/compare.py --compute gpu --rounds 3 --iterations 120 --warmup 20 --pool 8
```

The converter checks the source model's SHA-256 digest and expected preprocessing. Before the timed runs, the driver checks the derived model against the original image-input model on a deterministic image. MobileNet has two classes named `crane`; the check compares the other 999 unique labels, requiring maximum absolute difference at most 0.005 and total absolute difference at most 0.05. This is a conversion sanity check, not an accuracy evaluation.

The v1 baseline defaults to JJ revision `a0a081b0`; override it with `--baseline REVISION`. `--compute cpu`, `gpu` and `ane` request CPU only, CPU + GPU and CPU + Neural Engine respectively. Core ML still chooses actual placement. `--case relu` runs only the self-contained ReLU case. Run `python3 benchmarks/compare.py --help` for all options.

v1 requires a working Swift toolchain and runtime. If its executable cannot find `libswift_Concurrency.dylib`, pass `--swift-runtime-dir` with the directory containing a compatible library. For example, on some Command Line Tools installations it is `$(xcode-select -p)/usr/lib/swift-5.5/macosx`. Use a matching Swift runtime where possible; this machine emitted a duplicate Objective-C class warning with that compatibility library, limiting confidence in its v1 measurements. The benchmark needs access to Core ML system services and may fail inside a restricted sandbox.

## What the timings include

- `predict`: input construction and binding happen before the timer. The timer includes the public prediction call and making the output available as a Rust tensor. Both implementations still construct a native feature provider inside their prediction call.
- `end-to-end`: the timer also includes cloning the pre-generated input and constructing the Rust and native input objects.
- `reuse` is v2-only: one native input array and prepared provider are retained; each timed iteration updates the array from a pre-generated Rust tensor, predicts and copies the output to Rust. This is a distinct usage pattern, not an identical v1 operation.

The CSV also has `prepare_ns`, `call_ns` and `output_copy_ns`. These are diagnostic timings with extra clock reads. The v1 `call_ns` already includes v1's eager output copy; v2's `call_ns` returns native outputs and its `output_copy_ns` contains the subsequent copy. Do not compare those two call stages as equivalent operations. Total latency is the fairer comparison.

Inputs are generated before timing from a fixed non-cryptographic seed. The default pool contains eight distinct inputs; a deterministic balanced shuffle chooses their order. MobileNet inputs are synthetic normalized RGB tensors, not a representative photo dataset. Each input occurs repeatedly. The runners reject non-finite or incorrectly shaped output, check ReLU against its analytic result, check MobileNet probabilities are normalized and vary across the input pool, and compare every repeated output using an absolute tolerance of `1e-5` plus a relative tolerance of `1e-3`. The driver compares v1 and v2 for the same input and rejects mismatches using the same tolerance.

Each round runs the two versions in a shuffled job order. `summary.json` records per-run median, p95 and p99, stage timings, source and harness hashes, OS, CPU, toolchain and configuration. The printed comparison is the median of the round medians. Raw CSVs and metadata go under the ignored `demo/benchmarks/results/` directory, or a new directory supplied via `--output-dir`. Re-run with different seeds, more rounds, and on an otherwise idle machine; CPU scheduling, thermal state and Core ML placement can change results.

## One local run

On an Apple M4 Pro with macOS 26.3.1, using CPU + GPU, three rounds, 120 measured calls per round, 20 warmups and eight inputs, the median of the round medians was as follows. This run predates the bounded latency-history metrics now recorded by v2; rerun the benchmark to assess their overhead.

| Case | Measurement | v1 | v2 |
| --- | --- | ---: | ---: |
| ReLU | prediction | 0.182 ms | 0.156 ms |
| ReLU | end-to-end | 0.192 ms | 0.162 ms |
| MobileNetV2 | prediction | 2.257 ms | 2.249 ms |
| MobileNetV2 | end-to-end | 2.271 ms | 2.288 ms |

A second MobileNet-only CPU run used three rounds of 80 measured calls. Its median of round medians was 6.658 ms for v1 and 6.680 ms for v2 in `predict`, and 6.624 ms for v1 and 6.674 ms for v2 end to end.

A later ReLU-only CPU + GPU run with the new per-request latency history used the same three-round, 120-call settings. The median of round medians was 0.160 ms for both versions in `predict`, and 0.180 ms for v1 versus 0.167 ms for v2 end to end. Outputs passed the same validation. This is a separate noisy run, not a controlled measurement of metrics overhead; the v1 Swift runtime warning also applies.

The maximum v1/v2 output difference across the tested samples was zero. The original-versus-derived MobileNet check had a maximum absolute difference of 0.00183 and total absolute difference of 0.00408 across unique labels. In this run, v2's MobileNet end-to-end preparation median was about 0.030 ms and its output copy median about 0.002 ms, versus roughly 2.25 ms for the native prediction call. The v2 reuse path was about 2.256 ms total. These are noisy measurements from one machine, and the v1 runtime warning applies; they do not establish a general winner.

The next useful optimization experiment is to reduce or eliminate the input copy for large tensors, then compare `end-to-end` and `reuse` again on larger real workloads. For this MobileNet case, Core ML execution dominates, so improving the Rust preparation path alone has a limited possible effect. Use Instruments and model-specific tests to investigate execution, placement and memory behavior; these timings do not attribute hardware utilization to a model.

For a focused follow-up, compare these changes one at a time with the same seeds and settings:

1. Add an owned, no-copy input constructor that transfers a contiguous `Vec<T>` to `MLMultiArray` with a Rust-owned deallocator. Test lifetimes and failure cleanup, and verify whether Core ML itself still copies for the selected device. Compare it with `end-to-end` and `reuse`; do not assume zero-copy improves latency.
2. Reuse a preallocated output backing for individual predictions when Core ML accepts it, or keep outputs native when the next consumer supports that. The current MobileNet output is small, so test a model with a larger output before prioritizing this work. Core ML may ignore output backings, and batch predictions do not use them.
3. Profile the native call and device placement with Instruments or Core ML compute-plan APIs. Compare CPU-only, CPU + GPU and CPU + Neural Engine in separate runs. If throughput matters, benchmark safe asynchronous concurrency separately from single-request latency.

The observed 0.030 ms input-preparation median is roughly 1.3% of this run's 2.288 ms v2 end-to-end median. Even eliminating all of that stage would have limited impact on this particular workload. Treat that as a workload-specific upper-bound estimate, not a prediction for larger tensors or other models.
