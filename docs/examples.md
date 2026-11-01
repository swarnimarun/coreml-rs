# Examples and test models

Run `./scripts/download-example-models.sh` from the repository root. The script downloads into the ignored `demo/models` directory and checks SHA-256 digests. The sample photo comes from [PyTorch Hub](https://github.com/pytorch/hub/blob/master/images/dog.jpg). Model and photo licenses are separate from this crate.

| Model | Format | Source and terms | Example |
| --- | --- | --- | --- |
| MobileNetV2 FP16 | `.mlmodel` | [Apple model gallery](https://developer.apple.com/machine-learning/models/); model metadata points to [TensorFlow](https://github.com/tensorflow/tensorflow) | `classify` |
| FastViT T8 FP16 | `.mlpackage` | [Apple model gallery](https://developer.apple.com/machine-learning/models/); [model license](https://github.com/apple/ml-fastvit/blob/8af5928238cab99c45f64fc3e4e7b1516b8224ba/LICENSE) | `classify` |
| DeepLabV3 FP16 | `.mlmodel` | [Apple model gallery](https://developer.apple.com/machine-learning/models/); model metadata points to [TensorFlow](https://github.com/tensorflow/tensorflow) | `segment` |

```sh
cargo run --release --example simple -- demo/models/MobileNetV2FP16.mlmodel
cargo run --release --example classify -- demo/models/MobileNetV2FP16.mlmodel demo/models/dog.jpg 20 all
cargo run --release --example classify -- demo/models/FastViTT8F16.mlpackage demo/models/dog.jpg 20 cpu
cargo run --release --example segment -- demo/models/DeepLabV3FP16.mlmodel demo/models/dog.jpg demo/models/segmentation.png
cargo run --release --example tensors
cargo run --release --example stateful  # macOS 15+
cargo run --release --example metal     # compatible Metal device
cargo run --release --example compute_plan -- demo/models/MobileNetV2FP16.mlmodel  # macOS 14.4+
cargo run --release --example observe -- demo/models/MobileNetV2FP16.mlmodel
cargo run --release --example in_memory -- demo/models/MobileNetV2FP16.mlmodel  # macOS 13+
cargo run --release --example in_memory -- demo/models/FastViTT8F16.mlpackage/Data/com.apple.CoreML/model.mlmodel weights/weight.bin demo/models/FastViTT8F16.mlpackage/Data/com.apple.CoreML/weights/weight.bin  # macOS 15+
cargo run --release --example benchmark_tensors
python3 benchmarks/compare.py --case relu --compute cpu
cargo test --all-features -- --ignored --test-threads=1
```

The classifier prints compile and load durations, the first prediction and a mean for warmed predictions. These are a local diagnostic, not a benchmark across machines. Use the same model, image, compute selection, build profile, warmup and iteration count when comparing changes. The example resizes the image directly; use a model's required crop and normalization for production. The segmentation colors are an arbitrary visualization; integer outputs carry the model's class IDs.

On the development Mac, both classifiers returned Samoyed for the sample, and DeepLab returned a dog region. Exact probabilities, timings and placement can vary by hardware and OS. The ignored integration test checks the same three models after download; the normal test suite uses an embedded, deterministic ReLU model and does not download files.

The stateful example uses a small, embedded accumulator generated with Core ML Tools. The Metal example maps a half-float IOSurface-backed array to a Metal texture and explains the synchronization responsibility. Neither example requires a downloaded model. The in-memory example uses files as a source of bytes; an application can obtain those bytes another way. The compute-plan example reports estimated placement and costs, plus request metrics.

For the randomized v1/v2 comparison, setup, validation and interpretation, see [Comparing v1 and v2](performance.md).
