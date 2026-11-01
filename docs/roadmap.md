# Known limitations

- Encrypted model loading uses the ordinary Core ML load operation and preserves its error codes, but has not been exercised with an encrypted model and Apple's key service.
- The macOS 12.3 baseline and Intel binaries need testing on actual older systems. Current integration tests ran on macOS 26.3.1 on Apple silicon.
- The synchronous API confines a model to one thread. Concurrent asynchronous prediction and cancellation are not yet implemented.
- Per-model hardware usage is not exposed by the public prediction interface used here. Compute plans give estimated device placement and relative operation costs, not measured utilization. Capabilities report device properties and system-wide pressure, not attribution.
- On macOS 26.3.1, Core ML rejected the embedded stateful accumulator when loaded as an in-memory asset, although it runs from a compiled package. This was true with both the specification-only and empty blob-mapping initializers, including CPU-only configuration. The native error is returned; use the compiled-package path for that model.
- Core Video pixel-buffer-backed arrays can share a Metal texture; arbitrary `MLMultiArray` storage is not exposed as `MTLBuffer`.
- Multiarray construction currently copies Rust tensors into Core ML storage. Reuse native arrays and prepared inputs to avoid repeating allocation and provider creation. Scoped native access and pixel-buffer sharing require explicit synchronization.
- Dictionary, image and sequence values can pass through native interop. Safe dictionary reading and image construction are provided; sequence conversion and image extraction to Rust pixels are not yet implemented.
- The v1/v2 comparison covers a tiny ReLU model and MobileNet with synthetic input. Larger models, memory usage, concurrent throughput and more devices still need benchmarks.
