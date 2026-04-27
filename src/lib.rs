pub mod mlarray;
pub mod mlbatchmodel;
pub mod mlmodel;

mod swift;

// re-exports
pub use ffi::ComputePlatform;
pub use mlmodel::{CoreMLModelOptions, CoreMLModelWithState, PredictRetryOptions, RetryBackoff};

pub use swift::swift as ffi;
