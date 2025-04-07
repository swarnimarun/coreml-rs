pub mod mlarray;
// pub mod mlmodel;
pub mod mlmodel;
mod swift;

// re-exports
pub use ffi::ComputePlatform;
pub use mlmodel::{CoreMLModelOptions, CoreMLModelWithState};

pub use swift::swift as ffi;
