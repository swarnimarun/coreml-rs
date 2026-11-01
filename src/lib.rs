//! Rust-first Core ML inference on macOS.
//!
//! Compilation and prediction use Apple's public Objective-C interfaces. Models
//! and native values retain their Core ML objects and do not require Swift.

#[cfg(not(target_os = "macos"))]
compile_error!("coreml-rs currently supports macOS only");

mod cache;
mod callback;
mod compute_plan;
mod error;
mod feature;
mod memory;
pub mod metal;
mod metrics;
mod model;
mod resources;
pub mod tensor;

pub use cache::ModelCache;
pub use compute_plan::{
    ComputePlan, PlanBlock, PlanDeviceUsage, PlanFunction, PlanLayer, PlanOperation, PlanStructure,
    PlanSubmodel,
};
pub use error::{Error, NativeError, Result};
pub use feature::{DictionaryKey, FeatureValue, Inputs, MultiArray, Outputs, PreparedInputs};
pub use metrics::{HardwareUsage, LatencySummary, MetricsHandle, ModelMetrics, LATENCY_WINDOW};
pub use model::{
    Capabilities, CompiledModel, ComputeDevice, ComputeUnits, FeatureDescription, FeatureType,
    ImageConstraint, Model, ModelDescription, ModelOptions, PredictionOptions, ShapeConstraint,
    State, ThermalState,
};
pub use resources::{MemoryPressureEvent, MemoryPressureMonitor, ProcessMemory};
pub use tensor::{DataType, Tensor, TensorElement, TensorError};
