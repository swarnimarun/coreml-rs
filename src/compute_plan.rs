//! Owned snapshots of Core ML's anticipated execution plans.
use crate::{model::describe_compute_device, ComputeDevice, Error, Result};
use block2::RcBlock;
use objc2_core_ml::{
    MLComputePlan, MLComputePlanDeviceUsage, MLModelAsset, MLModelConfiguration, MLModelStructure,
    MLModelStructureProgramBlock,
};
use objc2_foundation::{NSError, NSURL};
use std::{future::Future, sync::Arc};

/// A snapshot of Core ML's anticipated plan, not a live utilization report.
#[derive(Clone, Debug, PartialEq)]
pub struct ComputePlan {
    pub structure: PlanStructure,
}

/// Model structures supported by Core ML's compute-plan API.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum PlanStructure {
    NeuralNetwork {
        layers: Vec<PlanLayer>,
    },
    Program {
        functions: Vec<PlanFunction>,
    },
    Pipeline {
        models: Vec<PlanSubmodel>,
    },
    /// Core ML did not provide a supported structure for this model.
    Other,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlanSubmodel {
    pub name: Option<String>,
    pub structure: PlanStructure,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlanLayer {
    pub name: String,
    pub layer_type: String,
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
    /// None means Core ML could not determine device usage.
    pub device_usage: Option<PlanDeviceUsage>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlanFunction {
    pub name: String,
    pub block: PlanBlock,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlanBlock {
    pub output_names: Vec<String>,
    pub operations: Vec<PlanOperation>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlanOperation {
    pub operator_name: String,
    pub output_names: Vec<String>,
    /// None means Core ML could not determine device usage.
    pub device_usage: Option<PlanDeviceUsage>,
    /// Relative estimated workload, normally between 0 and 1. Not elapsed time.
    pub estimated_cost: Option<f64>,
    pub blocks: Vec<PlanBlock>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlanDeviceUsage {
    pub preferred: ComputeDevice,
    pub supported: Vec<ComputeDevice>,
}

// Bound recursive Objective-C structures from malformed or unusually deep
// models. Return an error rather than silently dropping nested operations.
const MAX_DEPTH: usize = 128;

unsafe fn device_usage(usage: &MLComputePlanDeviceUsage) -> PlanDeviceUsage {
    PlanDeviceUsage {
        preferred: describe_compute_device(&usage.preferredComputeDevice()),
        supported: usage
            .supportedComputeDevices()
            .iter()
            .map(|device| describe_compute_device(&device))
            .collect(),
    }
}

unsafe fn block(
    plan: &MLComputePlan,
    value: &MLModelStructureProgramBlock,
    depth: usize,
) -> Result<PlanBlock> {
    if depth > MAX_DEPTH {
        return Err(Error::ComputePlan(
            "model structure exceeds supported nesting depth",
        ));
    }
    let mut operations = Vec::new();
    for operation in value.operations().iter() {
        let usage = plan.computeDeviceUsageForMLProgramOperation(&operation);
        let cost = plan.estimatedCostOfMLProgramOperation(&operation);
        let blocks = operation
            .blocks()
            .iter()
            .map(|child| block(plan, &child, depth + 1))
            .collect::<Result<Vec<_>>>()?;
        operations.push(PlanOperation {
            operator_name: operation.operatorName().to_string(),
            output_names: operation
                .outputs()
                .iter()
                .map(|name| name.name().to_string())
                .collect(),
            device_usage: usage.as_deref().map(|usage| device_usage(usage)),
            estimated_cost: cost.as_deref().map(|cost| cost.weight()),
            blocks,
        });
    }
    Ok(PlanBlock {
        output_names: value
            .outputNames()
            .iter()
            .map(|name| name.to_string())
            .collect(),
        operations,
    })
}

unsafe fn structure(
    plan: &MLComputePlan,
    value: &MLModelStructure,
    depth: usize,
) -> Result<PlanStructure> {
    if depth > MAX_DEPTH {
        return Err(Error::ComputePlan(
            "model structure exceeds supported nesting depth",
        ));
    }
    if let Some(network) = value.neuralNetwork() {
        let layers = network
            .layers()
            .iter()
            .map(|layer| {
                let usage = plan.computeDeviceUsageForNeuralNetworkLayer(&layer);
                PlanLayer {
                    name: layer.name().to_string(),
                    layer_type: layer.r#type().to_string(),
                    input_names: layer.inputNames().iter().map(|s| s.to_string()).collect(),
                    output_names: layer.outputNames().iter().map(|s| s.to_string()).collect(),
                    device_usage: usage.as_deref().map(|usage| device_usage(usage)),
                }
            })
            .collect();
        Ok(PlanStructure::NeuralNetwork { layers })
    } else if let Some(program) = value.program() {
        let (names, functions) = program.functions().to_vecs();
        let mut snapshots = Vec::with_capacity(names.len());
        for (name, function) in names.iter().zip(functions.iter()) {
            snapshots.push(PlanFunction {
                name: name.to_string(),
                block: block(plan, &function.block(), depth + 1)?,
            });
        }
        snapshots.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(PlanStructure::Program {
            functions: snapshots,
        })
    } else if let Some(pipeline) = value.pipeline() {
        let names: Vec<_> = pipeline
            .subModelNames()
            .iter()
            .map(|name| name.to_string())
            .collect();
        let mut models = Vec::new();
        for (index, model) in pipeline.subModels().iter().enumerate() {
            models.push(PlanSubmodel {
                name: names.get(index).cloned(),
                structure: structure(plan, &model, depth + 1)?,
            });
        }
        Ok(PlanStructure::Pipeline { models })
    } else {
        Ok(PlanStructure::Other)
    }
}

pub(crate) enum PlanSource<'a> {
    File {
        url: &'a NSURL,
        artifact: Option<Arc<tempfile::TempDir>>,
    },
    Memory {
        asset: &'a MLModelAsset,
        buffers: Arc<crate::memory::MemoryBuffers>,
    },
}

pub(crate) fn load(
    source: PlanSource<'_>,
    configuration: &MLModelConfiguration,
) -> impl Future<Output = Result<ComputePlan>> {
    let (sender, future) = crate::callback::channel();
    let (artifact, buffers) = match &source {
        PlanSource::File { artifact, .. } => (artifact.clone(), None),
        PlanSource::Memory { buffers, .. } => (None, Some(buffers.clone())),
    };
    let callback = RcBlock::new(move |plan: *mut MLComputePlan, error: *mut NSError| {
        // Keep a temporary compiled artifact alive even when the caller cancels
        // the Rust future before Core ML invokes its completion handler.
        let _keep_inputs_alive = (&artifact, &buffers);
        if sender.is_cancelled() {
            return;
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            objc2::rc::autoreleasepool(|_| {
                // SAFETY: Core ML supplies live callback arguments for this call.
                // The native plan is converted to owned Rust data before returning.
                unsafe {
                    if let Some(error) = error.as_ref() {
                        Err(Error::native("load compute plan", error))
                    } else if let Some(plan) = plan.as_ref() {
                        Ok(ComputePlan {
                            structure: structure(plan, &plan.modelStructure(), 0)?,
                        })
                    } else {
                        Err(Error::ComputePlan(
                            "Core ML returned neither a plan nor an error",
                        ))
                    }
                }
            })
        }))
        .unwrap_or(Err(Error::ComputePlan(
            "panic while reading Core ML compute plan",
        )));
        sender.complete(result);
    });
    // SAFETY: the URL and configuration are live during the call. Core ML owns
    // its copied completion block until it invokes it asynchronously.
    unsafe {
        match source {
            PlanSource::File { url, .. } => {
                MLComputePlan::loadContentsOfURL_configuration_completionHandler(
                    url,
                    configuration,
                    &callback,
                )
            }
            PlanSource::Memory { asset, .. } => {
                MLComputePlan::loadModelAsset_configuration_completionHandler(
                    asset,
                    configuration,
                    &callback,
                )
            }
        }
    };
    future
}
