#[path = "fixtures/in_memory_program.rs"]
mod fixture;

use coreml_rs::{
    Capabilities, ComputeUnits, Error, FeatureValue, Inputs, Model, ModelOptions, MultiArray,
    PlanStructure, Tensor,
};
use half::f16;

#[test]
fn loads_program_with_external_weights_and_returns_its_plan() {
    if !Capabilities::current().in_memory_blobs {
        return;
    }
    let model = futures_executor::block_on(Model::load_from_memory_with_blobs(
        fixture::MODEL.to_vec(),
        [("weights/weight.bin", fixture::WEIGHTS.to_vec())],
        ModelOptions {
            compute_units: ComputeUnits::CpuOnly,
            ..Default::default()
        },
    ))
    .unwrap();
    let input: Vec<_> = (0..16).map(|n| f16::from_f32(n as f32 - 8.0)).collect();
    let array =
        MultiArray::from_tensor(&Tensor::from_vec(vec![1, 16], input.clone()).unwrap()).unwrap();
    let output = model
        .predict(&Inputs::new().with("x", FeatureValue::multi_array(&array)))
        .unwrap();
    let result = output.multi_array("y").unwrap().to_tensor::<f16>().unwrap();
    for (actual, original) in result.as_slice().iter().zip(input) {
        assert_eq!(actual.to_f32(), 2.0 * original.to_f32());
    }
    let plan = futures_executor::block_on(model.compute_plan()).unwrap();
    match plan.structure {
        PlanStructure::Program { functions } => assert!(functions
            .iter()
            .any(|function| !function.block.operations.is_empty())),
        other => panic!("expected program plan, got {other:?}"),
    }
    let metrics = model.metrics();
    assert_eq!(
        (metrics.started, metrics.completed, metrics.failed),
        (1, 1, 0)
    );
    assert_eq!(metrics.latency.unwrap().samples, 1);
}

#[test]
fn invalid_blob_mappings_return_errors() {
    if !Capabilities::current().in_memory_blobs {
        return;
    }
    let missing = futures_executor::block_on(Model::load_from_memory_with_blobs(
        fixture::MODEL.to_vec(),
        std::iter::empty::<(&str, Vec<u8>)>(),
        ModelOptions::default(),
    ));
    assert!(missing.is_err());
    let duplicate = futures_executor::block_on(Model::load_from_memory_with_blobs(
        fixture::MODEL.to_vec(),
        [
            ("weights/weight.bin", fixture::WEIGHTS.to_vec()),
            ("weights/weight.bin", fixture::WEIGHTS.to_vec()),
        ],
        ModelOptions::default(),
    ));
    assert!(matches!(duplicate, Err(Error::InvalidInput(_))));
}
