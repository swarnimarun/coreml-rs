#[path = "fixtures/stateful.rs"]
mod fixture;
#[path = "fixtures/multifunction.rs"]
mod multifunction_fixture;

use coreml_rs::{
    Capabilities, Error, FeatureValue, Inputs, Model, ModelOptions, MultiArray, Tensor,
};
use half::f16;

#[test]
fn state_persists_and_is_tied_to_its_model() {
    if !Capabilities::current().stateful_prediction {
        return;
    }
    let temp = tempfile::tempdir().unwrap();
    let package = temp.path().join("Accumulator.mlpackage");
    let data = package.join("Data/com.apple.CoreML");
    std::fs::create_dir_all(data.join("weights")).unwrap();
    std::fs::write(data.join("model.mlmodel"), fixture::MODEL).unwrap();
    std::fs::write(package.join("Manifest.json"), r#"{"fileFormatVersion":"1.0.0","itemInfoEntries":{"model":{"author":"com.apple.CoreML","description":"model","name":"model.mlmodel","path":"com.apple.CoreML/model.mlmodel"},"weights":{"author":"com.apple.CoreML","description":"weights","name":"weights","path":"com.apple.CoreML/weights"}},"rootModelIdentifier":"model"}"#).unwrap();
    let compiled = Model::compile(&package).unwrap();
    let model = compiled.load(ModelOptions::default()).unwrap();
    let other = compiled.load(ModelOptions::default()).unwrap();
    let pinned = compiled.load(ModelOptions::default()).unwrap();
    let state_pin = pinned.new_state().unwrap();
    let mut cache = coreml_rs::ModelCache::new();
    cache.insert("stateful", pinned);
    assert!(cache.evict_idle(std::time::Duration::ZERO).is_empty());
    drop(state_pin);
    assert_eq!(
        cache.evict_idle(std::time::Duration::ZERO),
        vec!["stateful"]
    );
    drop(compiled); // Both models retain the temporary compiled artifact.
    let plan = futures_executor::block_on(model.compute_plan()).unwrap();
    match plan.structure {
        coreml_rs::PlanStructure::Program { functions } => {
            assert!(!functions.is_empty());
            assert!(functions
                .iter()
                .any(|function| !function.block.operations.is_empty()));
        }
        other => panic!("expected program plan, got {other:?}"),
    }
    let mut state = model.new_state().unwrap();
    let array =
        MultiArray::from_tensor(&Tensor::from_vec(vec![1], vec![f16::from_f32(1.0)]).unwrap())
            .unwrap();
    let inputs = Inputs::new().with("x", FeatureValue::multi_array(&array));
    assert!(matches!(
        other.predict_with_state(&inputs, &mut state),
        Err(Error::WrongModelState)
    ));
    for expected in [1.0, 2.0, 3.0] {
        let output = model.predict_with_state(&inputs, &mut state).unwrap();
        let tensor = output.multi_array("y").unwrap().to_tensor::<f16>().unwrap();
        assert_eq!(tensor.as_slice()[0].to_f32(), expected);
    }
    let mut independent = model.new_state().unwrap();
    let output = model.predict_with_state(&inputs, &mut independent).unwrap();
    assert_eq!(
        output
            .multi_array("y")
            .unwrap()
            .to_tensor::<f16>()
            .unwrap()
            .as_slice()[0]
            .to_f32(),
        1.0
    );
}

#[test]
fn named_functions_share_state_only_when_loaded_from_the_same_family() {
    if !Capabilities::current().stateful_prediction {
        return;
    }
    let temp = tempfile::tempdir().unwrap();
    let package = temp.path().join("MultiAccumulator.mlpackage");
    let data = package.join("Data/com.apple.CoreML");
    std::fs::create_dir_all(data.join("weights")).unwrap();
    std::fs::write(data.join("model.mlmodel"), multifunction_fixture::MODEL).unwrap();
    std::fs::write(package.join("Manifest.json"), r#"{"fileFormatVersion":"1.0.0","itemInfoEntries":{"model":{"author":"com.apple.CoreML","description":"model","name":"model.mlmodel","path":"com.apple.CoreML/model.mlmodel"},"weights":{"author":"com.apple.CoreML","description":"weights","name":"weights","path":"com.apple.CoreML/weights"}},"rootModelIdentifier":"model"}"#).unwrap();
    let compiled = Model::compile(package).unwrap();
    let decode = compiled
        .load(ModelOptions {
            compute_units: coreml_rs::ComputeUnits::CpuOnly,
            function_name: Some("decode".into()),
            ..Default::default()
        })
        .unwrap();
    let prefill =
        futures_executor::block_on(decode.load_function_sharing_state("prefill")).unwrap();
    assert!(matches!(
        futures_executor::block_on(decode.load_function_sharing_state("different")),
        Err(Error::ModelLoad(_))
    ));
    let separate = compiled
        .load(ModelOptions {
            compute_units: coreml_rs::ComputeUnits::CpuOnly,
            function_name: Some("prefill".into()),
            ..Default::default()
        })
        .unwrap();
    drop(compiled);
    let mut state = decode.new_state().unwrap();
    let mut array =
        MultiArray::from_tensor(&Tensor::from_vec(vec![1], vec![f16::from_f32(1.0)]).unwrap())
            .unwrap();
    let inputs = Inputs::new().with("x", FeatureValue::multi_array(&array));
    let prepared = inputs.prepare().unwrap();
    for (model, input, expected) in [
        (&decode, 1.0, 1.0),
        (&prefill, 2.0, 3.0),
        (&decode, 3.0, 6.0),
    ] {
        unsafe { array.with_contiguous_mut::<f16, _>(|values| values[0] = f16::from_f32(input)) }
            .unwrap();
        let output = model
            .predict_prepared_with_state(&prepared, &mut state)
            .unwrap();
        assert_eq!(
            output
                .multi_array("y")
                .unwrap()
                .to_tensor::<f16>()
                .unwrap()
                .as_slice()[0]
                .to_f32(),
            expected
        );
    }
    assert!(matches!(
        separate.predict_with_state(&inputs, &mut state),
        Err(Error::WrongModelState)
    ));
    assert_eq!(decode.metrics().completed, 2);
    assert_eq!(prefill.metrics().completed, 1);
}
