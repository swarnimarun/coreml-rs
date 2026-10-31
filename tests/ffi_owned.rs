//! Synthetic, redistributable Core ML regression fixtures (no downloaded model).
//! Field identifiers are taken from Apple's published protobuf schemas:
//! https://github.com/apple/coremltools/blob/main/mlmodel/format/Model.proto
//! https://github.com/apple/coremltools/blob/main/mlmodel/format/FeatureTypes.proto
//! https://github.com/apple/coremltools/blob/main/mlmodel/format/NeuralNetwork.proto

use coreml_rs::{
    ffi, mlarray::MLArray, mlbatchmodel::CoreMLBatchModelWithState, mlmodel::CoreMLError,
    ComputePlatform, CoreMLModelOptions, CoreMLModelWithState,
};
use std::{
    process::Command,
    thread,
    time::{Duration, Instant},
};

// A crash or a blocked Swift task must fail this test without hanging Cargo.
fn subprocess(case: &str, run: impl FnOnce()) {
    const CASE_ENV: &str = "COREML_OWNED_FFI_CASE";
    if std::env::var(CASE_ENV).as_deref() == Ok(case) {
        run();
        return;
    }
    let mut child = Command::new(std::env::current_exe().expect("test executable"))
        .arg("--exact")
        .arg(case)
        .arg("--nocapture")
        .env(CASE_ENV, case)
        .spawn()
        .expect("start isolated Core ML test");
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match child.try_wait().expect("inspect isolated test") {
            Some(status) => {
                assert!(status.success(), "{case} failed: {status}");
                return;
            }
            None if Instant::now() >= deadline => {
                child.kill().expect("stop timed-out Core ML test");
                child.wait().expect("reap timed-out Core ML test");
                panic!("{case} did not finish within 30 seconds");
            }
            None => thread::sleep(Duration::from_millis(25)),
        }
    }
}

#[derive(Default)]
struct Proto(Vec<u8>);

impl Proto {
    fn varint(&mut self, mut value: u64) {
        loop {
            let byte = (value & 0x7f) as u8;
            value >>= 7;
            self.0.push(byte | if value == 0 { 0 } else { 0x80 });
            if value == 0 {
                return;
            }
        }
    }

    fn integer(&mut self, field: u64, value: u64) {
        self.varint(field << 3);
        self.varint(value);
    }

    fn bytes(&mut self, field: u64, value: &[u8]) {
        self.varint((field << 3) | 2);
        self.varint(value.len() as u64);
        self.0.extend_from_slice(value);
    }
}

fn scalar_feature(name: &str) -> Proto {
    let mut array = Proto::default();
    array.integer(1, 1); // one element
    array.integer(2, 65568); // FLOAT32
    let mut feature_type = Proto::default();
    feature_type.bytes(5, &array.0); // multiArrayType
    let mut feature = Proto::default();
    feature.bytes(1, name.as_bytes());
    feature.bytes(3, &feature_type.0);
    feature
}

fn relu_specification() -> Vec<u8> {
    let mut description = Proto::default();
    description.bytes(1, &scalar_feature("x").0);
    description.bytes(10, &scalar_feature("y").0);
    let mut activation = Proto::default();
    activation.bytes(10, &[]); // ReLU
    let mut layer = Proto::default();
    layer.bytes(1, b"owned_ffi_relu");
    layer.bytes(2, b"x");
    layer.bytes(3, b"y");
    layer.bytes(130, &activation.0);
    let mut network = Proto::default();
    network.bytes(1, &layer.0);
    let mut model = Proto::default();
    model.integer(1, 2); // specificationVersion
    model.bytes(2, &description.0);
    model.bytes(500, &network.0);
    model.0
}

fn options() -> CoreMLModelOptions {
    CoreMLModelOptions {
        compute_platform: ComputePlatform::Cpu,
        ..Default::default()
    }
}

fn output_value(output: &MLArray) -> f32 {
    match output {
        MLArray::Float32Array(values) => {
            assert_eq!(values.len(), 1);
            *values.iter().next().unwrap()
        }
        _ => panic!("expected an f32 output"),
    }
}

#[test]
fn invalid_assets_return_unloaded_without_hanging() {
    subprocess("invalid_assets_return_unloaded_without_hanging", || {
        for bytes in [vec![], vec![0xff; 17], vec![8, 2]] {
            let single = CoreMLModelWithState::from_buf(bytes.clone(), options()).load();
            assert!(matches!(
                single,
                Err(CoreMLError::FailedToLoadStatic(
                    _,
                    CoreMLModelWithState::Unloaded(_, _)
                ))
            ));
            let batch = CoreMLBatchModelWithState::from_buf(bytes, options()).load();
            assert!(matches!(
                batch,
                Err(CoreMLError::FailedToLoadBatchStatic(
                    _,
                    CoreMLBatchModelWithState::Unloaded(_, _)
                ))
            ));
        }
    });
}

#[test]
fn owned_buffer_predicts_after_reload() {
    subprocess("owned_buffer_predicts_after_reload", || {
        let specification = relu_specification();
        let mut bytes = Vec::with_capacity(specification.len() + 257);
        bytes.extend_from_slice(&specification);
        assert!(bytes.capacity() > bytes.len());
        let mut model = CoreMLModelWithState::from_buf(bytes, options())
            .load()
            .unwrap();
        for value in [-3.0f32, 7.0, 0.0] {
            model
                .add_input("x", ndarray::arr1(&[value]).into_dyn())
                .unwrap();
            let output = model.predict().unwrap();
            assert_eq!(output_value(&output.outputs["y"]), value.max(0.0));
            model = model.unload().unwrap().load().unwrap();
        }
    });
}

#[test]
fn owned_batch_buffer_predicts_after_reload() {
    subprocess("owned_batch_buffer_predicts_after_reload", || {
        let mut model = CoreMLBatchModelWithState::from_buf(relu_specification(), options())
            .load()
            .unwrap();
        for _ in 0..2 {
            for (index, value) in [-5.0f32, 9.0].into_iter().enumerate() {
                model
                    .add_input("x", ndarray::arr1(&[value]).into_dyn(), index as isize)
                    .unwrap();
            }
            let output = model.predict().unwrap();
            assert_eq!(output.outputs.len(), 2);
            assert_eq!(output_value(&output.outputs[0]["y"]), 0.0);
            assert_eq!(output_value(&output.outputs[1]["y"]), 9.0);
            model = model.unload().unwrap().load().unwrap();
        }
    });
}

#[test]
fn owned_bindings_reject_invalid_shapes_and_batch_indices() {
    subprocess(
        "owned_bindings_reject_invalid_shapes_and_batch_indices",
        || {
            let model = ffi::modelWithAssets(relu_specification(), ComputePlatform::Cpu);
            assert!(!model.failed());
            assert!(!model.bindInputF32(vec![2], "x".into(), vec![1.0]));
            assert!(!model.bindInputI32(vec![1], "x".into(), vec![]));
            assert!(!model.bindInputU16(vec![0], "x".into(), vec![]));
            assert!(!model.bindInputF32(vec![usize::MAX, 2], "x".into(), vec![]));
            assert!(!model.bindOutputF32(vec![-1], "y".into(), vec![0.0]));
            // These validate ownership and tensor construction, not dtype inference.
            assert!(model.bindInputF32(vec![1], "x".into(), vec![2.0]));
            assert!(model.bindInputI32(vec![1], "x".into(), vec![2]));
            assert!(model.bindInputU16(
                vec![1],
                "x".into(),
                vec![half::f16::from_f32(2.0).to_bits()]
            ));
            assert!(model.bindOutputF32(vec![1], "y".into(), vec![0.0]));

            let batch = ffi::modelWithAssetsBatch(relu_specification(), ComputePlatform::Cpu);
            assert!(!batch.bindInputF32(vec![1], "x".into(), vec![1.0], -1));
            assert!(!batch.bindInputF32(vec![1], "x".into(), vec![1.0], 1));
            assert!(batch.bindInputF32(vec![1], "x".into(), vec![1.0], 0));
            assert!(batch.bindInputF32(vec![1], "x".into(), vec![2.0], 1));
        },
    );
}
