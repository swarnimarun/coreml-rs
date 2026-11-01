use coreml_rs::{
    Capabilities, ComputeUnits, Error, FeatureType, FeatureValue, Inputs, Model, ModelCache,
    ModelOptions, MultiArray, PredictionOptions, Tensor,
};

// A three-element Float32 ReLU network, serialized as a Core ML specification.
// Generated with coremltools NeuralNetworkBuilder: one RELU layer, input and
// output named "input" and "output". Keeping it inline avoids external fixtures.
const MODEL: &[u8] = &[
    8, 1, 18, 41, 10, 18, 10, 5, 105, 110, 112, 117, 116, 26, 9, 42, 7, 10, 1, 3, 16, 160, 128, 4,
    82, 19, 10, 6, 111, 117, 116, 112, 117, 116, 26, 9, 42, 7, 10, 1, 3, 16, 160, 128, 4, 162, 31,
    28, 10, 26, 10, 4, 114, 101, 108, 117, 18, 5, 105, 110, 112, 117, 116, 26, 6, 111, 117, 116,
    112, 117, 116, 146, 8, 2, 82, 0,
];

fn inputs(values: Vec<f32>) -> Inputs {
    let tensor = Tensor::from_vec(vec![3], values).unwrap();
    let array = MultiArray::from_tensor(&tensor).unwrap();
    Inputs::new().with("input", FeatureValue::multi_array(&array))
}

#[test]
fn idle_cache_respects_thresholds_and_live_clones() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("relu.mlmodel");
    std::fs::write(&path, MODEL).unwrap();
    let compiled = Model::compile(path).unwrap();
    let model = compiled.load(ModelOptions::default()).unwrap();
    let clone = model.clone();
    let observer = model.metrics_handle();
    let mut cache = ModelCache::new();
    cache.insert("relu", model);
    assert_eq!(cache.len(), 1);
    assert!(cache.evict_idle(std::time::Duration::ZERO).is_empty());
    drop(clone);
    cache
        .get(&"relu")
        .unwrap()
        .predict(&inputs(vec![1.0, 2.0, 3.0]))
        .unwrap();
    assert_eq!(observer.snapshot().completed, 1);
    assert!(cache.evict_idle(std::time::Duration::MAX).is_empty());
    assert_eq!(cache.evict_idle(std::time::Duration::ZERO), vec!["relu"]);
    assert!(cache.is_empty());
}

#[test]
fn compile_load_predict_batch_state_and_metrics() {
    let tmp = tempfile::tempdir().unwrap();
    let source = tmp.path().join("relu.mlmodel");
    std::fs::write(&source, MODEL).unwrap();
    let compiled = Model::compile(&source).unwrap();
    let model = Model::load(
        &compiled,
        ModelOptions {
            compute_units: ComputeUnits::CpuOnly,
            ..Default::default()
        },
    )
    .unwrap();
    if Capabilities::current().compute_plan {
        let plan = futures_executor::block_on(model.compute_plan()).unwrap();
        match plan.structure {
            coreml_rs::PlanStructure::NeuralNetwork { layers } => {
                assert!(!layers.is_empty());
                assert!(layers.iter().any(|layer| !layer.name.is_empty()));
                assert!(layers
                    .iter()
                    .filter_map(|layer| layer.device_usage.as_ref())
                    .all(|usage| !usage.supported.is_empty()));
            }
            other => panic!("expected neural network plan, got {other:?}"),
        }
    }
    let description = model.description();
    assert_eq!(description.inputs[0].name, "input");
    assert_eq!(description.inputs[0].feature_type, FeatureType::MultiArray);
    assert_eq!(description.inputs[0].shape.as_deref(), Some(&[3][..]));
    let observer = model.metrics_handle();
    let output = model.predict(&inputs(vec![-2.0, 1.0, 3.0])).unwrap();
    let array = output.multi_array("output").unwrap();
    assert_eq!(
        array.to_tensor::<f32>().unwrap().as_slice(),
        &[0.0, 1.0, 3.0]
    );
    assert!(array.to_tensor::<i32>().is_err());
    let backing =
        MultiArray::from_tensor(&Tensor::from_vec(vec![3], vec![0.0f32; 3]).unwrap()).unwrap();
    let with_backing = model
        .predict_with_options(
            &inputs(vec![-2.0, 1.0, 3.0]),
            &PredictionOptions::new().with_output("output", backing.clone()),
        )
        .unwrap();
    let returned = with_backing.multi_array("output").unwrap();
    assert_eq!(
        returned.to_tensor::<f32>().unwrap().as_slice(),
        &[0.0, 1.0, 3.0]
    );
    if returned.is_same_object(&backing) {
        assert_eq!(
            backing.to_tensor::<f32>().unwrap().as_slice(),
            &[0.0, 1.0, 3.0]
        );
    }
    let mut reused =
        MultiArray::from_tensor(&Tensor::from_vec(vec![3], vec![1.0f32; 3]).unwrap()).unwrap();
    let prepared = Inputs::new()
        .with("input", FeatureValue::multi_array(&reused))
        .prepare()
        .unwrap();
    reused
        .update(&Tensor::from_vec(vec![3], vec![-3.0f32, 4.0, 5.0]).unwrap())
        .unwrap();
    let output = model.predict_prepared(&prepared, None).unwrap();
    assert_eq!(
        output
            .multi_array("output")
            .unwrap()
            .to_tensor::<f32>()
            .unwrap()
            .as_slice(),
        &[0.0, 4.0, 5.0]
    );
    let batch = model
        .predict_batch(&[inputs(vec![1.0, -2.0, 3.0]), inputs(vec![-1.0, 2.0, -3.0])])
        .unwrap();
    assert_eq!(batch.len(), 2);
    assert_eq!(
        batch[1]
            .multi_array("output")
            .unwrap()
            .to_tensor::<f32>()
            .unwrap()
            .as_slice(),
        &[0.0, 2.0, 0.0]
    );
    assert!(model.predict(&Inputs::new()).is_err());
    if Capabilities::current().stateful_prediction {
        assert!(matches!(model.new_state(), Err(Error::Unsupported(_))));
    }
    let metrics = std::thread::spawn(move || observer.snapshot())
        .join()
        .unwrap();
    assert_eq!(metrics.failed, 1);
    assert_eq!(metrics.in_flight, 0);
    assert_eq!(metrics.batch_items, 2);
    assert_eq!(metrics.completed + metrics.failed, metrics.started);
    assert!(metrics.hardware_usage.is_none());
    assert!(metrics.last_started.is_some());
    assert!(metrics.last_completed.is_some());
    assert!(metrics.idle_for.is_some());
    assert!(metrics.latency.unwrap().samples >= 3);
    let persistent = tmp.path().join("persistent.mlmodelc");
    compiled.persist(&persistent).unwrap();
    assert!(compiled.persist(&persistent).is_err());
    drop(compiled);
    assert!(Model::load(&persistent, ModelOptions::default()).is_ok());
}

#[test]
fn errors_preserve_native_details() {
    let err = match Model::load("/no/such/model.mlmodelc", ModelOptions::default()) {
        Ok(_) => panic!("missing model loaded"),
        Err(err) => err,
    };
    match err {
        Error::Native { operation, source } => {
            assert_eq!(operation, "load model");
            assert!(!source.domain.is_empty());
            assert!(!source.description.is_empty());
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn pixel_buffer_to_metal_texture_shares_storage() {
    use coreml_rs::metal::MetalTextureCache;
    use objc2_core_foundation::{CFDictionary, CFRetained, CFString};
    use objc2_core_video::{
        kCVPixelBufferIOSurfacePropertiesKey, kCVPixelFormatType_OneComponent16Half, CVPixelBuffer,
        CVPixelBufferCreate,
    };
    use objc2_metal::{MTLCreateSystemDefaultDevice, MTLPixelFormat, MTLTexture};
    use std::ptr::NonNull;

    let Some(device) = MTLCreateSystemDefaultDevice() else {
        return;
    };
    let empty = CFDictionary::<CFString, CFString>::empty();
    let attrs = CFDictionary::from_slices(
        &[unsafe { kCVPixelBufferIOSurfacePropertiesKey }],
        &[&*empty],
    );
    // Core Video accepts an untyped CFDictionary; the actual key and value are
    // the documented IOSurface properties key and an empty dictionary.
    let attrs = unsafe { &*((&*attrs as *const _) as *const CFDictionary) };
    let mut ptr: *mut CVPixelBuffer = std::ptr::null_mut();
    let status = unsafe {
        CVPixelBufferCreate(
            None,
            4,
            2,
            kCVPixelFormatType_OneComponent16Half,
            Some(attrs),
            NonNull::from(&mut ptr),
        )
    };
    assert_eq!(status, 0);
    let pixel = unsafe { CFRetained::from_raw(NonNull::new(ptr).unwrap()) };
    let array = unsafe { MultiArray::from_pixel_buffer(&pixel, &[2, 4]) }.unwrap();
    assert!(unsafe { MultiArray::from_pixel_buffer(&pixel, &[4, 4]) }.is_err());
    let cache = MetalTextureCache::new(&device).unwrap();
    let texture = unsafe { cache.texture(&array, MTLPixelFormat::R16Float) }.unwrap();
    assert_eq!(texture.as_native().width(), 4);
    assert_eq!(texture.as_native().height(), 2);
}

#[test]
fn tensor_roundtrip_and_native_borrow() {
    let input = Tensor::from_vec(vec![2, 2], vec![1i32, -2, 3, 4]).unwrap();
    let mut array = MultiArray::from_tensor(&input).unwrap();
    assert_eq!(
        array.to_tensor::<i32>().unwrap().as_slice(),
        input.as_slice()
    );
    let sum =
        unsafe { array.with_contiguous::<i32, _>(|values| values.iter().sum::<i32>()) }.unwrap();
    assert_eq!(sum, 6);
    assert!(unsafe { array.with_contiguous::<f32, _>(|_| ()) }.is_err());
    unsafe { array.with_contiguous_mut::<i32, _>(|values| values[0] = 9) }.unwrap();
    assert_eq!(array.to_tensor::<i32>().unwrap().as_slice()[0], 9);
    assert!(unsafe { array.with_contiguous_mut::<f32, _>(|_| ()) }.is_err());
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let _ = array.with_contiguous_mut::<i32, ()>(|_| panic!("mutable callback panic"));
    }));
    assert!(panic.is_err());
    assert_eq!(array.to_tensor::<i32>().unwrap().as_slice()[0], 9);
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let _ = array.with_contiguous::<i32, ()>(|_| panic!("callback panic"));
    }));
    assert!(panic.is_err());
    if Capabilities::current().macos_version >= (26, 0, 0) {
        let ints = Tensor::from_vec(vec![3], vec![-4i8, 0, 7]).unwrap();
        assert_eq!(
            MultiArray::from_tensor(&ints)
                .unwrap()
                .to_tensor::<i8>()
                .unwrap()
                .as_slice(),
            ints.as_slice()
        );
    }
    assert_eq!(FeatureValue::int64(-12).as_int64(), Some(-12));
    assert_eq!(
        FeatureValue::string("hello").as_string().as_deref(),
        Some("hello")
    );
    assert!(FeatureValue::double(1.5).as_multi_array().is_none());
}

#[test]
fn loads_legacy_specification_from_memory() {
    if !Capabilities::current().in_memory_model {
        return;
    }
    let model = futures_executor::block_on(Model::load_from_memory(
        MODEL.to_vec(),
        ModelOptions {
            compute_units: ComputeUnits::CpuOnly,
            ..Default::default()
        },
    ))
    .unwrap();
    let output = model.predict(&inputs(vec![-2.0, 1.0, 3.0])).unwrap();
    assert_eq!(
        output
            .multi_array("output")
            .unwrap()
            .to_tensor::<f32>()
            .unwrap()
            .as_slice(),
        &[0.0, 1.0, 3.0]
    );
    assert_eq!(model.metrics().completed, 1);
    if Capabilities::current().compute_plan {
        let plan = futures_executor::block_on(model.compute_plan()).unwrap();
        assert!(matches!(
            plan.structure,
            coreml_rs::PlanStructure::NeuralNetwork { .. }
        ));
    }
    let invalid = futures_executor::block_on(Model::load_from_memory(
        vec![0, 1, 2],
        ModelOptions::default(),
    ));
    assert!(matches!(invalid, Err(Error::Native { .. })));
}
