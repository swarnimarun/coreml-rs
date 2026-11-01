//! Opt-in integration checks for real, separately downloaded models.
use coreml_rs::{FeatureValue, Inputs, Model, ModelOptions};
use image::imageops::FilterType;

#[test]
#[ignore = "run scripts/download-example-models.sh first; requires Core ML system services"]
fn apple_models_classify_and_segment_the_sample_image() {
    for (path, dictionary, width, height) in [
        (
            "demo/models/MobileNetV2FP16.mlmodel",
            "classLabelProbs",
            224,
            224,
        ),
        (
            "demo/models/FastViTT8F16.mlpackage",
            "classLabel_probs",
            256,
            256,
        ),
        ("demo/models/DeepLabV3FP16.mlmodel", "", 513, 513),
    ] {
        let compiled = Model::compile(path).unwrap();
        let model = compiled.load(ModelOptions::default()).unwrap();
        let image = image::open("demo/models/dog.jpg")
            .unwrap()
            .resize_exact(width, height, FilterType::Triangle)
            .to_rgba8();
        let feature =
            FeatureValue::image_from_rgba8(width as usize, height as usize, image.as_raw())
                .unwrap();
        let output = model
            .predict(&Inputs::new().with("image", feature))
            .unwrap();
        if dictionary.is_empty() {
            let mask = output
                .multi_array("semanticPredictions")
                .unwrap()
                .to_tensor::<i32>()
                .unwrap();
            assert_eq!(mask.shape(), &[height as usize, width as usize]);
            assert!(
                mask.as_slice().contains(&12),
                "dog class absent from segmentation"
            );
        } else {
            let label = output.get("classLabel").unwrap().as_string().unwrap();
            assert!(
                label.to_lowercase().contains("samoyed"),
                "unexpected label: {label}"
            );
            assert!(!output
                .get(dictionary)
                .unwrap()
                .as_dictionary()
                .unwrap()
                .unwrap()
                .is_empty());
        }
    }
}

#[test]
#[ignore = "requires downloaded FastViT package and macOS 15+"]
fn loads_mlprogram_weights_from_memory() {
    if !coreml_rs::Capabilities::current().in_memory_blobs {
        return;
    }
    let root = std::path::Path::new("demo/models/FastViTT8F16.mlpackage/Data/com.apple.CoreML");
    let specification = std::fs::read(root.join("model.mlmodel")).unwrap();
    let weights = std::fs::read(root.join("weights/weight.bin")).unwrap();
    let model = futures_executor::block_on(Model::load_from_memory_with_blobs(
        specification,
        [("weights/weight.bin", weights)],
        ModelOptions::default(),
    ))
    .unwrap();
    let image = image::open("demo/models/dog.jpg")
        .unwrap()
        .resize_exact(256, 256, FilterType::Triangle)
        .to_rgba8();
    let feature = FeatureValue::image_from_rgba8(256, 256, image.as_raw()).unwrap();
    let input = Inputs::new().with("image", feature);
    for _ in 0..3 {
        let output = model.predict(&input).unwrap();
        let label = output.get("classLabel").unwrap().as_string().unwrap();
        assert!(
            label.to_lowercase().contains("samoyed"),
            "unexpected label: {label}"
        );
    }
    let plan = futures_executor::block_on(model.compute_plan()).unwrap();
    assert!(matches!(
        plan.structure,
        coreml_rs::PlanStructure::Program { .. }
    ));
}
