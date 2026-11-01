//! Render the integer output of Apple's DeepLabV3 into a color mask.
//! Run: cargo run --release --example segment -- MODEL IMAGE OUTPUT.png
use coreml_rs::{FeatureValue, Inputs, Model, ModelOptions};
use image::{imageops::FilterType, Rgb, RgbImage};
use std::{error::Error, path::Path};

const LABELS: [&str; 21] = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "dining table",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv or monitor",
];

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: segment <DeepLabV3 model> <image> <output.png>".into());
    }
    let compiled = if Path::new(&args[1])
        .extension()
        .is_some_and(|s| s == "mlmodelc")
    {
        None
    } else {
        Some(Model::compile(&args[1])?)
    };
    let path = compiled
        .as_ref()
        .map(|c| c.path())
        .unwrap_or_else(|| Path::new(&args[1]));
    let model = Model::load(path, ModelOptions::default())?;
    let description = model.description();
    let input = description
        .inputs
        .iter()
        .find(|v| v.image.is_some())
        .ok_or("model has no image input")?;
    let size = input.image.as_ref().unwrap();
    let (width, height) = (u32::try_from(size.width)?, u32::try_from(size.height)?);
    let image = image::open(&args[2])?
        .resize_exact(width, height, FilterType::Triangle)
        .to_rgba8();
    let inputs = Inputs::new().with(
        &input.name,
        FeatureValue::image_from_rgba8(width as usize, height as usize, image.as_raw())?,
    );
    let output = model.predict(&inputs)?;
    let mask = output
        .multi_array("semanticPredictions")?
        .to_tensor::<i32>()?;
    if mask.shape() != [height as usize, width as usize] {
        return Err("unexpected mask shape".into());
    }
    let mut pixels = RgbImage::new(width, height);
    let mut counts = [0usize; 21];
    for (pixel, &class) in pixels.pixels_mut().zip(mask.as_slice()) {
        let class = usize::try_from(class)?;
        if class >= counts.len() {
            return Err("model returned an unknown class".into());
        }
        counts[class] += 1;
        // Deterministic visualization colors, not the model's training palette.
        *pixel = if class == 0 {
            Rgb([0, 0, 0])
        } else {
            Rgb([
                ((class * 67) % 256) as u8,
                ((class * 139) % 256) as u8,
                ((class * 197) % 256) as u8,
            ])
        };
    }
    pixels.save(&args[3])?;
    for (i, count) in counts.iter().enumerate().filter(|(_, n)| **n > 0) {
        println!("{}: {} pixels", LABELS[i], count);
    }
    println!("saved {} ({} x {})", args[3], width, height);
    Ok(())
}
