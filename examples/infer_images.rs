use std::fs;
use std::path::{Path, PathBuf};

use coreml_rs::{ComputePlatform, CoreMLModelOptions, CoreMLModelWithState, mlarray::MLArray};
use image::{io::Reader as ImageReader};
use ndarray::Array3;
use image::imageops::FilterType;

pub struct ImageDetails {
    buffer: Array3<f32>,
    width: u32,
    height: u32

}

fn load_image_to_tensor(path: &Path, width: u32, height: u32) -> Result<ImageDetails, Box<dyn std::error::Error>> {
    let img = ImageReader::open(path)?.decode()?.to_rgb8();
    // Resize to 1024x1024 as expected by the model
    let (orgwidth, orgheight) = img.dimensions();
    let img = image::imageops::resize(&img, width, height, FilterType::Lanczos3);
    
    let mut arr = Array3::<f32>::zeros((width as usize, height as usize, 3));
    for y in 0..width {
        for x in 0..height {
            let p = img.get_pixel(x, y);
            arr[[y as usize, x as usize, 0]] = p[0] as f32;
            arr[[y as usize, x as usize, 1]] = p[1] as f32;
            arr[[y as usize, x as usize, 2]] = p[2] as f32;
        }
    }
    let result = ImageDetails {
        buffer: arr,
        width: orgwidth,
        height: orgheight
    };
    Ok(result)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = Path::new("./models/model_13.mlmodel");
    let images_dir = Path::new("./data/samples/");
    let output_dir = Path::new("./output");
    fs::create_dir_all(output_dir)?;

    let buf = fs::read(model_path)?;
    let mut opts = CoreMLModelOptions::default();
    opts.compute_platform = ComputePlatform::CpuAndANE;
    let mut model = CoreMLModelWithState::from_buf(buf, opts);
    
    // Load the model
    model = model.load().map_err(|e| format!("Failed to load model: {:?}", e))?;
    let shape = model.input_shape()?;
    let (width, height) = (shape[0], shape[1]);

    let mut inputs: Vec<PathBuf> = fs::read_dir(images_dir)?
        .filter_map(|e| e.ok().map(|f| f.path()))
        .collect();
    inputs.sort();

    for img_path in inputs.iter() {
        let image_details = load_image_to_tensor(img_path, width.try_into().unwrap(), height.try_into().unwrap())?;
        model.add_input("image", image_details.buffer.into_dyn()).map_err(|e| format!("{:?}", e))?;
        let output = model.predict().map_err(|e| format!("{:?}", e))?;
        let (_, raw_output) = output.outputs.into_iter().next().expect("no output");
        
        // Convert raw output RGB array to image and save as JPEG
        let shape = raw_output.shape();
        
        // Assuming the output is in [height, width, channels] format
        let (height, width) = (shape[0] as u32, shape[1] as u32);
        
        // Extract the tensor data - already in correct uint8 format
        match raw_output {
            MLArray::Float32Array(ref array) => {
                let rgb_data: Vec<u8> = array.as_slice().unwrap()
                    .iter()
                    .map(|&x| x as u8)
                    .collect();
                
                // Create image from RGB data
                if let Some(img) = image::RgbImage::from_raw(width, height, rgb_data) {
                    let mut out_path = output_dir.join(img_path.file_stem().unwrap());
                    out_path.set_extension("jpg");
                    let img = image::imageops::resize(&img,
                        image_details.width, image_details.height, FilterType::Lanczos3);
                    img.save(&out_path)?;
                    println!("Saved: {}", out_path.display());
                } else {
                    eprintln!("Failed to create image from raw output for {}", img_path.display());
                }
            },
            MLArray::UInt8Array(ref array) => {
                let rgb_data: Vec<u8> = array.as_slice().unwrap().to_vec();
                
                // Create image from RGB data
                if let Some(img) = image::RgbImage::from_raw(width, height, rgb_data) {
                    let mut out_path = output_dir.join(img_path.file_stem().unwrap());
                    out_path.set_extension("jpg");
                    let img = image::imageops::resize(&img,
                        image_details.width, image_details.height, FilterType::Lanczos3);
                    img.save(&out_path)?;
                    println!("Saved: {}", out_path.display());
                } else {
                    eprintln!("Failed to create image from raw output for {}", img_path.display());
                }
            },
            _ => {
                eprintln!("Unexpected output type for {}", img_path.display());
            }
        }
    }
    Ok(())
}
