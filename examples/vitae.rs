use anyhow::{bail, Context, Result};
use coreml_rs::{ComputePlatform, CoreMLModelOptions, CoreMLModelWithState};
use image::{GrayImage, RgbImage};
use ndarray::{Array, ArrayD, IxDyn};
use std::ffi::OsString;
use std::path::{Path, PathBuf};

fn main() -> Result<()> {
    let args = Args::parse()?;

    let mut options = CoreMLModelOptions::default();
    options.compute_platform = args.compute_platform;
    options.disable_experimental_mle = args.disable_experimental_mle;

    let mut model = CoreMLModelWithState::new(&args.model_path, options)
        .load()
        .map_err(|err| anyhow::anyhow!("{err}"))?;

    println!("Model loaded: {}", args.model_path.display());
    let description = model
        .description()
        .map_err(|err| anyhow::anyhow!("{err}"))?;
    println!("Inputs:");
    for input in description.get("input").into_iter().flatten() {
        println!("  {input}");
    }
    println!("Outputs:");
    for output in description.get("output").into_iter().flatten() {
        println!("  {output}");
    }

    let input_shapes = model
        .input_shapes()
        .map_err(|err| anyhow::anyhow!("{err}"))?;
    if input_shapes.len() != 1 {
        bail!("expected exactly one input, found {}", input_shapes.len());
    }
    let (input_name, input_shape) = input_shapes.into_iter().next().unwrap();
    let tensor = image_file_tensor(&args.image_path, &input_shape)?;
    println!(
        "Prepared image for `{input_name}` with shape {:?}",
        tensor.shape()
    );

    model
        .add_input(&input_name, tensor)
        .map_err(|err| anyhow::anyhow!("{err}"))?;

    let output = model.predict().map_err(|err| anyhow::anyhow!("{err}"))?;
    if output.outputs.is_empty() {
        bail!("prediction returned no outputs");
    }

    std::fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("failed to create {}", args.output_dir.display()))?;

    for (name, array) in output.outputs {
        let values: ArrayD<f32> = array.extract_to_tensor();
        let stats = stats(values.iter().copied())?;
        println!(
            "{name}: shape={:?}, len={}, finite={}, min={:.6}, max={:.6}, mean={:.6}, l2_norm={:.6}",
            values.shape(),
            values.len(),
            stats.finite,
            stats.min,
            stats.max,
            stats.mean,
            stats.l2_norm
        );
        write_output_files(&args.output_dir, &name, &values, &stats)?;
    }

    Ok(())
}

struct Args {
    model_path: PathBuf,
    image_path: PathBuf,
    output_dir: PathBuf,
    compute_platform: ComputePlatform,
    disable_experimental_mle: bool,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut model_path = PathBuf::from("vitae.mlpackage");
        let mut output_dir = PathBuf::from("target/vitae-output");
        let mut compute_platform = ComputePlatform::CpuAndANE;
        let mut disable_experimental_mle = false;
        let mut image_path = None;
        let mut args = std::env::args_os().skip(1);

        while let Some(arg) = args.next() {
            match arg.to_string_lossy().as_ref() {
                "--model" => {
                    let value = args.next().context(
                        "usage: cargo run --example vitae -- <image> [--model MODEL] [--output-dir DIR]",
                    )?;
                    model_path = PathBuf::from(value);
                }
                "--output-dir" => {
                    let value = args.next().context(
                        "usage: cargo run --example vitae -- <image> [--model MODEL] [--output-dir DIR]",
                    )?;
                    output_dir = PathBuf::from(value);
                }
                "--compute" => {
                    let value = args.next().context(
                        "usage: cargo run --example vitae -- <image> [--model MODEL] [--compute ane|gpu|cpu] [--disable-experimental-mle] [--output-dir DIR]",
                    )?;
                    compute_platform = parse_compute_platform(&value)?;
                }
                "--disable-experimental-mle" => disable_experimental_mle = true,
                "--help" | "-h" => {
                    bail!(
                        "usage: cargo run --example vitae -- <image> [--model MODEL] [--compute ane|gpu|cpu] [--disable-experimental-mle] [--output-dir DIR]"
                    )
                }
                value if value.starts_with("--") => bail!("unknown option `{value}`"),
                _ => {
                    if image_path.is_some() {
                        bail!(
                            "usage: cargo run --example vitae -- <image> [--model MODEL] [--compute ane|gpu|cpu] [--disable-experimental-mle] [--output-dir DIR]"
                        );
                    }
                    image_path = Some(PathBuf::from(arg));
                }
            }
        }

        Ok(Self {
            model_path: resolve_path(&model_path, "model")?,
            image_path: resolve_path(
                image_path.as_ref().context(
                    "usage: cargo run --example vitae -- <image> [--model MODEL] [--compute ane|gpu|cpu] [--disable-experimental-mle] [--output-dir DIR]",
                )?,
                "image",
            )?,
            output_dir,
            compute_platform,
            disable_experimental_mle,
        })
    }
}

fn parse_compute_platform(value: &OsString) -> Result<ComputePlatform> {
    match value.to_string_lossy().as_ref() {
        "ane" | "cpu-and-ane" | "cpu_and_ane" => Ok(ComputePlatform::CpuAndANE),
        "gpu" | "cpu-and-gpu" | "cpu_and_gpu" => Ok(ComputePlatform::CpuAndGpu),
        "cpu" | "cpu-only" | "cpu_only" => Ok(ComputePlatform::Cpu),
        value => bail!("unknown compute platform `{value}`, expected `ane`, `gpu`, or `cpu`"),
    }
}

fn resolve_path(path: &Path, label: &str) -> Result<PathBuf> {
    if path.is_absolute() {
        return path
            .canonicalize()
            .with_context(|| format!("failed to canonicalize {label} path {}", path.display()));
    }

    if let Ok(canonical) = path.canonicalize() {
        return Ok(canonical);
    }

    let manifest_relative = Path::new(env!("CARGO_MANIFEST_DIR")).join(path);
    manifest_relative.canonicalize().with_context(|| {
        format!(
            "failed to canonicalize {label} path {} from current directory or {}",
            path.display(),
            manifest_relative.display()
        )
    })
}

fn image_file_tensor(path: &Path, shape: &[usize]) -> Result<ArrayD<f32>> {
    let layout = ImageLayout::from_shape(shape)?;
    let source = image::open(path)
        .with_context(|| format!("failed to open image {}", path.display()))?
        .to_rgb8();
    let resized = image::imageops::resize(
        &source,
        layout.width as u32,
        layout.height as u32,
        image::imageops::FilterType::Triangle,
    );

    let mut tensor = Array::zeros(IxDyn(shape));
    for y in 0..layout.height {
        for x in 0..layout.width {
            let pixel = resized.get_pixel(x as u32, y as u32).0;
            write_pixel(
                &mut tensor,
                &layout,
                y,
                x,
                [
                    pixel[0] as f32 / 255.0,
                    pixel[1] as f32 / 255.0,
                    pixel[2] as f32 / 255.0,
                ],
            );
        }
    }

    Ok(tensor)
}

fn write_output_files(
    output_dir: &Path,
    name: &str,
    values: &ArrayD<f32>,
    stats: &Stats,
) -> Result<()> {
    let base = output_dir.join(sanitize_filename(name));

    let raw_path = base.with_extension("f32");
    let raw_bytes = bytemuck::cast_slice(
        values
            .as_slice()
            .context("output array is not contiguous; cannot dump raw buffer")?,
    );
    std::fs::write(&raw_path, raw_bytes)
        .with_context(|| format!("failed to write {}", raw_path.display()))?;
    println!("  raw f32 buffer: {}", raw_path.display());

    let preview_path = base.with_extension("txt");
    let preview = values
        .iter()
        .take(32)
        .map(|value| format!("{value:.6}"))
        .collect::<Vec<_>>()
        .join(", ");
    std::fs::write(
        &preview_path,
        format!(
            "shape={:?}\nlen={}\nmin={:.6}\nmax={:.6}\nmean={:.6}\nfirst_values=[{}]\n",
            values.shape(),
            values.len(),
            stats.min,
            stats.max,
            stats.mean,
            preview
        ),
    )
    .with_context(|| format!("failed to write {}", preview_path.display()))?;
    println!("  text preview: {}", preview_path.display());

    let Some(image) = output_image(values)? else {
        println!("  skipped image save for non-image-shaped output");
        return Ok(());
    };

    let image_path = base.with_extension("png");
    image
        .save(&image_path)
        .with_context(|| format!("failed to write {}", image_path.display()))?;
    println!("  image: {}", image_path.display());

    Ok(())
}

fn output_image(values: &ArrayD<f32>) -> Result<Option<image::DynamicImage>> {
    let shape = values.shape();
    let image = match shape {
        [height, width] => {
            let mut image = GrayImage::new(*width as u32, *height as u32);
            for y in 0..*height {
                for x in 0..*width {
                    image.put_pixel(
                        x as u32,
                        y as u32,
                        image::Luma([unit_to_u8(values[[y, x]])]),
                    );
                }
            }
            image::DynamicImage::ImageLuma8(image)
        }
        [height, width, 3] => {
            let mut image = RgbImage::new(*width as u32, *height as u32);
            for y in 0..*height {
                for x in 0..*width {
                    image.put_pixel(
                        x as u32,
                        y as u32,
                        image::Rgb([
                            unit_to_u8(values[[y, x, 0]]),
                            unit_to_u8(values[[y, x, 1]]),
                            unit_to_u8(values[[y, x, 2]]),
                        ]),
                    );
                }
            }
            image::DynamicImage::ImageRgb8(image)
        }
        [1, 3, height, width] => {
            let mut image = RgbImage::new(*width as u32, *height as u32);
            for y in 0..*height {
                for x in 0..*width {
                    image.put_pixel(
                        x as u32,
                        y as u32,
                        image::Rgb([
                            unit_to_u8(values[[0, 0, y, x]]),
                            unit_to_u8(values[[0, 1, y, x]]),
                            unit_to_u8(values[[0, 2, y, x]]),
                        ]),
                    );
                }
            }
            image::DynamicImage::ImageRgb8(image)
        }
        [1, height, width, 3] => {
            let mut image = RgbImage::new(*width as u32, *height as u32);
            for y in 0..*height {
                for x in 0..*width {
                    image.put_pixel(
                        x as u32,
                        y as u32,
                        image::Rgb([
                            unit_to_u8(values[[0, y, x, 0]]),
                            unit_to_u8(values[[0, y, x, 1]]),
                            unit_to_u8(values[[0, y, x, 2]]),
                        ]),
                    );
                }
            }
            image::DynamicImage::ImageRgb8(image)
        }
        _ => return Ok(None),
    };

    Ok(Some(image))
}

fn unit_to_u8(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn sanitize_filename(name: &str) -> OsString {
    let sanitized = name
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>();
    OsString::from(if sanitized.is_empty() {
        "output".to_string()
    } else {
        sanitized
    })
}

fn write_pixel(
    tensor: &mut ArrayD<f32>,
    layout: &ImageLayout,
    y: usize,
    x: usize,
    pixel: [f32; 3],
) {
    for channel in 0..3 {
        match layout.kind {
            ImageLayoutKind::Hwc => tensor[[y, x, channel]] = pixel[channel],
            ImageLayoutKind::Nchw => tensor[[0, channel, y, x]] = pixel[channel],
            ImageLayoutKind::Nhwc => tensor[[0, y, x, channel]] = pixel[channel],
        }
    }
}

enum ImageLayoutKind {
    Hwc,
    Nchw,
    Nhwc,
}

struct ImageLayout {
    kind: ImageLayoutKind,
    height: usize,
    width: usize,
}

impl ImageLayout {
    fn from_shape(shape: &[usize]) -> Result<Self> {
        match shape {
            [height, width, 3] => Ok(Self {
                kind: ImageLayoutKind::Hwc,
                height: *height,
                width: *width,
            }),
            [1, 3, height, width] => Ok(Self {
                kind: ImageLayoutKind::Nchw,
                height: *height,
                width: *width,
            }),
            [1, height, width, 3] => Ok(Self {
                kind: ImageLayoutKind::Nhwc,
                height: *height,
                width: *width,
            }),
            _ => bail!(
                "expected image shape [H, W, 3], [1, 3, H, W], or [1, H, W, 3], found {shape:?}"
            ),
        }
    }
}

struct Stats {
    finite: bool,
    min: f32,
    max: f32,
    mean: f32,
    l2_norm: f32,
}

fn stats(values: impl Iterator<Item = f32>) -> Result<Stats> {
    let mut count = 0usize;
    let mut sum = 0.0f64;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut finite = true;
    let mut squared_sum = 0.0f64;

    for value in values {
        finite &= value.is_finite();
        min = min.min(value);
        max = max.max(value);
        sum += value as f64;
        squared_sum += (value as f64) * (value as f64);
        count += 1;
    }

    if count == 0 {
        bail!("cannot compute stats for an empty output");
    }

    Ok(Stats {
        finite,
        min,
        max,
        mean: (sum / count as f64) as f32,
        l2_norm: squared_sum.sqrt() as f32,
    })
}
