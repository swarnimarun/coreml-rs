use anyhow::{bail, Context, Result};
use coreml_rs::{ComputePlatform, CoreMLModelOptions, CoreMLModelWithState};
use image::imageops::FilterType;
use image::{GrayImage, ImageBuffer, RgbImage, RgbaImage};
use ndarray::{Array, ArrayD, IxDyn};
use std::ffi::OsString;
use std::path::{Path, PathBuf};

fn main() -> Result<()> {
    let args = Args::parse()?;

    let mut options = CoreMLModelOptions::default();
    options.compute_platform = args.compute_platform;
    options.normalize_input_by_255 = args.normalize_input_by_255;

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
        bail!(
            "expected exactly one model input for this smoke test, found {}",
            input_shapes.len()
        );
    }

    let (input_name, input_shape) = input_shapes.into_iter().next().unwrap();
    let input_sources = if args.image_paths.is_empty() {
        vec![InputSource::Random]
    } else {
        args.image_paths
            .iter()
            .cloned()
            .map(InputSource::Image)
            .collect()
    };

    let mut predictions = Vec::new();
    for (idx, source) in input_sources.iter().enumerate() {
        let prepared = match source {
            InputSource::Image(path) => image_file_tensor(path, &input_shape, args.pixel_scale)?,
            InputSource::Random => PreparedInput {
                tensor: random_resized_image_tensor(&input_shape)?,
                source_description: "deterministic random RGB image".to_string(),
                restore: None,
            },
        };
        println!(
            "Prepared {} for input `{input_name}` with shape {:?}",
            prepared.source_description,
            prepared.tensor.shape()
        );
        model
            .add_input(&input_name, prepared.tensor)
            .map_err(|err| anyhow::anyhow!("{err}"))?;

        let output = model.predict().map_err(|err| anyhow::anyhow!("{err}"))?;
        if output.outputs.is_empty() {
            bail!("prediction returned no outputs");
        }

        let label = format!("input{}", idx + 1);
        let output_dir = args.output_dir.as_ref().map(|dir| dir.join(&label));
        println!("Prediction outputs for {label}:");
        let mut prediction_outputs = Vec::new();
        for (name, array) in output.outputs {
            let values: ArrayD<f32> = array.extract_to_tensor();
            let stats = stats(values.iter().copied())?;
            println!(
                "  {name}: shape={:?}, len={}, finite={}, min={:.6}, max={:.6}, mean={:.6}, l2_norm={:.6}",
                values.shape(),
                values.len(),
                stats.finite,
                stats.min,
                stats.max,
                stats.mean,
                stats.l2_norm
            );
            if !stats.finite || values.is_empty() {
                bail!("output `{name}` failed sanity checks");
            }
            write_output_formats(
                &name,
                &values,
                &stats,
                output_dir.as_deref(),
                prepared.restore.as_ref(),
            )?;
            prediction_outputs.push((name, values));
        }
        predictions.push(Prediction {
            label,
            source: prepared.source_description,
            outputs: prediction_outputs,
        });
    }

    if predictions.len() == 2 {
        compare_predictions(&predictions[0], &predictions[1])?;
    }

    Ok(())
}

struct Args {
    model_path: PathBuf,
    image_paths: Vec<PathBuf>,
    output_dir: Option<PathBuf>,
    pixel_scale: PixelScale,
    compute_platform: ComputePlatform,
    normalize_input_by_255: bool,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut positionals = Vec::new();
        let mut output_dir = Some(PathBuf::from("target/coreml-outputs"));
        let mut pixel_scale = PixelScale::Unit;
        let mut compute_platform = ComputePlatform::CpuAndANE;
        let mut normalize_input_by_255 = false;
        let mut args = std::env::args_os().skip(1).peekable();

        while let Some(arg) = args.next() {
            match arg.to_string_lossy().as_ref() {
                "--output-dir" => {
                    let value = args.next().context(
                        "usage: cargo run --example run_local_mlpackage -- <model> [image] [--output-dir DIR] [--no-output-files]",
                    )?;
                    output_dir = Some(PathBuf::from(value));
                }
                "--no-output-files" => output_dir = None,
                "--pixel-scale" => {
                    let value = args.next().context(
                        "usage: cargo run --example run_local_mlpackage -- <model> [image] [second-image] [--pixel-scale unit|byte] [--output-dir DIR] [--no-output-files]",
                    )?;
                    pixel_scale = PixelScale::parse(&value)?;
                }
                "--compute" => {
                    let value = args.next().context(
                        "usage: cargo run --example run_local_mlpackage -- <model> [image] [second-image] [--compute ane|gpu|cpu] [--pixel-scale unit|byte] [--output-dir DIR] [--no-output-files]",
                    )?;
                    compute_platform = parse_compute_platform(&value)?;
                }
                "--normalize-input-by-255" => normalize_input_by_255 = true,
                "--help" | "-h" => bail!(
                    "usage: cargo run --example run_local_mlpackage -- <model> [image] [second-image] [--compute ane|gpu|cpu] [--pixel-scale unit|byte] [--normalize-input-by-255] [--output-dir DIR] [--no-output-files]"
                ),
                _ => positionals.push(arg),
            }
        }

        if positionals.is_empty() || positionals.len() > 3 {
            bail!(
                "usage: cargo run --example run_local_mlpackage -- <model> [image] [second-image] [--output-dir DIR] [--no-output-files]"
            );
        }

        let model_path = PathBuf::from(&positionals[0]);
        let image_paths = positionals
            .iter()
            .skip(1)
            .map(PathBuf::from)
            .collect::<Vec<_>>();

        Ok(Self {
            model_path: resolve_path(&model_path, "model")?,
            image_paths: image_paths
                .iter()
                .map(|path| resolve_path(path, "image"))
                .collect::<Result<Vec<_>>>()?,
            output_dir,
            pixel_scale,
            compute_platform,
            normalize_input_by_255,
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

#[derive(Clone, Copy)]
enum PixelScale {
    Unit,
    Byte,
}

impl PixelScale {
    fn parse(value: &OsString) -> Result<Self> {
        match value.to_string_lossy().as_ref() {
            "unit" | "0-1" => Ok(Self::Unit),
            "byte" | "0-255" => Ok(Self::Byte),
            value => bail!("unknown pixel scale `{value}`, expected `unit` or `byte`"),
        }
    }

    fn convert(self, value: u8) -> f32 {
        match self {
            Self::Unit => value as f32 / 255.0,
            Self::Byte => value as f32,
        }
    }

    fn description(self) -> &'static str {
        match self {
            Self::Unit => "0..1",
            Self::Byte => "0..255",
        }
    }
}

enum InputSource {
    Image(PathBuf),
    Random,
}

struct Prediction {
    label: String,
    source: String,
    outputs: Vec<(String, ArrayD<f32>)>,
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

struct PreparedInput {
    tensor: ArrayD<f32>,
    source_description: String,
    restore: Option<RestoreInfo>,
}

#[derive(Clone)]
struct RestoreInfo {
    original_path: PathBuf,
    original_width: u32,
    original_height: u32,
    content_x: u32,
    content_y: u32,
    content_width: u32,
    content_height: u32,
}

fn image_file_tensor(
    path: &Path,
    shape: &[usize],
    pixel_scale: PixelScale,
) -> Result<PreparedInput> {
    let layout = ImageLayout::from_shape(shape)?;
    let source = image::open(path)
        .with_context(|| format!("failed to open image {}", path.display()))?
        .to_rgb8();
    let original_width = source.width();
    let original_height = source.height();
    let target_width = layout.width as u32;
    let target_height = layout.height as u32;
    let scale = (target_width as f32 / original_width as f32)
        .min(target_height as f32 / original_height as f32);
    let content_width = ((original_width as f32 * scale).round() as u32).max(1);
    let content_height = ((original_height as f32 * scale).round() as u32).max(1);
    let content_x = (target_width - content_width) / 2;
    let content_y = (target_height - content_height) / 2;
    let resized =
        image::imageops::resize(&source, content_width, content_height, FilterType::Triangle);
    let mut letterboxed = RgbImage::new(target_width, target_height);
    image::imageops::replace(
        &mut letterboxed,
        &resized,
        i64::from(content_x),
        i64::from(content_y),
    );

    let mut tensor = Array::zeros(IxDyn(shape));
    for y in 0..layout.height {
        for x in 0..layout.width {
            let pixel = letterboxed.get_pixel(x as u32, y as u32).0;
            write_pixel(
                &mut tensor,
                &layout,
                y,
                x,
                [
                    pixel_scale.convert(pixel[0]),
                    pixel_scale.convert(pixel[1]),
                    pixel_scale.convert(pixel[2]),
                ],
            );
        }
    }

    Ok(PreparedInput {
        tensor,
        source_description: format!(
            "{} letterboxed from {}x{} to {}x{} with {} RGB pixels",
            path.display(),
            original_width,
            original_height,
            target_width,
            target_height,
            pixel_scale.description()
        ),
        restore: Some(RestoreInfo {
            original_path: path.to_path_buf(),
            original_width,
            original_height,
            content_x,
            content_y,
            content_width,
            content_height,
        }),
    })
}

fn write_output_formats(
    name: &str,
    values: &ArrayD<f32>,
    stats: &Stats,
    output_dir: Option<&Path>,
    restore: Option<&RestoreInfo>,
) -> Result<()> {
    let Some(output_dir) = output_dir else {
        return Ok(());
    };

    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;

    let base = output_dir.join(sanitize_filename(name));
    let raw_path = base.with_extension("f32");
    let raw_bytes = bytemuck::cast_slice(
        values
            .as_slice()
            .context("output array is not contiguous; cannot dump raw buffer")?,
    );
    std::fs::write(&raw_path, raw_bytes)
        .with_context(|| format!("failed to write {}", raw_path.display()))?;
    println!("    raw f32 buffer: {}", raw_path.display());

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
    println!("    text preview: {}", preview_path.display());

    if let Some(image) = output_image(values)? {
        let image_path = base.with_extension("png");
        image
            .save(&image_path)
            .with_context(|| format!("failed to write {}", image_path.display()))?;
        println!("    image: {}", image_path.display());

        if let Some(restore) = restore {
            write_restored_mask_outputs(&base, &image.to_luma8(), restore)?;
        }
    }

    Ok(())
}

fn compare_predictions(lhs: &Prediction, rhs: &Prediction) -> Result<()> {
    println!(
        "Comparing outputs:\n  {}: {}\n  {}: {}",
        lhs.label, lhs.source, rhs.label, rhs.source
    );

    for (name, lhs_values) in &lhs.outputs {
        let Some((_, rhs_values)) = rhs.outputs.iter().find(|(rhs_name, _)| rhs_name == name)
        else {
            println!("  {name}: skipped, missing from {}", rhs.label);
            continue;
        };

        if lhs_values.shape() != rhs_values.shape() {
            println!(
                "  {name}: skipped, shape mismatch {:?} vs {:?}",
                lhs_values.shape(),
                rhs_values.shape()
            );
            continue;
        }

        let lhs_slice = lhs_values
            .as_slice()
            .context("left output array is not contiguous")?;
        let rhs_slice = rhs_values
            .as_slice()
            .context("right output array is not contiguous")?;
        let metrics = vector_metrics(lhs_slice, rhs_slice)?;
        println!(
            "  {name}: cosine={:.6}, l2_distance={:.6}, mean_abs_diff={:.6}, max_abs_diff={:.6}",
            metrics.cosine, metrics.l2_distance, metrics.mean_abs_diff, metrics.max_abs_diff
        );

        if metrics.cosine < 0.7 {
            println!(
                "    warning: cosine similarity is lower than expected for fairly similar images"
            );
        }
    }

    Ok(())
}

struct VectorMetrics {
    cosine: f32,
    l2_distance: f32,
    mean_abs_diff: f32,
    max_abs_diff: f32,
}

fn vector_metrics(lhs: &[f32], rhs: &[f32]) -> Result<VectorMetrics> {
    if lhs.len() != rhs.len() {
        bail!(
            "cannot compare vectors with lengths {} and {}",
            lhs.len(),
            rhs.len()
        );
    }
    if lhs.is_empty() {
        bail!("cannot compare empty vectors");
    }

    let mut dot = 0.0f64;
    let mut lhs_norm = 0.0f64;
    let mut rhs_norm = 0.0f64;
    let mut l2_distance = 0.0f64;
    let mut abs_diff = 0.0f64;
    let mut max_abs_diff = 0.0f32;

    for (&l, &r) in lhs.iter().zip(rhs) {
        dot += (l as f64) * (r as f64);
        lhs_norm += (l as f64) * (l as f64);
        rhs_norm += (r as f64) * (r as f64);
        let diff = l - r;
        l2_distance += (diff as f64) * (diff as f64);
        let abs = diff.abs();
        abs_diff += abs as f64;
        max_abs_diff = max_abs_diff.max(abs);
    }

    let denom = lhs_norm.sqrt() * rhs_norm.sqrt();
    let cosine = if denom == 0.0 { 0.0 } else { dot / denom };

    Ok(VectorMetrics {
        cosine: cosine as f32,
        l2_distance: l2_distance.sqrt() as f32,
        mean_abs_diff: (abs_diff / lhs.len() as f64) as f32,
        max_abs_diff,
    })
}

fn write_restored_mask_outputs(base: &Path, mask: &GrayImage, restore: &RestoreInfo) -> Result<()> {
    let cropped = image::imageops::crop_imm(
        mask,
        restore.content_x,
        restore.content_y,
        restore.content_width,
        restore.content_height,
    )
    .to_image();
    let restored = image::imageops::resize(
        &cropped,
        restore.original_width,
        restore.original_height,
        FilterType::Triangle,
    );

    let mask_path = sibling_with_suffix(base, "original-size-mask", "png");
    restored
        .save(&mask_path)
        .with_context(|| format!("failed to write {}", mask_path.display()))?;
    println!("    original-size mask: {}", mask_path.display());

    let original = image::open(&restore.original_path)
        .with_context(|| format!("failed to reopen {}", restore.original_path.display()))?
        .to_rgba8();
    let mut overlay: RgbaImage = ImageBuffer::new(restore.original_width, restore.original_height);
    for y in 0..restore.original_height {
        for x in 0..restore.original_width {
            let source = original.get_pixel(x, y).0;
            let mask = restored.get_pixel(x, y).0[0] as f32 / 255.0;
            let alpha = 0.45 * mask;
            overlay.put_pixel(
                x,
                y,
                image::Rgba([
                    ((source[0] as f32) * (1.0 - alpha) + 255.0 * alpha).round() as u8,
                    ((source[1] as f32) * (1.0 - alpha)).round() as u8,
                    ((source[2] as f32) * (1.0 - alpha)).round() as u8,
                    source[3],
                ]),
            );
        }
    }

    let overlay_path = sibling_with_suffix(base, "overlay", "png");
    overlay
        .save(&overlay_path)
        .with_context(|| format!("failed to write {}", overlay_path.display()))?;
    println!("    overlay: {}", overlay_path.display());

    Ok(())
}

fn sibling_with_suffix(base: &Path, suffix: &str, extension: &str) -> PathBuf {
    let stem = base
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or("output");
    base.with_file_name(format!("{stem}.{suffix}.{extension}"))
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

fn random_resized_image_tensor(shape: &[usize]) -> Result<ArrayD<f32>> {
    if shape.len() != 3 && shape.len() != 4 {
        bail!("expected a 3D or 4D image tensor input, found shape {shape:?}");
    }

    let layout = ImageLayout::from_shape(shape)?;
    let (h, w) = (layout.height, layout.width);

    let source_h = 375;
    let source_w = 503;
    let mut seed = 0x1234_5678_9abc_def0_u64;
    let source: Vec<[f32; 3]> = (0..source_h * source_w)
        .map(|_| {
            [
                next_unit_f32(&mut seed),
                next_unit_f32(&mut seed),
                next_unit_f32(&mut seed),
            ]
        })
        .collect();

    let mut tensor = Array::zeros(IxDyn(shape));
    for y in 0..h {
        let src_y = y * source_h / h;
        for x in 0..w {
            let src_x = x * source_w / w;
            let pixel = source[src_y * source_w + src_x];
            write_pixel(&mut tensor, &layout, y, x, pixel);
        }
    }

    Ok(tensor)
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

fn next_unit_f32(seed: &mut u64) -> f32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    ((*seed >> 40) as f32) / ((1_u32 << 24) as f32)
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
