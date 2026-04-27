use std::{
    cell::UnsafeCell,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

use coreml_rs::{ComputePlatform, CoreMLModelOptions, CoreMLModelWithState, PredictRetryOptions};
use ndarray::{ArrayD, IxDyn};

#[test]
#[ignore = "load-heavy local CoreML stress test; requires target/vitae.mlpackage"]
fn retry_vitae_under_concurrent_load() {
    let model_paths = stress_model_paths();
    if model_paths.is_empty() {
        eprintln!("skipping: no local stress models found under target/*.mlpackage");
        return;
    }

    let workers = env_usize("COREML_RETRY_STRESS_WORKERS").unwrap_or(1).max(1);
    let total_iterations = env_usize("COREML_RETRY_STRESS_ITERATIONS").unwrap_or(1000);
    let max_retries = env_usize("COREML_RETRY_STRESS_MAX_RETRIES").unwrap_or(3);
    let progress_every = env_usize("COREML_RETRY_STRESS_PROGRESS_EVERY").unwrap_or(100);
    let compute_platform =
        env_compute_platform("COREML_RETRY_STRESS_COMPUTE").unwrap_or(ComputePlatform::CpuAndANE);
    let disable_experimental_mle =
        env_bool("COREML_RETRY_STRESS_DISABLE_EXPERIMENTAL_MLE").unwrap_or(false);
    let run_retry = env_bool("COREML_RETRY_STRESS_RUN_RETRY").unwrap_or(true);

    let require_baseline_failure = std::env::var_os("COREML_RETRY_STRESS_REQUIRE_BASELINE_FAILURE")
        .is_some_and(|value| value != "0");

    let baseline = stress_predict(
        &model_paths,
        workers,
        total_iterations,
        progress_every,
        "baseline",
        compute_platform,
        disable_experimental_mle,
        PredictRetryOptions::none(),
    );
    let retry = run_retry.then(|| {
        stress_predict(
            &model_paths,
            workers,
            total_iterations,
            progress_every,
            "retry",
            compute_platform,
            disable_experimental_mle,
            PredictRetryOptions::fixed(max_retries, Duration::from_millis(10)),
        )
    });

    eprintln!(
        "coreml stress: models={}, workers={workers}, total_iterations={total_iterations}, compute={}, disable_experimental_mle={disable_experimental_mle}, baseline_failures={}, retry_failures={}",
        model_paths
            .iter()
            .map(|path| path.file_name().unwrap().to_string_lossy())
            .collect::<Vec<_>>()
            .join(","),
        compute_platform_name(compute_platform),
        baseline.failures,
        retry
            .as_ref()
            .map(|result| result.failures.to_string())
            .unwrap_or_else(|| "skipped".to_string())
    );
    baseline
        .timings
        .print("baseline", baseline.completed, baseline.predictions);
    if let Some(retry) = &retry {
        retry
            .timings
            .print("retry", retry.completed, retry.predictions);
    }
    print_samples("baseline", &baseline.sample_errors);
    if let Some(retry) = &retry {
        print_samples("retry", &retry.sample_errors);
    }

    if require_baseline_failure {
        assert!(
            baseline.failures > 0,
            "baseline run did not reproduce prediction failures"
        );
    }
    if let Some(retry) = retry {
        assert!(
            retry.failures <= baseline.failures,
            "retry should not increase final prediction failures"
        );
    }
}

struct StressResult {
    completed: usize,
    predictions: usize,
    failures: usize,
    sample_errors: Vec<String>,
    timings: StressTimings,
}

fn stress_predict(
    model_paths: &[PathBuf],
    workers: usize,
    total_iterations: usize,
    progress_every: usize,
    label: &'static str,
    compute_platform: ComputePlatform,
    disable_experimental_mle: bool,
    retry_options: PredictRetryOptions,
) -> StressResult {
    let start = Instant::now();
    let completed = Arc::new(AtomicUsize::new(0));
    let predictions = Arc::new(AtomicUsize::new(0));
    let failures = Arc::new(AtomicUsize::new(0));
    let sample_errors = Arc::new(Mutex::new(Vec::new()));
    let timings = Arc::new(StressTimings::default());
    let mut handles = Vec::with_capacity(workers);

    let models = Arc::new(load_stress_models(
        model_paths,
        compute_platform,
        disable_experimental_mle,
        &timings,
    ));

    for worker_idx in 0..workers {
        let worker_iterations = worker_iterations(total_iterations, workers, worker_idx);
        let models = Arc::clone(&models);
        let completed = Arc::clone(&completed);
        let predictions = Arc::clone(&predictions);
        let failures = Arc::clone(&failures);
        let sample_errors = Arc::clone(&sample_errors);
        let timings = Arc::clone(&timings);
        handles.push(thread::spawn(move || {
            let mut rng = XorShift64::new(0x4d595df4d0f33173 ^ worker_idx as u64);

            for _ in 0..worker_iterations {
                for stress_model in models.iter() {
                    let input_start = Instant::now();
                    let input = ArrayD::<f32>::from_shape_fn(
                        IxDyn(stress_model.input_shape.as_slice()),
                        |_| rng.next_f32(),
                    );
                    timings
                        .input_ns
                        .fetch_add(duration_ns(input_start.elapsed()), Ordering::Relaxed);

                    let bind_start = Instant::now();
                    let model = stress_model.model.get();
                    if let Err(err) = model.add_input(stress_model.input_name.as_str(), input) {
                        eprintln!("failed to add input for {}: {err}", stress_model.name);
                        failures.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                    timings
                        .bind_ns
                        .fetch_add(duration_ns(bind_start.elapsed()), Ordering::Relaxed);

                    let predict_start = Instant::now();
                    if let Err(err) = model.predict_with_retry(retry_options) {
                        failures.fetch_add(1, Ordering::Relaxed);
                        let mut sample_errors = sample_errors.lock().unwrap();
                        if sample_errors.len() < 8 {
                            sample_errors.push(format!("{}: {err}", stress_model.name));
                        }
                    }
                    timings
                        .predict_ns
                        .fetch_add(duration_ns(predict_start.elapsed()), Ordering::Relaxed);
                    predictions.fetch_add(1, Ordering::Relaxed);
                }

                let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                if should_log_progress(progress_every, done, total_iterations) {
                    eprintln!("{label}: completed {done}/{total_iterations} requests");
                }
            }
        }));
    }

    for handle in handles {
        handle.join().expect("stress worker panicked");
    }

    timings
        .wall_ns
        .store(duration_ns(start.elapsed()), Ordering::Relaxed);
    let completed = completed.load(Ordering::Relaxed);
    let predictions = predictions.load(Ordering::Relaxed);
    let failures = failures.load(Ordering::Relaxed);
    let sample_errors = Arc::try_unwrap(sample_errors)
        .expect("all workers should have released sample error collector")
        .into_inner()
        .expect("sample error collector mutex should not be poisoned");
    let timings = Arc::try_unwrap(timings).expect("all workers should have released timings");

    StressResult {
        completed,
        predictions,
        failures,
        sample_errors,
        timings,
    }
}

struct StressModel {
    name: String,
    model: SharedModel,
    input_name: String,
    input_shape: Vec<usize>,
}

fn load_stress_models(
    model_paths: &[PathBuf],
    compute_platform: ComputePlatform,
    disable_experimental_mle: bool,
    timings: &StressTimings,
) -> Vec<StressModel> {
    model_paths
        .iter()
        .map(|model_path| {
            let mut options = CoreMLModelOptions::default();
            options.compute_platform = compute_platform;
            options.disable_experimental_mle = disable_experimental_mle;

            let load_start = Instant::now();
            let model = CoreMLModelWithState::new(model_path, options)
                .load()
                .unwrap_or_else(|err| panic!("failed to load {}: {err}", model_path.display()));
            timings
                .load_ns
                .fetch_add(duration_ns(load_start.elapsed()), Ordering::Relaxed);

            let shapes_start = Instant::now();
            let input_shapes = model
                .input_shapes()
                .unwrap_or_else(|err| panic!("failed to read input shapes: {err}"));
            timings
                .shape_ns
                .fetch_add(duration_ns(shapes_start.elapsed()), Ordering::Relaxed);
            let Some((input_name, input_shape)) = input_shapes.into_iter().next() else {
                panic!("model has no inputs: {}", model_path.display());
            };

            StressModel {
                name: model_path
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .into_owned(),
                model: SharedModel::new(model),
                input_name,
                input_shape,
            }
        })
        .collect()
}

fn stress_model_paths() -> Vec<PathBuf> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        manifest_dir.join("target/saliency.mlpackage"),
        manifest_dir.join("target/vitae.mlpackage"),
    ];
    candidates
        .into_iter()
        .filter(|path| path.exists())
        .collect()
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.parse().ok()
}

fn env_bool(name: &str) -> Option<bool> {
    match std::env::var(name).ok()?.as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn env_compute_platform(name: &str) -> Option<ComputePlatform> {
    match std::env::var(name).ok()?.as_str() {
        "ane" | "cpu-and-ane" | "cpu_and_ane" => Some(ComputePlatform::CpuAndANE),
        "gpu" | "cpu-and-gpu" | "cpu_and_gpu" => Some(ComputePlatform::CpuAndGpu),
        "cpu" | "cpu-only" | "cpu_only" => Some(ComputePlatform::Cpu),
        _ => None,
    }
}

fn compute_platform_name(compute_platform: ComputePlatform) -> &'static str {
    match compute_platform {
        ComputePlatform::Cpu => "cpu",
        ComputePlatform::CpuAndANE => "cpu-and-ane",
        ComputePlatform::CpuAndGpu => "cpu-and-gpu",
    }
}

fn should_log_progress(progress_every: usize, completed: usize, iterations: usize) -> bool {
    completed == iterations || (progress_every > 0 && completed % progress_every == 0)
}

fn worker_iterations(total_iterations: usize, workers: usize, worker_idx: usize) -> usize {
    let base = total_iterations / workers;
    let extra = usize::from(worker_idx < total_iterations % workers);
    base + extra
}

fn duration_ns(duration: Duration) -> u64 {
    duration.as_nanos().min(u64::MAX as u128) as u64
}

#[derive(Default, Debug)]
struct StressTimings {
    wall_ns: AtomicU64,
    load_ns: AtomicU64,
    shape_ns: AtomicU64,
    input_ns: AtomicU64,
    bind_ns: AtomicU64,
    predict_ns: AtomicU64,
}

impl StressTimings {
    fn print(&self, label: &str, completed: usize, predictions: usize) {
        let predictions = predictions.max(1) as f64;
        let wall = Duration::from_nanos(self.wall_ns.load(Ordering::Relaxed));
        let load = Duration::from_nanos(self.load_ns.load(Ordering::Relaxed));
        let shape = Duration::from_nanos(self.shape_ns.load(Ordering::Relaxed));
        let input = Duration::from_nanos(self.input_ns.load(Ordering::Relaxed));
        let bind = Duration::from_nanos(self.bind_ns.load(Ordering::Relaxed));
        let predict = Duration::from_nanos(self.predict_ns.load(Ordering::Relaxed));
        eprintln!(
            "{label} timings: requests={}, predictions={}, wall={:.2?}, load_total={:.2?}, shape_total={:.2?}, input_avg={:.2?}, bind_avg={:.2?}, predict_avg={:.2?}",
            completed,
            predictions as usize,
            wall,
            load,
            shape,
            input.div_f64(predictions),
            bind.div_f64(predictions),
            predict.div_f64(predictions),
        );
    }
}

struct SharedModel {
    model: UnsafeCell<CoreMLModelWithState>,
}

unsafe impl Send for SharedModel {}
unsafe impl Sync for SharedModel {}

impl SharedModel {
    fn new(model: CoreMLModelWithState) -> Self {
        Self {
            model: UnsafeCell::new(model),
        }
    }

    fn get(&self) -> &mut CoreMLModelWithState {
        // This ignored stress test intentionally shares one loaded model across
        // threads without an external lock to reproduce the production access
        // pattern that relies on lower-level synchronization.
        unsafe { &mut *self.model.get() }
    }
}

struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_f32(&mut self) -> f32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        (x as u32) as f32 / u32::MAX as f32
    }
}

fn print_samples(label: &str, sample_errors: &[String]) {
    if sample_errors.is_empty() {
        eprintln!("{label} sample errors: none");
        return;
    }

    eprintln!("{label} sample errors:");
    for err in sample_errors {
        eprintln!("  {err}");
    }
}
