use crate::MetricsHandle;
use crate::{Error, Inputs, ModelMetrics, MultiArray, Outputs, PreparedInputs, Result};
use objc2::{
    rc::Retained,
    runtime::{AnyClass, AnyObject, NSObjectProtocol, ProtocolObject},
    AnyThread,
};
use objc2_core_ml::{
    MLArrayBatchProvider, MLBatchProvider, MLComputeUnits, MLFeatureDescription, MLFeatureProvider,
    MLFeatureType, MLModel, MLModelConfiguration, MLMultiArrayShapeConstraintType,
    MLPredictionOptions, MLState,
};
use objc2_foundation::{NSArray, NSDictionary, NSProcessInfo, NSRange, NSString, NSURL};
use objc2_metal::MTLDevice;
use std::{
    ffi::{CStr, CString},
    os::unix::ffi::{OsStrExt, OsStringExt},
    path::{Path, PathBuf},
    rc::Rc,
};

/// Hardware categories Core ML is allowed to use. Core ML decides placement.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ComputeUnits {
    CpuOnly,
    CpuAndGpu,
    CpuAndNeuralEngine,
    #[default]
    All,
}

/// Load-time configuration.
#[derive(Clone, Debug, Default)]
pub struct ModelOptions {
    pub compute_units: ComputeUnits,
    pub allow_low_precision_accumulation_on_gpu: bool,
    /// Label visible to Instruments on macOS 13 and later.
    pub display_name: Option<String>,
    /// Named function in a multifunction model, supported on macOS 15 and later.
    pub function_name: Option<String>,
}

/// Optional output buffers proposed to Core ML for a prediction.
/// Core ML may ignore a backing. Compare the returned array with the proposed
/// array using `MultiArray::is_same_object` before assuming it was used.
#[derive(Clone, Debug, Default)]
pub struct PredictionOptions {
    output_backings: std::collections::BTreeMap<String, MultiArray>,
}
impl PredictionOptions {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_output(mut self, name: impl Into<String>, array: MultiArray) -> Self {
        self.output_backings.insert(name.into(), array);
        self
    }
    fn native(&self) -> Retained<MLPredictionOptions> {
        let names: Vec<_> = self
            .output_backings
            .keys()
            .map(|s| NSString::from_str(s))
            .collect();
        let keys: Vec<_> = names.iter().map(|s| &**s).collect();
        let objects: Vec<&AnyObject> = self
            .output_backings
            .values()
            .map(|v| AsRef::<AnyObject>::as_ref(&*v.0))
            .collect();
        let dict = NSDictionary::from_slices(&keys, &objects);
        // SAFETY: every dictionary value is an MLMultiArray, as required by Core ML.
        unsafe {
            let options = MLPredictionOptions::new();
            options.setOutputBackings(&dict);
            options
        }
    }
}

/// Runtime features and system-wide capacity, not per-model hardware usage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Capabilities {
    pub macos_version: (u32, u32, u32),
    pub stateful_prediction: bool,
    pub cpu_and_neural_engine: bool,
    pub compute_device_enumeration: bool,
    /// Core ML compute plans are available on macOS 14.4 and later.
    pub compute_plan: bool,
    /// In-memory specifications are supported on macOS 13 and later.
    pub in_memory_model: bool,
    /// In-memory ML Program blob mappings require macOS 15 and later.
    pub in_memory_blobs: bool,
    pub physical_memory_bytes: u64,
    pub logical_cpu_count: usize,
    pub active_cpu_count: usize,
    /// Current system thermal pressure, not attribution to this model.
    pub thermal_state: ThermalState,
    /// Devices Core ML may use, or None when enumeration is unavailable.
    pub compute_devices: Option<Vec<ComputeDevice>>,
}
pub(crate) fn macos_version() -> (u32, u32, u32) {
    static VERSION: std::sync::OnceLock<(u32, u32, u32)> = std::sync::OnceLock::new();
    *VERSION.get_or_init(|| {
        let version = NSProcessInfo::processInfo().operatingSystemVersion();
        (
            version.majorVersion as u32,
            version.minorVersion as u32,
            version.patchVersion as u32,
        )
    })
}
impl Capabilities {
    pub fn current() -> Self {
        let info = NSProcessInfo::processInfo();
        let v = macos_version();
        Self {
            macos_version: v,
            stateful_prediction: v >= (15, 0, 0),
            cpu_and_neural_engine: v >= (13, 0, 0),
            compute_device_enumeration: v >= (14, 0, 0),
            compute_plan: v >= (14, 4, 0),
            in_memory_model: v >= (13, 0, 0),
            in_memory_blobs: v >= (15, 0, 0),
            physical_memory_bytes: info.physicalMemory(),
            logical_cpu_count: info.processorCount(),
            active_cpu_count: info.activeProcessorCount(),
            thermal_state: ThermalState::from(info.thermalState().0),
            compute_devices: if v >= (14, 0, 0) {
                Some(compute_devices())
            } else {
                None
            },
        }
    }
}

/// Current operating system thermal pressure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThermalState {
    Nominal,
    Fair,
    Serious,
    Critical,
    Unknown(isize),
}
impl From<isize> for ThermalState {
    fn from(value: isize) -> Self {
        match value {
            0 => Self::Nominal,
            1 => Self::Fair,
            2 => Self::Serious,
            3 => Self::Critical,
            n => Self::Unknown(n),
        }
    }
}

/// Devices visible to Core ML. GPU memory is a performance recommendation,
/// not a reservation, hard limit, or measurement of current usage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComputeDevice {
    Cpu,
    NeuralEngine {
        total_cores: usize,
    },
    Gpu {
        name: String,
        recommended_working_set_bytes: u64,
        unified_memory: bool,
    },
    Unknown,
}

fn compute_devices() -> Vec<ComputeDevice> {
    // SAFETY: the caller checked the macOS 14 availability of this selector.
    let devices = unsafe { MLModel::availableComputeDevices() };
    devices
        .iter()
        .map(|device| describe_compute_device(&device))
        .collect()
}

pub(crate) fn describe_compute_device(
    device: &ProtocolObject<dyn objc2_core_ml::MLComputeDeviceProtocol>,
) -> ComputeDevice {
    let class = |name: &'static [u8]| {
        AnyClass::get(CStr::from_bytes_with_nul(name).expect("static class name"))
    };
    let cpu = class(b"MLCPUComputeDevice\0");
    let gpu = class(b"MLGPUComputeDevice\0");
    let ane = class(b"MLNeuralEngineComputeDevice\0");
    if cpu.is_some_and(|c| device.isKindOfClass(c)) {
        ComputeDevice::Cpu
    } else if ane.is_some_and(|c| device.isKindOfClass(c)) {
        // SAFETY: this object is a Core ML Neural Engine device.
        let count: isize = unsafe { objc2::msg_send![device, totalCoreCount] };
        ComputeDevice::NeuralEngine {
            total_cores: count.max(0) as usize,
        }
    } else if gpu.is_some_and(|c| device.isKindOfClass(c)) {
        // SAFETY: this object is a Core ML GPU device and returns an MTLDevice.
        let metal: Option<Retained<ProtocolObject<dyn MTLDevice>>> =
            unsafe { objc2::msg_send![device, metalDevice] };
        match metal {
            Some(metal) => ComputeDevice::Gpu {
                name: metal.name().to_string(),
                recommended_working_set_bytes: metal.recommendedMaxWorkingSetSize(),
                unified_memory: metal.hasUnifiedMemory(),
            },
            None => ComputeDevice::Unknown,
        }
    } else {
        ComputeDevice::Unknown
    }
}

fn file_url(path: &Path) -> Result<Retained<NSURL>> {
    let bytes = CString::new(path.as_os_str().as_bytes())
        .map_err(|_| Error::InvalidPath(path.to_owned()))?;
    // SAFETY: the C string is NUL-terminated and remains alive during the call.
    Ok(unsafe {
        NSURL::fileURLWithFileSystemRepresentation_isDirectory_relativeToURL(
            std::ptr::NonNull::new_unchecked(bytes.as_ptr() as *mut _),
            path.is_dir(),
            None,
        )
    })
}

fn model_configuration(options: &ModelOptions) -> Result<Retained<MLModelConfiguration>> {
    let version = macos_version();
    if options.compute_units == ComputeUnits::CpuAndNeuralEngine && version < (13, 0, 0) {
        return Err(Error::Unsupported(
            "CPU and Neural Engine requires macOS 13",
        ));
    }
    if options.display_name.is_some() && version < (13, 0, 0) {
        return Err(Error::Unsupported("model display name requires macOS 13"));
    }
    if options.function_name.is_some() && version < (15, 0, 0) {
        return Err(Error::Unsupported("named model functions require macOS 15"));
    }
    // SAFETY: every selector is available on the checked OS version.
    unsafe {
        let config = MLModelConfiguration::new();
        let units = match options.compute_units {
            ComputeUnits::CpuOnly => MLComputeUnits::CPUOnly,
            ComputeUnits::CpuAndGpu => MLComputeUnits::CPUAndGPU,
            ComputeUnits::CpuAndNeuralEngine => MLComputeUnits::CPUAndNeuralEngine,
            ComputeUnits::All => MLComputeUnits::All,
        };
        config.setComputeUnits(units);
        if let Some(name) = options.display_name.as_deref() {
            config.setModelDisplayName(Some(&NSString::from_str(name)));
        }
        if let Some(name) = options.function_name.as_deref() {
            config.setFunctionName(Some(&NSString::from_str(name)));
        }
        config
            .setAllowLowPrecisionAccumulationOnGPU(options.allow_low_precision_accumulation_on_gpu);
        Ok(config)
    }
}

/// A compiled model in a temporary directory owned by this value.
/// Persist it explicitly before dropping it if it must survive the process.
#[derive(Debug)]
pub struct CompiledModel {
    _directory: std::sync::Arc<tempfile::TempDir>,
    path: PathBuf,
}
impl AsRef<Path> for CompiledModel {
    fn as_ref(&self) -> &Path {
        &self.path
    }
}
impl CompiledModel {
    pub fn path(&self) -> &Path {
        &self.path
    }
    /// Load while retaining the temporary compiled artifact for the model's lifetime.
    pub fn load(&self, options: ModelOptions) -> Result<Model> {
        let mut model = Model::load(&self.path, options)?;
        // The newly created model has exactly one owner at this point.
        if let ModelSource::File { artifact, .. } = &mut Rc::get_mut(&mut model.0)
            .expect("newly loaded model is unique")
            .source
        {
            *artifact = Some(self._directory.clone());
        }
        Ok(model)
    }
    /// Copy the compiled directory to a new destination. Existing files are not overwritten.
    pub fn persist(&self, destination: impl AsRef<Path>) -> Result<()> {
        let destination = destination.as_ref();
        std::fs::create_dir(destination).map_err(|e| Error::io("persist compiled model", e))?;
        if let Err(error) = copy_directory_contents(&self.path, destination) {
            // Only remove the directory this call created; existing destinations
            // are rejected by create_dir before this point.
            let _ = std::fs::remove_dir_all(destination);
            return Err(Error::io("persist compiled model", error));
        }
        Ok(())
    }
}
fn copy_directory_contents(source: &Path, dest: &Path) -> std::io::Result<()> {
    for entry in std::fs::read_dir(source)? {
        let entry = entry?;
        let target = dest.join(entry.file_name());
        let kind = entry.file_type()?;
        if kind.is_dir() {
            std::fs::create_dir(&target)?;
            copy_directory_contents(&entry.path(), &target)?;
        } else if kind.is_file() {
            std::fs::copy(entry.path(), target)?;
        } else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "compiled model contains a non-regular file",
            ));
        }
    }
    Ok(())
}

/// Core ML feature kinds, retaining unknown values for forward compatibility.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeatureType {
    Int64,
    Double,
    String,
    Image,
    MultiArray,
    Dictionary,
    Sequence,
    State,
    Unknown(isize),
}
impl From<MLFeatureType> for FeatureType {
    fn from(t: MLFeatureType) -> Self {
        match t {
            MLFeatureType::Int64 => Self::Int64,
            MLFeatureType::Double => Self::Double,
            MLFeatureType::String => Self::String,
            MLFeatureType::Image => Self::Image,
            MLFeatureType::MultiArray => Self::MultiArray,
            MLFeatureType::Dictionary => Self::Dictionary,
            MLFeatureType::Sequence => Self::Sequence,
            MLFeatureType::State => Self::State,
            _ => Self::Unknown(t.0),
        }
    }
}
/// Shape constraints as reported by Core ML. Range uses NSRange semantics:
/// `location` is inclusive and `length` is the number of allowed values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ShapeConstraint {
    Unspecified,
    Enumerated(Vec<Vec<usize>>),
    Range(Vec<(usize, usize)>),
    Unknown(isize),
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FeatureDescription {
    pub name: String,
    pub feature_type: FeatureType,
    pub optional: bool,
    /// Default multiarray shape.
    pub shape: Option<Vec<usize>>,
    pub shape_constraint: Option<ShapeConstraint>,
    /// Raw Core ML multiarray dtype value, when applicable.
    pub data_type: Option<isize>,
    /// Default width, height and required Core Video pixel format.
    pub image: Option<ImageConstraint>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageConstraint {
    pub width: usize,
    pub height: usize,
    pub pixel_format: u32,
}
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ModelDescription {
    pub inputs: Vec<FeatureDescription>,
    pub outputs: Vec<FeatureDescription>,
}
fn feature_descriptions(
    dict: &NSDictionary<NSString, MLFeatureDescription>,
) -> Vec<FeatureDescription> {
    let (keys, values) = dict.to_vecs();
    let mut descriptions: Vec<_> = keys
        .iter()
        .zip(values.iter())
        .map(|(name, value)| {
            // SAFETY: all objects are retained model descriptions from Core ML.
            unsafe {
                let constraint = value.multiArrayConstraint();
                FeatureDescription {
                    name: name.to_string(),
                    feature_type: value.r#type().into(),
                    optional: value.isOptional(),
                    shape: constraint
                        .as_ref()
                        .map(|c| c.shape().iter().map(|n| n.unsignedIntegerValue()).collect()),
                    shape_constraint: constraint.as_ref().map(|c| shape_constraint(c)),
                    data_type: constraint.as_ref().map(|c| c.dataType().0),
                    image: value.imageConstraint().map(|c| ImageConstraint {
                        width: c.pixelsWide().max(0) as usize,
                        height: c.pixelsHigh().max(0) as usize,
                        pixel_format: c.pixelFormatType(),
                    }),
                }
            }
        })
        .collect();
    descriptions.sort_by(|a, b| a.name.cmp(&b.name));
    descriptions
}

fn shape_constraint(array: &objc2_core_ml::MLMultiArrayConstraint) -> ShapeConstraint {
    // SAFETY: shape constraints and NSRange values are documented Core ML types.
    unsafe {
        let constraint = array.shapeConstraint();
        match constraint.r#type() {
            MLMultiArrayShapeConstraintType::Unspecified => ShapeConstraint::Unspecified,
            MLMultiArrayShapeConstraintType::Enumerated => ShapeConstraint::Enumerated(
                constraint
                    .enumeratedShapes()
                    .iter()
                    .map(|shape| shape.iter().map(|n| n.unsignedIntegerValue()).collect())
                    .collect(),
            ),
            MLMultiArrayShapeConstraintType::Range => ShapeConstraint::Range(
                constraint
                    .sizeRangeForDimension()
                    .iter()
                    .map(|value| {
                        let range: NSRange = objc2::msg_send![&*value, rangeValue];
                        (range.location, range.length)
                    })
                    .collect(),
            ),
            other => ShapeConstraint::Unknown(other.0),
        }
    }
}

fn state_schema(
    model: &MLModel,
) -> Result<std::collections::BTreeMap<String, (Vec<usize>, isize)>> {
    // SAFETY: callers check macOS 15 and hold the model and descriptions alive.
    unsafe {
        let descriptions = model.modelDescription().stateDescriptionsByName();
        let (names, values) = descriptions.to_vecs();
        names
            .iter()
            .zip(values.iter())
            .map(|(name, value)| {
                let constraint = value
                    .stateConstraint()
                    .ok_or(Error::ModelLoad("missing state constraint"))?;
                let shape = constraint
                    .bufferShape()
                    .iter()
                    .map(|n| n.unsignedIntegerValue())
                    .collect();
                Ok((name.to_string(), (shape, constraint.dataType().0)))
            })
            .collect()
    }
}

#[derive(Clone)]
enum ModelSource {
    File {
        url: Retained<NSURL>,
        artifact: Option<std::sync::Arc<tempfile::TempDir>>,
    },
    Memory(std::sync::Arc<crate::memory::MemoryBuffers>),
}
struct ModelInner {
    native: Retained<MLModel>,
    metrics: MetricsHandle,
    source: ModelSource,
    options: ModelOptions,
    state_family: Rc<()>,
}
/// A loaded model. Clones share the native model and metrics. It is deliberately
/// not Send or Sync until native object concurrency is proven safe.
#[derive(Clone)]
pub struct Model(Rc<ModelInner>);

/// State is tied to its creating model and functions explicitly loaded from that
/// model with matching state schemas. Predictions require mutable access.
pub struct State {
    native: Retained<MLState>,
    model: Rc<ModelInner>,
}

impl Model {
    pub(crate) fn is_exclusively_owned(&self) -> bool {
        Rc::strong_count(&self.0) == 1
    }

    pub(crate) fn idle_for_eviction(
        &self,
        inserted_at: std::time::Instant,
    ) -> Option<std::time::Duration> {
        self.0.metrics.idle_for_eviction(inserted_at)
    }
    /// Compile an .mlmodel or .mlpackage into a temporary owned artifact.
    pub fn compile(path: impl AsRef<Path>) -> Result<CompiledModel> {
        let url = file_url(path.as_ref())?;
        // SAFETY: url is a live file URL. This synchronous API is supported on the baseline OS.
        #[allow(deprecated)]
        let compiled = unsafe { MLModel::compileModelAtURL_error(&url) }
            .map_err(|e| Error::native("compile model", &e))?;
        let source = PathBuf::from(std::ffi::OsString::from_vec(
            unsafe { CStr::from_ptr(compiled.fileSystemRepresentation().as_ptr()) }
                .to_bytes()
                .to_vec(),
        ));
        let directory = tempfile::tempdir().map_err(|e| Error::io("create model directory", e))?;
        let target = directory.path().join("model.mlmodelc");
        std::fs::rename(&source, &target).map_err(|e| Error::io("retain compiled model", e))?;
        Ok(CompiledModel {
            _directory: std::sync::Arc::new(directory),
            path: target,
        })
    }
    /// Load a compiled .mlmodelc, including one encrypted using Apple's workflow.
    /// The caller must keep the compiled directory available for the model's
    /// lifetime. For a temporary `CompiledModel`, use `CompiledModel::load`.
    pub fn load(path: impl AsRef<Path>, options: ModelOptions) -> Result<Self> {
        let load_started = std::time::Instant::now();
        let configuration = model_configuration(&options)?;
        let url = file_url(path.as_ref())?;
        // SAFETY: the URL and configuration are live and supported on macOS 12.3.
        let native =
            unsafe { MLModel::modelWithContentsOfURL_configuration_error(&url, &configuration) }
                .map_err(|e| Error::native("load model", &e))?;
        Ok(Self(Rc::new(ModelInner {
            native,
            metrics: MetricsHandle::with_load_duration(load_started.elapsed()),
            source: ModelSource::File {
                url,
                artifact: None,
            },
            options,
            state_family: Rc::new(()),
        })))
    }

    /// Load an in-memory `.mlmodel` specification without staging it to a file.
    /// Requires macOS 13 or later. Core ML manages its own internal caches.
    /// The owned bytes remain alive for the lifetime of the model.
    pub async fn load_from_memory(specification: Vec<u8>, options: ModelOptions) -> Result<Self> {
        Self::load_memory(specification, None, options).await
    }

    /// Load an ML Program specification and its external weight blobs from memory.
    /// Requires macOS 15 or later. Keys are the blob paths relative to the model,
    /// for example `weights/weight.bin`, and every referenced blob is required.
    pub async fn load_from_memory_with_blobs<I, S>(
        specification: Vec<u8>,
        blobs: I,
        options: ModelOptions,
    ) -> Result<Self>
    where
        I: IntoIterator<Item = (S, Vec<u8>)>,
        S: Into<String>,
    {
        if macos_version() < (15, 0, 0) {
            return Err(Error::Unsupported(
                "in-memory model blob mappings require macOS 15",
            ));
        }
        let blobs = blobs
            .into_iter()
            .map(|(key, data)| (key.into(), data))
            .collect();
        Self::load_memory(specification, Some(blobs), options).await
    }

    async fn load_memory(
        specification: Vec<u8>,
        blobs: Option<Vec<(String, Vec<u8>)>>,
        options: ModelOptions,
    ) -> Result<Self> {
        if macos_version() < (13, 0, 0) {
            return Err(Error::Unsupported(
                "in-memory model assets require macOS 13",
            ));
        }
        let load_started = std::time::Instant::now();
        let configuration = model_configuration(&options)?;
        let buffers = crate::memory::asset(specification, blobs)?;
        let native = crate::memory::load(buffers.clone(), &configuration).await?;
        Ok(Self(Rc::new(ModelInner {
            native,
            metrics: MetricsHandle::with_load_duration(load_started.elapsed()),
            source: ModelSource::Memory(buffers),
            options,
            state_family: Rc::new(()),
        })))
    }

    /// Load another named function from the same asset and permit it to use
    /// this model's `State`. Core ML must report identical state names, shapes
    /// and data types for both functions. Requires macOS 15 or later.
    ///
    /// Each function has independent request metrics. Calls using the same
    /// state must still be serialized; `&mut State` enforces this in safe Rust.
    /// Loading a file-backed function performs a synchronous Core ML load.
    pub async fn load_function_sharing_state(&self, name: &str) -> Result<Self> {
        if macos_version() < (15, 0, 0) {
            return Err(Error::Unsupported("named model functions require macOS 15"));
        }
        let started = std::time::Instant::now();
        let mut options = self.0.options.clone();
        options.function_name = Some(name.to_owned());
        let configuration = model_configuration(&options)?;
        let native = match &self.0.source {
            ModelSource::File { url, .. } => {
                // SAFETY: the retained URL and configuration are valid for this call.
                unsafe { MLModel::modelWithContentsOfURL_configuration_error(url, &configuration) }
                    .map_err(|error| Error::native("load model function", &error))?
            }
            ModelSource::Memory(buffers) => {
                crate::memory::load(buffers.clone(), &configuration).await?
            }
        };
        let expected = state_schema(&self.0.native)?;
        if expected.is_empty() || expected != state_schema(&native)? {
            return Err(Error::ModelLoad(
                "model functions do not have matching state schemas",
            ));
        }
        Ok(Self(Rc::new(ModelInner {
            native,
            metrics: MetricsHandle::with_load_duration(started.elapsed()),
            source: self.0.source.clone(),
            options,
            state_family: self.0.state_family.clone(),
        })))
    }
    pub fn description(&self) -> ModelDescription {
        // SAFETY: the description is retained for the duration of conversion.
        unsafe {
            let d = self.0.native.modelDescription();
            ModelDescription {
                inputs: feature_descriptions(&d.inputDescriptionsByName()),
                outputs: feature_descriptions(&d.outputDescriptionsByName()),
            }
        }
    }
    /// Load an owned snapshot of the compute plan for this model's compiled
    /// artifact and configuration. Requires macOS 14.4 or later.
    ///
    /// The returned device placement and operation costs are Core ML estimates,
    /// not measurements of live utilization. Dropping this future does not
    /// cancel Core ML's underlying plan load.
    pub async fn compute_plan(&self) -> Result<crate::ComputePlan> {
        if macos_version() < (14, 4, 0) {
            return Err(Error::Unsupported("compute plans require macOS 14.4"));
        }
        // SAFETY: the retained model remains alive, and this selector is available
        // on the supported baseline. Core ML returns its load-time configuration.
        let configuration = unsafe { self.0.native.configuration() };
        let source = match &self.0.source {
            ModelSource::File { url, artifact } => crate::compute_plan::PlanSource::File {
                url,
                artifact: artifact.clone(),
            },
            ModelSource::Memory(buffers) => crate::compute_plan::PlanSource::Memory {
                asset: &buffers.asset,
                buffers: buffers.clone(),
            },
        };
        crate::compute_plan::load(source, &configuration).await
    }

    pub fn metrics(&self) -> ModelMetrics {
        self.0.metrics.snapshot()
    }
    /// Obtain a Send + Sync handle for observing this model from another thread.
    pub fn metrics_handle(&self) -> MetricsHandle {
        self.0.metrics.clone()
    }
    pub fn predict(&self, inputs: &Inputs) -> Result<Outputs> {
        let mut measurement = self.0.metrics.start(0);
        let provider = inputs.provider()?;
        let result = self.predict_provider(&provider, None)?;
        measurement.success();
        Ok(result)
    }
    /// Predict with caller-proposed output backing storage.
    pub fn predict_with_options(
        &self,
        inputs: &Inputs,
        options: &PredictionOptions,
    ) -> Result<Outputs> {
        let mut measurement = self.0.metrics.start(0);
        let provider = inputs.provider()?;
        let result = self.predict_provider(&provider, Some(options))?;
        measurement.success();
        Ok(result)
    }
    /// Predict using an already prepared native feature provider.
    pub fn predict_prepared(
        &self,
        inputs: &PreparedInputs,
        options: Option<&PredictionOptions>,
    ) -> Result<Outputs> {
        let mut measurement = self.0.metrics.start(0);
        let result = self.predict_provider(&inputs.0, options)?;
        measurement.success();
        Ok(result)
    }
    fn predict_provider(
        &self,
        provider: &objc2_core_ml::MLDictionaryFeatureProvider,
        options: Option<&PredictionOptions>,
    ) -> Result<Outputs> {
        // SAFETY: the provider and model remain live for this synchronous call.
        let output = unsafe {
            let provider = ProtocolObject::from_ref(provider);
            if let Some(options) = options {
                let options = options.native();
                self.0
                    .native
                    .predictionFromFeatures_options_error(provider, &options)
            } else {
                self.0.native.predictionFromFeatures_error(provider)
            }
        }
        .map_err(|e| Error::native("predict", &e))?;
        Outputs::from_provider(&output)
    }
    pub fn predict_batch(&self, inputs: &[Inputs]) -> Result<Vec<Outputs>> {
        let mut measurement = self.0.metrics.start(inputs.len() as u64);
        let providers: Vec<_> = inputs.iter().map(Inputs::provider).collect::<Result<_>>()?;
        let refs: Vec<&ProtocolObject<dyn MLFeatureProvider>> = providers
            .iter()
            .map(|p| ProtocolObject::from_ref(&**p))
            .collect();
        let array = NSArray::from_slice(&refs);
        // SAFETY: batch and providers remain retained until the synchronous call returns.
        let outputs = unsafe {
            let batch = MLArrayBatchProvider::initWithFeatureProviderArray(
                MLArrayBatchProvider::alloc(),
                &array,
            );
            self.0
                .native
                .predictionsFromBatch_error(ProtocolObject::from_ref(&*batch))
        }
        .map_err(|e| Error::native("batch prediction", &e))?;
        let mut result = Vec::new();
        // SAFETY: returned provider defines indices from zero up to count.
        unsafe {
            for i in 0..outputs.count() {
                result.push(Outputs::from_provider(&outputs.featuresAtIndex(i))?);
            }
        }
        measurement.success();
        Ok(result)
    }
    pub fn new_state(&self) -> Result<State> {
        if macos_version() < (15, 0, 0) {
            return Err(Error::Unsupported("stateful prediction requires macOS 15"));
        }
        // Core ML can throw for legacy models even though newer models may have empty state.
        // Check the model description first, before invoking the state factory.
        if unsafe {
            self.0
                .native
                .modelDescription()
                .stateDescriptionsByName()
                .is_empty()
        } {
            return Err(Error::Unsupported("model does not declare state"));
        }
        // SAFETY: availability was checked and the model remains owned by State.
        Ok(State {
            native: unsafe { self.0.native.newState() },
            model: self.0.clone(),
        })
    }
    pub fn predict_with_state(&self, inputs: &Inputs, state: &mut State) -> Result<Outputs> {
        let mut measurement = self.0.metrics.start(0);
        if !Rc::ptr_eq(&self.0.state_family, &state.model.state_family) {
            return Err(Error::WrongModelState);
        }
        let provider = inputs.provider()?;
        let output = self.predict_state_provider(&provider, state)?;
        measurement.success();
        Ok(output)
    }

    /// Stateful prediction using a previously prepared feature provider.
    pub fn predict_prepared_with_state(
        &self,
        inputs: &PreparedInputs,
        state: &mut State,
    ) -> Result<Outputs> {
        let mut measurement = self.0.metrics.start(0);
        if !Rc::ptr_eq(&self.0.state_family, &state.model.state_family) {
            return Err(Error::WrongModelState);
        }
        let output = self.predict_state_provider(&inputs.0, state)?;
        measurement.success();
        Ok(output)
    }

    fn predict_state_provider(
        &self,
        provider: &objc2_core_ml::MLDictionaryFeatureProvider,
        state: &mut State,
    ) -> Result<Outputs> {
        // SAFETY: the state belongs to this model's verified function family,
        // and its mutable borrow prevents overlapping safe predictions.
        let output = unsafe {
            self.0.native.predictionFromFeatures_usingState_error(
                ProtocolObject::from_ref(provider),
                &state.native,
            )
        }
        .map_err(|e| Error::native("stateful prediction", &e))?;
        Outputs::from_provider(&output)
    }
    /// Access the native model for APIs not wrapped by the crate.
    ///
    /// # Safety
    /// The caller must honor Core ML's synchronization rules and preserve state ownership.
    pub unsafe fn as_native(&self) -> &MLModel {
        &self.0.native
    }
}
