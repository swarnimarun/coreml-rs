use objc2_foundation::NSError;
use std::path::PathBuf;

/// A Core ML or Foundation error, preserved for diagnostics and fallback policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeError {
    pub domain: String,
    pub code: isize,
    pub description: String,
}

impl From<&NSError> for NativeError {
    fn from(value: &NSError) -> Self {
        Self {
            domain: value.domain().to_string(),
            code: value.code(),
            description: value.localizedDescription().to_string(),
        }
    }
}

impl NativeError {
    /// Core ML failed to retrieve an encrypted model's key.
    pub fn is_decryption_key_fetch(&self) -> bool {
        // SAFETY: this framework constant exists on all supported macOS versions.
        self.domain == unsafe { objc2_core_ml::MLModelErrorDomain }.to_string()
            && self.code == objc2_core_ml::MLModelError::ModelDecryptionKeyFetch.0
    }
    /// Core ML reported a model decryption failure.
    pub fn is_decryption(&self) -> bool {
        self.domain == unsafe { objc2_core_ml::MLModelErrorDomain }.to_string()
            && self.code == objc2_core_ml::MLModelError::ModelDecryption.0
    }
}

impl std::fmt::Display for NativeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({}: {})", self.description, self.domain, self.code)
    }
}

impl std::error::Error for NativeError {}

/// Errors from model management and prediction.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("{operation}: {source}")]
    Native {
        operation: &'static str,
        #[source]
        source: NativeError,
    },
    #[error("{operation}: {source}")]
    Io {
        operation: &'static str,
        #[source]
        source: std::io::Error,
    },
    #[error("path contains a NUL byte: {0:?}")]
    InvalidPath(PathBuf),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("feature {0:?} is missing")]
    MissingFeature(String),
    #[error("feature {name:?} has type {actual}, expected {expected}")]
    FeatureType {
        name: String,
        expected: &'static str,
        actual: String,
    },
    #[error("{operation} failed with platform status {code}")]
    PlatformStatus { operation: &'static str, code: i32 },
    #[error("unsupported capability: {0}")]
    Unsupported(&'static str),
    #[error("model load: {0}")]
    ModelLoad(&'static str),
    #[error("compute plan: {0}")]
    ComputePlan(&'static str),
    #[error("state was created by a different model or unrelated function")]
    WrongModelState,
    #[error(transparent)]
    Tensor(#[from] crate::TensorError),
}

impl Error {
    pub(crate) fn native(operation: &'static str, error: &NSError) -> Self {
        Self::Native {
            operation,
            source: error.into(),
        }
    }
    pub(crate) fn io(operation: &'static str, source: std::io::Error) -> Self {
        Self::Io { operation, source }
    }
}

pub type Result<T> = std::result::Result<T, Error>;
