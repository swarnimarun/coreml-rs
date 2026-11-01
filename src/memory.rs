//! Loading Core ML specifications and ML Program blobs without staging files.
use crate::{Error, Result};
use block2::RcBlock;
use objc2::rc::Retained;
use objc2_core_ml::{MLModel, MLModelAsset, MLModelConfiguration};
use objc2_foundation::{NSData, NSDictionary, NSError, NSString, NSURL};
use std::{future::Future, sync::Arc};

/// Immutable native buffers stay retained for the model's lifetime and until
/// any cancelled native completion handlers have finished.
pub(crate) struct MemoryBuffers {
    pub(crate) asset: Retained<MLModelAsset>,
    _specification: Retained<NSData>,
    _blobs: Vec<Retained<NSData>>,
    _mapping: Option<Retained<NSDictionary<NSURL, NSData>>>,
}

// SAFETY: Every NSData and NSDictionary is created immutable by this module;
// URLs are immutable and no mutable reference or native handle is exposed.
// The Core ML asset is only passed to documented asynchronous APIs. Callback
// threads hold this object solely to keep it alive and may release it there;
// they do not access its fields. Objective-C retain/release is thread-safe.
unsafe impl Send for MemoryBuffers {}
unsafe impl Sync for MemoryBuffers {}

pub(crate) fn asset(
    specification: Vec<u8>,
    blobs: Option<Vec<(String, Vec<u8>)>>,
) -> Result<Arc<MemoryBuffers>> {
    let specification = NSData::from_vec(specification);
    let mut native_blobs = Vec::new();
    let mut urls = Vec::new();
    let mapping = if let Some(blobs) = blobs {
        for (path, data) in blobs {
            if path.is_empty() || path.as_bytes().contains(&0) {
                return Err(Error::InvalidInput(
                    "blob paths must be nonempty and contain no NUL".into(),
                ));
            }
            urls.push(NSURL::fileURLWithPath(&NSString::from_str(&path)));
            native_blobs.push(NSData::from_vec(data));
        }
        let keys: Vec<_> = urls.iter().map(|url| &**url).collect();
        let values: Vec<_> = native_blobs.iter().map(|data| &**data).collect();
        let map = NSDictionary::from_slices(&keys, &values);
        if map.len() != keys.len() {
            return Err(Error::InvalidInput("duplicate blob URLs".into()));
        }
        Some(map)
    } else {
        None
    };
    // SAFETY: all data and URL objects remain retained by MemoryBuffers; the
    // caller checked availability of the chosen asset initializer.
    let asset = unsafe {
        match mapping.as_deref() {
            Some(map) => {
                MLModelAsset::modelAssetWithSpecificationData_blobMapping_error(&specification, map)
            }
            None => MLModelAsset::modelAssetWithSpecificationData_error(&specification),
        }
    }
    .map_err(|error| Error::native("create in-memory model asset", &error))?;
    Ok(Arc::new(MemoryBuffers {
        asset,
        _specification: specification,
        _blobs: native_blobs,
        _mapping: mapping,
    }))
}

/// A single owned model is transferred from Core ML's completion queue to the
/// Rust caller. The public Model remains thread-confined and predictions are
/// serialized; the wrapper never permits concurrent access to the native model.
struct TransferredModel(Retained<MLModel>);
// SAFETY: Core ML's asynchronous load completion returns a fully initialized
// object on an unspecified queue. Ownership is moved once through a mutex to
// the polling thread; no Rust code accesses it concurrently during transfer.
unsafe impl Send for TransferredModel {}

pub(crate) fn load(
    buffers: Arc<MemoryBuffers>,
    configuration: &MLModelConfiguration,
) -> impl Future<Output = Result<Retained<MLModel>>> {
    let (sender, future) = crate::callback::channel::<TransferredModel>();
    let callback_buffers = buffers.clone();
    let callback = RcBlock::new(move |model: *mut MLModel, error: *mut NSError| {
        let _keep_buffers_alive = &callback_buffers;
        if sender.is_cancelled() {
            return;
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            objc2::rc::autoreleasepool(|_| {
                // SAFETY: Core ML's completion arguments are live for the callback.
                // Retain a successful model before returning from that callback.
                unsafe {
                    if let Some(error) = error.as_ref() {
                        Err(Error::native("load in-memory model", error))
                    } else if let Some(model) = Retained::retain(model) {
                        Ok(TransferredModel(model))
                    } else {
                        Err(Error::ModelLoad(
                            "Core ML returned neither a model nor an error",
                        ))
                    }
                }
            })
        }))
        .unwrap_or(Err(Error::ModelLoad(
            "panic while receiving in-memory model",
        )));
        sender.complete(result);
    });
    // SAFETY: the asset, configuration and completion handler are live for the
    // call. Core ML retains the handler; its captured buffers keep inputs alive.
    unsafe {
        MLModel::loadModelAsset_configuration_completionHandler(
            &buffers.asset,
            configuration,
            &callback,
        )
    };
    async move { Ok(future.await?.0) }
}
