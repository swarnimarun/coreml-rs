//! Explicit Core Video to Metal texture sharing for pixel-buffer-backed arrays.
//!
//! This does not turn arbitrary Core ML arrays into `MTLBuffer`s. Metal and Core
//! ML work must be synchronized by the caller before either accesses the buffer.
use crate::{Error, MultiArray, Result};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_core_foundation::CFRetained;
use objc2_core_video::{
    CVMetalTexture, CVMetalTextureCache, CVMetalTextureGetTexture, CVPixelBufferGetHeight,
    CVPixelBufferGetWidth,
};
use objc2_metal::{MTLDevice, MTLPixelFormat, MTLTexture};
use std::ptr::{self, NonNull};

/// Reusable Metal texture cache for Core Video-backed storage.
pub struct MetalTextureCache(CFRetained<CVMetalTextureCache>);
impl MetalTextureCache {
    pub fn new(device: &ProtocolObject<dyn MTLDevice>) -> Result<Self> {
        let mut ptr = ptr::null_mut();
        // SAFETY: ptr is a valid out parameter, and device remains live during creation.
        let code = unsafe {
            CVMetalTextureCache::create(None, None, device, None, NonNull::from(&mut ptr))
        };
        if code != 0 {
            return Err(Error::PlatformStatus {
                operation: "create Metal texture cache",
                code,
            });
        }
        let ptr =
            NonNull::new(ptr).ok_or(Error::Unsupported("Core Video returned no texture cache"))?;
        // SAFETY: a successful Create call transfers ownership to the caller.
        Ok(Self(unsafe { CFRetained::from_raw(ptr) }))
    }
    /// Create a Metal texture sharing the multiarray's pixel buffer.
    ///
    /// # Safety
    /// The caller must choose a compatible pixel format and synchronize GPU,
    /// CPU, and Core ML access. Retain the returned guard until GPU work is
    /// complete. Core Video may reject buffers without compatible storage.
    pub unsafe fn texture(
        &self,
        array: &MultiArray,
        format: MTLPixelFormat,
    ) -> Result<MetalTexture> {
        let pixel = array
            .pixel_buffer()
            .ok_or(Error::Unsupported("multiarray is not pixel-buffer backed"))?;
        let mut ptr = ptr::null_mut();
        let code = CVMetalTextureCache::create_texture_from_image(
            None,
            &self.0,
            &pixel,
            None,
            format,
            CVPixelBufferGetWidth(&pixel),
            CVPixelBufferGetHeight(&pixel),
            0,
            NonNull::from(&mut ptr),
        );
        if code != 0 {
            return Err(Error::PlatformStatus {
                operation: "create Metal texture",
                code,
            });
        }
        let ptr =
            NonNull::new(ptr).ok_or(Error::Unsupported("Core Video returned no Metal texture"))?;
        // SAFETY: a successful Create call transfers ownership to the caller.
        let cv_texture = CFRetained::from_raw(ptr);
        let texture = CVMetalTextureGetTexture(&cv_texture)
            .ok_or(Error::Unsupported("Core Video returned no Metal texture"))?;
        Ok(MetalTexture {
            texture,
            _cv_texture: cv_texture,
            _array: array.clone(),
        })
    }
}

/// Keeps the Core Video texture and source array alive for the duration of GPU work.
pub struct MetalTexture {
    texture: Retained<ProtocolObject<dyn MTLTexture>>,
    _cv_texture: CFRetained<CVMetalTexture>,
    _array: MultiArray,
}
impl MetalTexture {
    /// Access the Metal texture. Keep this guard alive until GPU work completes.
    pub fn as_native(&self) -> &ProtocolObject<dyn MTLTexture> {
        &self.texture
    }
}
