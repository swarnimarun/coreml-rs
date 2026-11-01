//! Map an IOSurface-backed Core ML multiarray to a Metal texture.
//! Run: cargo run --release --example metal
use coreml_rs::{metal::MetalTextureCache, MultiArray};
use objc2_core_foundation::{CFDictionary, CFRetained, CFString};
use objc2_core_video::{
    kCVPixelBufferIOSurfacePropertiesKey, kCVPixelFormatType_OneComponent16Half, CVPixelBuffer,
    CVPixelBufferCreate,
};
use objc2_metal::{MTLCreateSystemDefaultDevice, MTLPixelFormat, MTLTexture};
use std::{error::Error, ptr::NonNull};

fn main() -> Result<(), Box<dyn Error>> {
    let device = MTLCreateSystemDefaultDevice().ok_or("no Metal device")?;
    let empty = CFDictionary::<CFString, CFString>::empty();
    // SAFETY: this constant is provided by Core Video on supported macOS versions.
    let attrs = CFDictionary::from_slices(
        &[unsafe { kCVPixelBufferIOSurfacePropertiesKey }],
        &[&*empty],
    );
    // SAFETY: the typed dictionary is a CFDictionary with the documented Core Video key and value.
    let attrs = unsafe { &*((&*attrs as *const _) as *const CFDictionary) };
    let mut raw: *mut CVPixelBuffer = std::ptr::null_mut();
    // SAFETY: raw is a valid out parameter; attrs remain alive during the call.
    let status = unsafe {
        CVPixelBufferCreate(
            None,
            4,
            2,
            kCVPixelFormatType_OneComponent16Half,
            Some(attrs),
            NonNull::from(&mut raw),
        )
    };
    if status != 0 {
        return Err(format!("CVPixelBufferCreate failed: {status}").into());
    }
    // SAFETY: a successful Create transfers ownership of the buffer to us.
    let pixel = unsafe { CFRetained::from_raw(NonNull::new(raw).ok_or("no pixel buffer")?) };
    // SAFETY: no other consumer accesses pixel while the array and texture are used.
    let array = unsafe { MultiArray::from_pixel_buffer(&pixel, &[2, 4]) }?;
    let cache = MetalTextureCache::new(&device)?;
    // SAFETY: R16Float matches the half-float buffer. Any GPU commands must
    // complete before Core ML or the CPU uses the array, and this guard must
    // stay alive until those commands have completed.
    let texture = unsafe { cache.texture(&array, MTLPixelFormat::R16Float) }?;
    println!(
        "shared Metal texture: {} x {}",
        texture.as_native().width(),
        texture.as_native().height()
    );
    Ok(())
}
