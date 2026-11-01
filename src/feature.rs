use crate::{DataType, Error, Result, Tensor, TensorElement};
use block2::StackBlock;
use objc2::{
    rc::Retained,
    runtime::{AnyObject, ProtocolObject},
    AnyThread,
};
use objc2_core_ml::{
    MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureType, MLFeatureValue, MLMultiArray,
    MLMultiArrayDataType,
};
use objc2_foundation::{NSArray, NSDictionary, NSNumber, NSString};
use std::{cell::RefCell, collections::BTreeMap, ffi::c_void, ptr::NonNull};

fn native_dtype(dtype: DataType) -> MLMultiArrayDataType {
    match dtype {
        DataType::Float16 => MLMultiArrayDataType::Float16,
        DataType::Float32 => MLMultiArrayDataType::Float32,
        DataType::Float64 => MLMultiArrayDataType::Double,
        DataType::Int32 => MLMultiArrayDataType::Int32,
        DataType::Int8 => MLMultiArrayDataType::Int8,
    }
}

/// An owned Core ML multiarray. Converting from or to a Rust tensor copies data;
/// keeping this value between predictions avoids copies at the crate boundary.
#[derive(Clone, Debug)]
pub struct MultiArray(pub(crate) Retained<MLMultiArray>);

impl MultiArray {
    /// Copy a contiguous Rust tensor into Core ML-owned storage.
    pub fn from_tensor<T: TensorElement>(tensor: &Tensor<T>) -> Result<Self> {
        if crate::model::macos_version() < (12, 3, 0) {
            return Err(Error::Unsupported(
                "scoped multiarray access requires macOS 12.3",
            ));
        }
        if T::DATA_TYPE == DataType::Int8 && crate::model::macos_version() < (26, 0, 0) {
            return Err(Error::Unsupported("Int8 multiarrays require macOS 26"));
        }
        if tensor.shape().is_empty()
            || tensor.shape().contains(&0)
            || tensor.shape().iter().any(|&n| n > isize::MAX as usize)
        {
            return Err(Error::InvalidInput(
                "Core ML multiarrays require nonzero dimensions representable by NSInteger".into(),
            ));
        }
        let shape: Vec<_> = tensor
            .shape()
            .iter()
            .map(|&n| NSNumber::numberWithUnsignedInteger(n))
            .collect();
        let shape = NSArray::from_retained_slice(&shape);
        // SAFETY: the shape and dtype have been validated and the returned object owns its storage.
        let array = unsafe {
            MLMultiArray::initWithShape_dataType_error(
                MLMultiArray::alloc(),
                &shape,
                native_dtype(T::DATA_TYPE),
            )
        }
        .map_err(|e| Error::native("allocate multiarray", &e))?;
        let mut result = Self(array);
        result.update(tensor)?;
        Ok(result)
    }

    /// Copy new tensor contents into existing native storage without reallocating.
    /// All clones refer to the same multiarray and will observe the update.
    pub fn update<T: TensorElement>(&mut self, tensor: &Tensor<T>) -> Result<()> {
        if crate::model::macos_version() < (12, 3, 0) {
            return Err(Error::Unsupported(
                "scoped multiarray access requires macOS 12.3",
            ));
        }
        if self.shape() != tensor.shape()
            || unsafe { self.0.dataType() } != native_dtype(T::DATA_TYPE)
        {
            return Err(Error::InvalidInput(
                "tensor shape or dtype does not match the multiarray".into(),
            ));
        }
        let status = RefCell::new(None);
        let callback = StackBlock::new(
            |ptr: NonNull<c_void>, bytes: isize, strides: NonNull<NSArray<NSNumber>>| {
                // SAFETY: Core ML supplies a valid strides array for this callback.
                let strides = unsafe { strides.as_ref() };
                let strides: Vec<usize> =
                    strides.iter().map(|n| n.unsignedIntegerValue()).collect();
                *status.borrow_mut() = Some(copy_into_strided(ptr, bytes, &strides, tensor));
            },
        );
        // SAFETY: the callback does not escape and writes only checked offsets.
        unsafe { self.0.getMutableBytesWithHandler(&callback) };
        status.into_inner().ok_or(Error::Unsupported(
            "Core ML did not provide mutable array storage",
        ))?
    }

    /// Return the shape reported by Core ML.
    pub fn shape(&self) -> Vec<usize> {
        // SAFETY: the retained array remains alive while its immutable metadata is read.
        unsafe {
            self.0
                .shape()
                .iter()
                .map(|n| n.unsignedIntegerValue())
                .collect()
        }
    }

    /// Whether Core ML returned this exact multiarray object.
    pub fn is_same_object(&self, other: &Self) -> bool {
        std::ptr::eq(&*self.0, &*other.0)
    }

    /// Return Core ML strides, measured in elements.
    pub fn strides(&self) -> Vec<usize> {
        // SAFETY: the retained array remains alive while its immutable metadata is read.
        unsafe {
            self.0
                .strides()
                .iter()
                .map(|n| n.unsignedIntegerValue())
                .collect()
        }
    }

    /// Copy a multiarray into a contiguous Rust tensor, preserving logical order.
    pub fn to_tensor<T: TensorElement>(&self) -> Result<Tensor<T>> {
        if crate::model::macos_version() < (12, 3, 0) {
            return Err(Error::Unsupported(
                "scoped multiarray access requires macOS 12.3",
            ));
        }
        // SAFETY: querying the dtype is safe for a retained Core ML object.
        if unsafe { self.0.dataType() } != native_dtype(T::DATA_TYPE) {
            return Err(Error::InvalidInput(format!(
                "multiarray is not {:?}",
                T::DATA_TYPE
            )));
        }
        let shape = self.shape();
        if shape.contains(&0) {
            return Ok(Tensor::from_vec(shape, Vec::new())?);
        }
        let result = RefCell::new(None);
        let callback = StackBlock::new(|ptr: NonNull<c_void>, bytes: isize| {
            // The read callback pins the buffer. Query strides inside the callback.
            let strides = self.strides();
            *result.borrow_mut() = Some(copy_from_strided::<T>(ptr, bytes, &shape, &strides));
        });
        // SAFETY: Core ML invokes the callback synchronously and it reads only checked offsets.
        unsafe { self.0.getBytesWithHandler(&callback) };
        let data = result
            .into_inner()
            .ok_or(Error::Unsupported("Core ML did not provide array storage"))??;
        Ok(Tensor::from_vec(shape, data)?)
    }

    /// Borrow contiguous native storage for the duration of a callback without copying.
    ///
    /// # Safety
    /// No other native, CPU, GPU, or Core ML operation may mutate this storage
    /// during the callback, including through another retained alias. The callback
    /// must not start work that outlives the borrow.
    pub unsafe fn with_contiguous<T: TensorElement, R>(
        &self,
        f: impl FnOnce(&[T]) -> R,
    ) -> Result<R> {
        if crate::model::macos_version() < (12, 3, 0) {
            return Err(Error::Unsupported(
                "scoped multiarray access requires macOS 12.3",
            ));
        }
        if self.0.dataType() != native_dtype(T::DATA_TYPE) {
            return Err(Error::InvalidInput(format!(
                "multiarray is not {:?}",
                T::DATA_TYPE
            )));
        }
        let shape = self.shape();
        if shape.contains(&0) {
            return Ok(f(&[]));
        }
        let f = RefCell::new(Some(f));
        let result = RefCell::new(None);
        let callback = StackBlock::new(|ptr: NonNull<c_void>, bytes: isize| {
            let strides = self.strides();
            let value = layout(&shape, &strides, bytes, std::mem::size_of::<T>()).and_then(
                |(count, contiguous)| {
                    if !contiguous
                        || !(ptr.as_ptr() as usize).is_multiple_of(std::mem::align_of::<T>())
                    {
                        return Err(Error::InvalidInput(
                            "multiarray is not contiguous and aligned".into(),
                        ));
                    }
                    // SAFETY: Core ML pins the storage for this callback; layout and
                    // alignment were checked, and the caller guarantees no mutation.
                    let slice = std::slice::from_raw_parts(ptr.as_ptr().cast::<T>(), count);
                    let callback = f.borrow_mut().take().ok_or(Error::Unsupported(
                        "Core ML called its buffer handler more than once",
                    ))?;
                    Ok(std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                        || callback(slice),
                    )))
                },
            );
            *result.borrow_mut() = Some(value);
        });
        self.0.getBytesWithHandler(&callback);
        match result
            .into_inner()
            .ok_or(Error::Unsupported("Core ML did not provide array storage"))??
        {
            Ok(value) => Ok(value),
            Err(panic) => std::panic::resume_unwind(panic),
        }
    }

    /// Mutate contiguous native storage within a synchronous callback.
    ///
    /// # Safety
    /// No other native, CPU, GPU, or Core ML operation may access this storage
    /// during the callback, including through another retained alias. The callback
    /// must not start work that outlives the borrow.
    pub unsafe fn with_contiguous_mut<T: TensorElement, R>(
        &mut self,
        f: impl FnOnce(&mut [T]) -> R,
    ) -> Result<R> {
        if crate::model::macos_version() < (12, 3, 0) {
            return Err(Error::Unsupported(
                "scoped multiarray access requires macOS 12.3",
            ));
        }
        if self.0.dataType() != native_dtype(T::DATA_TYPE) {
            return Err(Error::InvalidInput(format!(
                "multiarray is not {:?}",
                T::DATA_TYPE
            )));
        }
        let shape = self.shape();
        if shape.contains(&0) {
            return Ok(f(&mut []));
        }
        let f = RefCell::new(Some(f));
        let result = RefCell::new(None);
        let callback = StackBlock::new(
            |ptr: NonNull<c_void>, bytes: isize, strides: NonNull<NSArray<NSNumber>>| {
                let strides: Vec<usize> = unsafe { strides.as_ref() }
                    .iter()
                    .map(|n| n.unsignedIntegerValue())
                    .collect();
                let value = layout(&shape, &strides, bytes, std::mem::size_of::<T>()).and_then(
                    |(count, contiguous)| {
                        if !contiguous
                            || !(ptr.as_ptr() as usize).is_multiple_of(std::mem::align_of::<T>())
                        {
                            return Err(Error::InvalidInput(
                                "multiarray is not contiguous and aligned".into(),
                            ));
                        }
                        // SAFETY: Core ML pins the buffer and the caller guarantees
                        // exclusive access; layout and alignment are checked above.
                        let slice = std::slice::from_raw_parts_mut(ptr.as_ptr().cast::<T>(), count);
                        let callback = f.borrow_mut().take().ok_or(Error::Unsupported(
                            "Core ML called its buffer handler more than once",
                        ))?;
                        Ok(std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                            || callback(slice),
                        )))
                    },
                );
                *result.borrow_mut() = Some(value);
            },
        );
        self.0.getMutableBytesWithHandler(&callback);
        match result
            .into_inner()
            .ok_or(Error::Unsupported("Core ML did not provide array storage"))??
        {
            Ok(value) => Ok(value),
            Err(panic) => std::panic::resume_unwind(panic),
        }
    }

    /// Retain the pixel buffer when this array is backed by one. This may be
    /// shared with Metal through a Core Video texture cache.
    ///
    /// # Safety
    /// Synchronize CPU, Core ML and GPU access to the shared storage.
    pub unsafe fn pixel_buffer(&self) -> Option<Retained<objc2_core_video::CVPixelBuffer>> {
        self.0.pixelBuffer()
    }

    /// Wrap a pixel buffer without copying. Core ML accepts half-float buffers
    /// on macOS 12 and one-component Int8 buffers on macOS 26 or later.
    ///
    /// # Safety
    /// The caller must synchronize access and avoid mutating the backing buffer
    /// while Core ML or another consumer is using it.
    pub unsafe fn from_pixel_buffer(
        buffer: &objc2_core_video::CVPixelBuffer,
        shape: &[usize],
    ) -> Result<Self> {
        use objc2_core_video::{
            kCVPixelFormatType_OneComponent16Half, kCVPixelFormatType_OneComponent8,
            CVPixelBufferGetHeight, CVPixelBufferGetPixelFormatType, CVPixelBufferGetWidth,
        };
        let format = CVPixelBufferGetPixelFormatType(buffer);
        if format != kCVPixelFormatType_OneComponent16Half
            && !(format == kCVPixelFormatType_OneComponent8
                && crate::model::macos_version() >= (26, 0, 0))
        {
            return Err(Error::Unsupported(
                "unsupported pixel-buffer format for multiarray",
            ));
        }
        let (width, outer) = shape
            .split_last()
            .ok_or_else(|| Error::InvalidInput("pixel-buffer shape must not be empty".into()))?;
        let height = outer.iter().try_fold(1usize, |n, &d| n.checked_mul(d));
        if *width != CVPixelBufferGetWidth(buffer)
            || height != Some(CVPixelBufferGetHeight(buffer))
            || shape.contains(&0)
            || shape.iter().any(|&n| n > isize::MAX as usize)
        {
            return Err(Error::InvalidInput(
                "pixel-buffer dimensions do not match the multiarray shape".into(),
            ));
        }
        let dims: Vec<_> = shape
            .iter()
            .map(|&n| NSNumber::numberWithUnsignedInteger(n))
            .collect();
        let dims = NSArray::from_retained_slice(&dims);
        // SAFETY: shape and format satisfy Core ML's documented requirements;
        // synchronization is the caller's responsibility.
        Ok(Self(MLMultiArray::initWithPixelBuffer_shape(
            MLMultiArray::alloc(),
            buffer,
            &dims,
        )))
    }

    /// Access the retained native object for APIs not wrapped by this crate.
    ///
    /// # Safety
    /// The caller must uphold Core ML's mutation, synchronization, and lifetime rules.
    pub unsafe fn as_native(&self) -> &MLMultiArray {
        &self.0
    }

    /// Wrap a retained native multiarray.
    ///
    /// # Safety
    /// The caller must prevent concurrent mutation while safe accessors run.
    pub unsafe fn from_native(array: Retained<MLMultiArray>) -> Self {
        Self(array)
    }
}

// Validate the maximum reachable element once, then traverse without allocating
// an offset vector proportional to the tensor size.
fn layout(
    shape: &[usize],
    strides: &[usize],
    bytes: isize,
    element_size: usize,
) -> Result<(usize, bool)> {
    if shape.len() != strides.len() || bytes < 0 {
        return Err(Error::InvalidInput("invalid Core ML buffer layout".into()));
    }
    let count = shape
        .iter()
        .try_fold(1usize, |a, &b| a.checked_mul(b))
        .ok_or_else(|| Error::InvalidInput("array size overflow".into()))?;
    if count == 0 {
        return Ok((0, true));
    }
    if count
        .checked_mul(element_size)
        .filter(|&n| n <= isize::MAX as usize)
        .is_none()
    {
        return Err(Error::InvalidInput("array byte size overflow".into()));
    }
    let mut max_offset = 0usize;
    let mut expected_stride = 1usize;
    let mut contiguous = true;
    for (&dim, &stride) in shape.iter().zip(strides).rev() {
        max_offset = max_offset
            .checked_add(
                (dim - 1)
                    .checked_mul(stride)
                    .ok_or_else(|| Error::InvalidInput("array stride overflow".into()))?,
            )
            .ok_or_else(|| Error::InvalidInput("array offset overflow".into()))?;
        if dim > 1 && stride != expected_stride {
            contiguous = false;
        }
        expected_stride = expected_stride
            .checked_mul(dim)
            .ok_or_else(|| Error::InvalidInput("array size overflow".into()))?;
    }
    if max_offset >= bytes as usize / element_size
        || max_offset > isize::MAX as usize / element_size
    {
        return Err(Error::InvalidInput(
            "Core ML buffer is smaller than its shape and strides".into(),
        ));
    }
    Ok((count, contiguous))
}
fn offset_for(mut index: usize, shape: &[usize], strides: &[usize]) -> usize {
    let mut offset = 0;
    for (&dim, &stride) in shape.iter().zip(strides).rev() {
        offset += (index % dim) * stride;
        index /= dim;
    }
    offset
}
fn copy_into_strided<T: TensorElement>(
    ptr: NonNull<c_void>,
    bytes: isize,
    strides: &[usize],
    tensor: &Tensor<T>,
) -> Result<()> {
    let (count, contiguous) = layout(tensor.shape(), strides, bytes, std::mem::size_of::<T>())?;
    if contiguous {
        // SAFETY: Core ML owns a separate buffer with enough bytes, checked by layout.
        unsafe {
            std::ptr::copy_nonoverlapping(
                tensor.as_slice().as_ptr().cast::<u8>(),
                ptr.as_ptr().cast::<u8>(),
                count * std::mem::size_of::<T>(),
            )
        };
    } else {
        for (index, value) in tensor.as_slice().iter().enumerate() {
            let offset = offset_for(index, tensor.shape(), strides);
            // SAFETY: layout checked the greatest reachable offset.
            unsafe { ptr.as_ptr().cast::<T>().add(offset).write_unaligned(*value) };
        }
    }
    Ok(())
}
fn copy_from_strided<T: TensorElement>(
    ptr: NonNull<c_void>,
    bytes: isize,
    shape: &[usize],
    strides: &[usize],
) -> Result<Vec<T>> {
    let (count, contiguous) = layout(shape, strides, bytes, std::mem::size_of::<T>())?;
    let mut out = Vec::<T>::with_capacity(count);
    if contiguous {
        // SAFETY: Core ML reports at least count elements; byte copying permits unaligned source.
        // Every bit pattern is valid for the sealed TensorElement types.
        unsafe {
            std::ptr::copy_nonoverlapping(
                ptr.as_ptr().cast::<u8>(),
                out.as_mut_ptr().cast::<u8>(),
                count * std::mem::size_of::<T>(),
            );
            out.set_len(count);
        }
    } else {
        for index in 0..count {
            let offset = offset_for(index, shape, strides);
            // SAFETY: layout checked the greatest reachable offset.
            out.push(unsafe { ptr.as_ptr().cast::<T>().add(offset).read_unaligned() });
        }
    }
    Ok(out)
}

/// Keys supported by Core ML dictionary features.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum DictionaryKey {
    String(String),
    Integer(i64),
}

/// A retained Core ML feature value. Other native feature types can be passed
/// through without being converted to a Rust representation.
#[derive(Clone, Debug)]
pub struct FeatureValue(pub(crate) Retained<MLFeatureValue>);
impl FeatureValue {
    pub fn int64(value: i64) -> Self {
        Self(unsafe { MLFeatureValue::featureValueWithInt64(value) })
    }
    pub fn double(value: f64) -> Self {
        Self(unsafe { MLFeatureValue::featureValueWithDouble(value) })
    }
    pub fn string(value: &str) -> Self {
        Self(unsafe { MLFeatureValue::featureValueWithString(&NSString::from_str(value)) })
    }
    pub fn multi_array(value: &MultiArray) -> Self {
        Self(unsafe { MLFeatureValue::featureValueWithMultiArray(&value.0) })
    }
    /// Copy tightly packed RGBA8 pixels into an owned Core Video image feature.
    pub fn image_from_rgba8(width: usize, height: usize, pixels: &[u8]) -> Result<Self> {
        use objc2_core_foundation::CFRetained;
        use objc2_core_video::{
            kCVPixelFormatType_32BGRA, CVPixelBuffer, CVPixelBufferCreate,
            CVPixelBufferGetBaseAddress, CVPixelBufferGetBytesPerRow, CVPixelBufferGetDataSize,
            CVPixelBufferLockBaseAddress, CVPixelBufferLockFlags, CVPixelBufferUnlockBaseAddress,
        };
        let row = width
            .checked_mul(4)
            .ok_or_else(|| Error::InvalidInput("image size overflow".into()))?;
        let size = row
            .checked_mul(height)
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or_else(|| Error::InvalidInput("image size overflow".into()))?;
        if width == 0 || height == 0 || pixels.len() != size {
            return Err(Error::InvalidInput(
                "RGBA image dimensions do not match its byte length".into(),
            ));
        }
        let mut raw: *mut CVPixelBuffer = std::ptr::null_mut();
        // SAFETY: raw is a valid out parameter and no custom buffer attributes are supplied.
        let status = unsafe {
            CVPixelBufferCreate(
                None,
                width,
                height,
                kCVPixelFormatType_32BGRA,
                None,
                NonNull::from(&mut raw),
            )
        };
        if status != 0 {
            return Err(Error::PlatformStatus {
                operation: "create image buffer",
                code: status,
            });
        }
        let raw =
            NonNull::new(raw).ok_or(Error::Unsupported("Core Video returned no image buffer"))?;
        // SAFETY: a successful Create call transfers ownership to us.
        let buffer = unsafe { CFRetained::from_raw(raw) };
        let flags = CVPixelBufferLockFlags(0);
        let status = unsafe { CVPixelBufferLockBaseAddress(&buffer, flags) };
        if status != 0 {
            return Err(Error::PlatformStatus {
                operation: "lock image buffer",
                code: status,
            });
        }
        let stride = CVPixelBufferGetBytesPerRow(&buffer);
        let available = CVPixelBufferGetDataSize(&buffer);
        let dest = CVPixelBufferGetBaseAddress(&buffer).cast::<u8>();
        let valid = !dest.is_null()
            && stride >= row
            && stride
                .checked_mul(height)
                .is_some_and(|n| n <= available && n <= isize::MAX as usize);
        if valid {
            for y in 0..height {
                for x in 0..width {
                    let src = y * row + x * 4;
                    let dst = y * stride + x * 4;
                    // SAFETY: both buffers were checked above and do not overlap.
                    unsafe {
                        *dest.add(dst) = pixels[src + 2];
                        *dest.add(dst + 1) = pixels[src + 1];
                        *dest.add(dst + 2) = pixels[src];
                        *dest.add(dst + 3) = pixels[src + 3];
                    }
                }
            }
        }
        let unlock = unsafe { CVPixelBufferUnlockBaseAddress(&buffer, flags) };
        if unlock != 0 {
            return Err(Error::PlatformStatus {
                operation: "unlock image buffer",
                code: unlock,
            });
        }
        if !valid {
            return Err(Error::InvalidInput(
                "Core Video returned an invalid image buffer layout".into(),
            ));
        }
        // SAFETY: this newly created buffer is private to the returned feature.
        Ok(unsafe { Self::image(&buffer) })
    }

    /// Convert a Core ML dictionary feature into typed keys and numeric values.
    pub fn as_dictionary(&self) -> Result<Option<Vec<(DictionaryKey, f64)>>> {
        // SAFETY: access the dictionary property only when the feature has that type.
        if unsafe { self.0.r#type() } != MLFeatureType::Dictionary {
            return Ok(None);
        }
        let dictionary = unsafe { self.0.dictionaryValue() };
        let (keys, values) = dictionary.to_vecs();
        keys.iter()
            .zip(values.iter())
            .map(|(key, value)| {
                let key = if let Some(s) = key.downcast_ref::<NSString>() {
                    DictionaryKey::String(s.to_string())
                } else if let Some(n) = key.downcast_ref::<NSNumber>() {
                    DictionaryKey::Integer(n.longLongValue())
                } else {
                    return Err(Error::Unsupported(
                        "dictionary key is neither string nor integer",
                    ));
                };
                Ok((key, value.doubleValue()))
            })
            .collect::<Result<Vec<_>>>()
            .map(Some)
    }

    /// Create an image feature from a Core Video pixel buffer.
    ///
    /// # Safety
    /// The caller must synchronize access to the buffer across consumers.
    pub unsafe fn image(buffer: &objc2_core_video::CVPixelBuffer) -> Self {
        Self(MLFeatureValue::featureValueWithPixelBuffer(buffer))
    }
    /// Retain an image feature's pixel buffer, if present.
    ///
    /// # Safety
    /// The caller must synchronize access to the buffer across consumers.
    pub unsafe fn image_buffer(&self) -> Option<Retained<objc2_core_video::CVPixelBuffer>> {
        if self.0.r#type() == MLFeatureType::Image {
            self.0.imageBufferValue()
        } else {
            None
        }
    }
    pub fn as_int64(&self) -> Option<i64> {
        unsafe { (self.0.r#type() == MLFeatureType::Int64).then(|| self.0.int64Value()) }
    }
    pub fn as_double(&self) -> Option<f64> {
        unsafe { (self.0.r#type() == MLFeatureType::Double).then(|| self.0.doubleValue()) }
    }
    pub fn as_string(&self) -> Option<String> {
        unsafe {
            (self.0.r#type() == MLFeatureType::String).then(|| self.0.stringValue().to_string())
        }
    }
    pub fn as_multi_array(&self) -> Option<MultiArray> {
        unsafe {
            if self.0.r#type() == MLFeatureType::MultiArray {
                self.0.multiArrayValue().map(MultiArray)
            } else {
                None
            }
        }
    }
    /// # Safety
    /// The caller must follow Core ML's native object and synchronization rules.
    pub unsafe fn as_native(&self) -> &MLFeatureValue {
        &self.0
    }
    /// # Safety
    /// The caller must ensure the native value is valid and not concurrently mutated.
    pub unsafe fn from_native(value: Retained<MLFeatureValue>) -> Self {
        Self(value)
    }
}

/// Per-request named features. Clones retain native objects without copying buffers.
#[derive(Clone, Debug, Default)]
pub struct Inputs(BTreeMap<String, FeatureValue>);
impl Inputs {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with(mut self, name: impl Into<String>, value: FeatureValue) -> Self {
        self.0.insert(name.into(), value);
        self
    }
    pub fn insert(&mut self, name: impl Into<String>, value: FeatureValue) -> Option<FeatureValue> {
        self.0.insert(name.into(), value)
    }
    /// Prepare and retain the native feature provider for repeated predictions.
    pub fn prepare(&self) -> Result<PreparedInputs> {
        self.provider().map(PreparedInputs)
    }
    pub(crate) fn provider(&self) -> Result<Retained<MLDictionaryFeatureProvider>> {
        let keys: Vec<_> = self.0.keys().map(|s| NSString::from_str(s)).collect();
        let values: Vec<&AnyObject> = self
            .0
            .values()
            .map(|v| AsRef::<AnyObject>::as_ref(&*v.0))
            .collect();
        let key_refs: Vec<_> = keys.iter().map(|k| &**k).collect();
        let dict = NSDictionary::from_slices(&key_refs, &values);
        // SAFETY: every dictionary value is a retained MLFeatureValue, accepted by this initializer.
        unsafe {
            MLDictionaryFeatureProvider::initWithDictionary_error(
                MLDictionaryFeatureProvider::alloc(),
                &dict,
            )
        }
        .map_err(|e| Error::native("create feature provider", &e))
    }
}

/// A native feature provider that can be reused for repeated predictions.
/// Updating a multiarray used by this provider updates what subsequent calls see.
#[derive(Clone, Debug)]
pub struct PreparedInputs(pub(crate) Retained<MLDictionaryFeatureProvider>);

/// Prediction outputs, retaining their native storage without an eager copy.
#[derive(Clone, Debug)]
pub struct Outputs(pub(crate) BTreeMap<String, FeatureValue>);
impl Outputs {
    pub fn get(&self, name: &str) -> Option<&FeatureValue> {
        self.0.get(name)
    }
    pub fn iter(&self) -> impl Iterator<Item = (&str, &FeatureValue)> {
        self.0.iter().map(|(k, v)| (k.as_str(), v))
    }
    pub fn multi_array(&self, name: &str) -> Result<MultiArray> {
        let value = self
            .get(name)
            .ok_or_else(|| Error::MissingFeature(name.into()))?;
        value.as_multi_array().ok_or_else(|| Error::FeatureType {
            name: name.into(),
            expected: "multiarray",
            actual: format!("{:?}", unsafe { value.0.r#type() }),
        })
    }
    pub(crate) fn from_provider(provider: &ProtocolObject<dyn MLFeatureProvider>) -> Result<Self> {
        let mut map = BTreeMap::new();
        // SAFETY: protocol methods operate on a live provider returned by Core ML.
        unsafe {
            for name in provider.featureNames().iter() {
                let value = provider
                    .featureValueForName(&name)
                    .ok_or_else(|| Error::MissingFeature(name.to_string()))?;
                map.insert(name.to_string(), FeatureValue(value));
            }
        }
        Ok(Self(map))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checks_strided_layout_and_copies_logical_order() {
        let mut backing = [1i32, 2, 99, 3, 4, 99];
        let ptr = NonNull::new(backing.as_mut_ptr().cast()).unwrap();
        let copied = copy_from_strided::<i32>(ptr, 24, &[2, 2], &[3, 1]).unwrap();
        assert_eq!(copied, [1, 2, 3, 4]);
        let input = Tensor::from_vec(vec![2, 2], vec![5, 6, 7, 8]).unwrap();
        copy_into_strided(ptr, 24, &[3, 1], &input).unwrap();
        assert_eq!(backing, [5, 6, 99, 7, 8, 99]);
        assert!(copy_from_strided::<i32>(ptr, 12, &[2, 2], &[3, 1]).is_err());
        assert!(layout(&[usize::MAX, 2], &[2, 1], isize::MAX, 4).is_err());
    }
}
