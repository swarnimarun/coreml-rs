#[swift_bridge::bridge]
pub mod swift {
    extern "Rust" {
        fn rust_vec_from_ptr_i32(ptr: *mut i32, len: usize) -> Vec<i32>;
        fn rust_vec_from_ptr_f32(ptr: *mut f32, len: usize) -> Vec<f32>;
        fn rust_vec_from_ptr_u16(ptr: *mut u16, len: usize) -> Vec<u16>;
        fn rust_vec_free_f32(ptr: *mut f32, len: usize);
        fn rust_vec_free_i32(ptr: *mut i32, len: usize);
        fn rust_vec_free_u16(ptr: *mut u16, len: usize);

    }

    extern "Swift" {
        type Model;

        #[swift_bridge(init)]
        fn compileModel(path: String) -> Model;
        fn bindInputF32(&self, shape: Vec<i32>, featureName: String, data: *mut f32, len: usize);
        fn bindInputI32(&self, shape: Vec<i32>, featureName: String, data: Vec<i32>);
        fn bindInputF16(&self, shape: Vec<i32>, featureName: String, data: Vec<u16>);
        #[swift_bridge(swift_name = "load")]
        fn modelLoad(&mut self);
        #[swift_bridge(swift_name = "description")]
        fn modelDescription(&self) -> ModelDescription;
        #[swift_bridge(swift_name = "predict")]
        fn modelRun(&self) -> ModelOutput;
    }

    extern "Swift" {
        type ModelDescription;

        fn inputs(&self) -> Vec<String>;
        fn outputs(&self) -> Vec<String>;
    }

    extern "Swift" {
        type ModelOutput;

        fn outputDescription(&self) -> Vec<String>;
        fn outputF32(&self, name: String) -> Vec<f32>;
        fn outputF16(&self, name: String) -> Vec<u16>;
        fn outputI32(&self, name: String) -> Vec<i32>;
    }
}

/// performs a memcpy
fn rust_vec_from_ptr_f32(ptr: *mut f32, len: usize) -> Vec<f32> {
    unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
}
/// performs a memcpy
fn rust_vec_from_ptr_u16(ptr: *mut u16, len: usize) -> Vec<u16> {
    unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
}
/// performs a memcpy
fn rust_vec_from_ptr_i32(ptr: *mut i32, len: usize) -> Vec<i32> {
    unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
}

fn rust_vec_free_f32(ptr: *mut f32, len: usize) {
    unsafe {
        _ = Vec::from_raw_parts(ptr, len, len);
    }
}
fn rust_vec_free_u16(ptr: *mut u16, len: usize) {
    unsafe {
        _ = Vec::from_raw_parts(ptr, len, len);
    }
}
fn rust_vec_free_i32(ptr: *mut i32, len: usize) {
    unsafe {
        _ = Vec::from_raw_parts(ptr, len, len);
    }
}
