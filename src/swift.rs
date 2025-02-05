#[swift_bridge::bridge]
pub mod swift {
    extern "Rust" {
        fn rust_vec_from_ptr(ptr: *mut f32, len: usize) -> Vec<f32>;
    }

    extern "Swift" {
        type Model;

        #[swift_bridge(init)]
        fn compileModel(path: String) -> Model;
        fn bindInput(&self, shape: Vec<i32>, featureName: String, data: Vec<f32>);
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
        fn bytesFrom(&self, name: String) -> Vec<f32>;
    }
}

/// performs a memcpy
fn rust_vec_from_ptr(ptr: *mut f32, len: usize) -> Vec<f32> {
    unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
}
