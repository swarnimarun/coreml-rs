pub mod mlarray;
// pub mod mlmodel;
pub mod mlmodel;
mod swift;
pub use swift::swift as ffi;

pub use swift::swift::load_save;
