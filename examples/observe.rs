//! Inspect process memory and model metrics, then try idle cache eviction.
//! Run: cargo run --release --example observe -- model.mlmodel
use coreml_rs::{MemoryPressureMonitor, Model, ModelCache, ModelOptions, ProcessMemory};
use std::{error::Error, time::Duration};

fn main() -> Result<(), Box<dyn Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: observe <model.mlmodel|model.mlpackage>")?;
    let pressure = MemoryPressureMonitor::new();
    let compiled = Model::compile(path)?;
    let model = compiled.load(ModelOptions::default())?;
    let observer = model.metrics_handle();
    let mut cache = ModelCache::new();
    cache.insert("model", model);

    println!("request metrics: {:?}", observer.snapshot());
    println!("process memory: {:?}", ProcessMemory::current()?);
    println!(
        "latest system memory-pressure event: {:?}",
        pressure.latest_event()
    );

    // Call periodically on the model's owning thread using an application-
    // specific threshold. Outstanding model clones or states prevent eviction.
    println!("evicted: {:?}", cache.evict_idle(Duration::from_secs(300)));
    Ok(())
}
