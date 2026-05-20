use std::{
    env::temp_dir,
    fs, io,
    io::Write,
    sync::atomic::{AtomicU64, Ordering},
};

static COUNTER: AtomicU64 = AtomicU64::new(0);

pub(crate) fn write_temp_file(data: &[u8]) -> io::Result<Vec<u8>> {
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let file_name = format!("coreml-{}-{}.bin", std::process::id(), id);
    let path = temp_dir().join(file_name);

    let mut file = fs::File::create(&path)?;
    file.write_all(data)?;

    let result = fs::read(&path)?;
    let _ = fs::remove_file(&path);

    Ok(result)
}
