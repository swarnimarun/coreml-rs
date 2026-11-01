//! Process-wide memory snapshots and system memory-pressure notifications.
use crate::{Error, Result};
use block2::RcBlock;
use dispatch2::{
    _dispatch_source_type_memorypressure, DispatchObject, DispatchRetained, DispatchSource,
};
use std::{
    sync::{Arc, Mutex},
    time::Instant,
};

/// Current process memory, including all code and data, not a model attribution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProcessMemory {
    pub resident_bytes: u64,
    pub physical_footprint_bytes: u64,
    pub wired_bytes: u64,
}

impl ProcessMemory {
    /// Read macOS process accounting for the current process.
    pub fn current() -> Result<Self> {
        // SAFETY: zero is a valid initial bit pattern for rusage_info_v0.
        let mut info: libc::rusage_info_v0 = unsafe { std::mem::zeroed() };
        // SAFETY: the v0 flavor writes exactly rusage_info_v0 bytes into the
        // provided buffer, as documented by proc_pid_rusage.
        let status = unsafe {
            libc::proc_pid_rusage(
                std::process::id() as libc::c_int,
                libc::RUSAGE_INFO_V0,
                (&mut info as *mut libc::rusage_info_v0).cast(),
            )
        };
        if status != 0 {
            return Err(Error::io(
                "read process memory",
                std::io::Error::last_os_error(),
            ));
        }
        Ok(Self {
            resident_bytes: info.ri_resident_size,
            physical_footprint_bytes: info.ri_phys_footprint,
            wired_bytes: info.ri_wired_size,
        })
    }
}

/// A delivered system memory-pressure notification.
///
/// Dispatch may coalesce events, so more than one flag can be set. The event
/// describes a system-wide change, not a measurement of a model's memory use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryPressureEvent {
    pub normal: bool,
    pub warning: bool,
    pub critical: bool,
    pub observed_at: Instant,
    /// Number of delivered notifications since this monitor was created.
    pub deliveries: u64,
}

/// An opt-in observer of system memory-pressure notifications.
///
/// No initial pressure level is assumed. The latest event is `None` until
/// macOS delivers a notification. Dropping the monitor cancels observation.
pub struct MemoryPressureMonitor {
    source: Arc<Mutex<Option<DispatchRetained<DispatchSource>>>>,
    latest: Arc<Mutex<Option<MemoryPressureEvent>>>,
}

fn record_pressure(latest: &Mutex<Option<MemoryPressureEvent>>, flags: usize) {
    let mut latest = latest.lock().unwrap_or_else(|error| error.into_inner());
    let deliveries = latest.map_or(1, |event| event.deliveries.saturating_add(1));
    *latest = Some(MemoryPressureEvent {
        normal: flags & 0x01 != 0,
        warning: flags & 0x02 != 0,
        critical: flags & 0x04 != 0,
        observed_at: Instant::now(),
        deliveries,
    });
}

impl MemoryPressureMonitor {
    pub fn new() -> Self {
        // SAFETY: this is the documented memory-pressure source type; its
        // handle must be zero, and the mask requests all three event flags.
        let source = unsafe {
            DispatchSource::new(
                std::ptr::addr_of!(_dispatch_source_type_memorypressure).cast_mut(),
                0,
                0x01 | 0x02 | 0x04,
                None,
            )
        };
        let source = Arc::new(Mutex::new(Some(source)));
        let weak_source = Arc::downgrade(&source);
        let latest: Arc<Mutex<Option<MemoryPressureEvent>>> = Arc::new(Mutex::new(None));
        let callback_latest = latest.clone();
        let callback = RcBlock::new(move || {
            // A weak reference avoids a source -> block -> source retain cycle.
            // The lock keeps the source alive until its event data is read.
            let Some(source) = weak_source.upgrade() else {
                return;
            };
            let source = source.lock().unwrap_or_else(|error| error.into_inner());
            let Some(source) = source.as_ref() else {
                return;
            };
            record_pressure(&callback_latest, source.data());
        });
        {
            let guard = source.lock().unwrap_or_else(|error| error.into_inner());
            let source = guard.as_ref().expect("new dispatch source");
            // SAFETY: the block captures only synchronized shared state and a
            // weak reference; Dispatch copies it before this call returns.
            unsafe { source.set_event_handler_with_block(RcBlock::as_ptr(&callback)) };
            source.activate();
        }
        Self { source, latest }
    }

    pub fn latest_event(&self) -> Option<MemoryPressureEvent> {
        *self
            .latest
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }
}

impl Default for MemoryPressureMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for MemoryPressureMonitor {
    fn drop(&mut self) {
        let source = self
            .source
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take();
        if let Some(source) = source {
            source.cancel();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_process_memory_has_a_resident_footprint() {
        let memory = ProcessMemory::current().unwrap();
        assert!(memory.resident_bytes > 0);
        assert!(memory.physical_footprint_bytes > 0);
    }

    #[test]
    fn memory_pressure_monitor_can_start_and_stop() {
        let monitor = MemoryPressureMonitor::new();
        let _possibly_first_event = monitor.latest_event();
        drop(monitor);
    }

    #[test]
    fn records_coalesced_pressure_events_and_delivery_count() {
        let latest = Mutex::new(None);
        record_pressure(&latest, 0x02 | 0x04);
        let event = latest.lock().unwrap().unwrap();
        assert!(!event.normal && event.warning && event.critical);
        assert_eq!(event.deliveries, 1);
        record_pressure(&latest, 0x01);
        let event = latest.lock().unwrap().unwrap();
        assert!(event.normal && !event.warning && !event.critical);
        assert_eq!(event.deliveries, 2);
    }
}
