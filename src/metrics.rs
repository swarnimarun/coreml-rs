use std::{
    collections::VecDeque,
    sync::{Arc, Mutex, MutexGuard},
    time::{Duration, Instant},
};

/// Number of recent completed requests retained for percentile calculation.
pub const LATENCY_WINDOW: usize = 256;

/// Observed request metrics for one loaded model instance.
///
/// `elapsed` and `latency` include both successful and failed requests.
/// Batched prediction counts as one request; `batch_items` counts its items.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ModelMetrics {
    pub started: u64,
    pub completed: u64,
    pub failed: u64,
    pub in_flight: u64,
    pub batch_items: u64,
    pub elapsed: Duration,
    /// Time spent configuring and loading this instance, including the wait
    /// for Core ML's asynchronous in-memory loader when applicable.
    pub load_duration: Duration,
    /// Monotonic timestamps, suitable for calculating elapsed time in-process.
    pub last_started: Option<Instant>,
    pub last_completed: Option<Instant>,
    /// Time since the last request finished, or None while active / never used.
    pub idle_for: Option<Duration>,
    /// Percentiles of the most recent completed requests, including failures.
    pub latency: Option<LatencySummary>,
    /// Core ML's public prediction interface does not attribute hardware usage.
    pub hardware_usage: Option<HardwareUsage>,
}

/// Nearest-rank percentiles over at most [`LATENCY_WINDOW`] recent requests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LatencySummary {
    pub samples: usize,
    pub min: Duration,
    pub p50: Duration,
    pub p95: Duration,
    pub p99: Duration,
    pub max: Duration,
}

/// Reserved for documented per-model hardware attribution when available.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HardwareUsage {
    _private: (),
}

#[derive(Default, Debug)]
struct MetricsState {
    counters: ModelMetrics,
    recent: VecDeque<Duration>,
}

/// A thread-safe observer that remains usable while a prediction is in progress.
#[derive(Clone, Default, Debug)]
pub struct MetricsHandle(Arc<Mutex<MetricsState>>);
impl MetricsHandle {
    pub(crate) fn with_load_duration(duration: Duration) -> Self {
        let metrics = Self::default();
        metrics.lock().counters.load_duration = duration;
        metrics
    }
    fn lock(&self) -> MutexGuard<'_, MetricsState> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
    pub fn snapshot(&self) -> ModelMetrics {
        let (mut metrics, mut values) = {
            let state = self.lock();
            (
                state.counters,
                state.recent.iter().copied().collect::<Vec<_>>(),
            )
        };
        metrics.idle_for = if metrics.in_flight == 0 {
            metrics.last_completed.map(|instant| instant.elapsed())
        } else {
            None
        };
        if !values.is_empty() {
            values.sort_unstable();
            let nearest_rank = |percent: usize| values[(values.len() * percent).div_ceil(100) - 1];
            metrics.latency = Some(LatencySummary {
                samples: values.len(),
                min: values[0],
                p50: nearest_rank(50),
                p95: nearest_rank(95),
                p99: nearest_rank(99),
                max: values[values.len() - 1],
            });
        }
        metrics
    }
    pub(crate) fn idle_for_eviction(&self, inserted_at: Instant) -> Option<Duration> {
        let state = self.lock();
        if state.counters.in_flight != 0 {
            None
        } else {
            Some(
                state
                    .counters
                    .last_completed
                    .unwrap_or(inserted_at)
                    .elapsed(),
            )
        }
    }
    pub(crate) fn start(&self, items: u64) -> Measurement {
        let start = Instant::now();
        let mut state = self.lock();
        let m = &mut state.counters;
        m.started = m.started.saturating_add(1);
        m.in_flight = m.in_flight.saturating_add(1);
        m.batch_items = m.batch_items.saturating_add(items);
        m.last_started = Some(start);
        Measurement {
            metrics: self.clone(),
            start,
            success: false,
        }
    }
}
pub(crate) struct Measurement {
    metrics: MetricsHandle,
    start: Instant,
    success: bool,
}
impl Measurement {
    pub fn success(&mut self) {
        self.success = true;
    }
}
impl Drop for Measurement {
    fn drop(&mut self) {
        let finished = Instant::now();
        let elapsed = finished.duration_since(self.start);
        let mut state = self.metrics.lock();
        let m = &mut state.counters;
        m.in_flight = m.in_flight.saturating_sub(1);
        if self.success {
            m.completed = m.completed.saturating_add(1)
        } else {
            m.failed = m.failed.saturating_add(1)
        }
        m.elapsed = m.elapsed.saturating_add(elapsed);
        m.last_completed = Some(finished);
        if state.recent.len() == LATENCY_WINDOW {
            state.recent.pop_front();
        }
        state.recent.push_back(elapsed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn observer_sees_active_and_failed_calls() {
        let metrics = MetricsHandle::default();
        assert_eq!(metrics.snapshot().idle_for, None);
        let measurement = metrics.start(3);
        let observer = metrics.clone();
        let during = std::thread::spawn(move || observer.snapshot())
            .join()
            .unwrap();
        assert_eq!(
            (during.started, during.in_flight, during.batch_items),
            (1, 1, 3)
        );
        assert!(during.last_started.is_some());
        assert_eq!(during.idle_for, None);
        drop(measurement);
        let after = metrics.snapshot();
        assert_eq!((after.failed, after.in_flight), (1, 0));
        assert!(after.last_completed.is_some());
        assert!(after.idle_for.is_some());
        assert_eq!(after.latency.unwrap().samples, 1);
    }
    #[test]
    fn nearest_rank_percentiles_are_correct() {
        let metrics = MetricsHandle::default();
        {
            let mut state = metrics.lock();
            state.recent.extend((1..=100).map(Duration::from_millis));
        }
        let latency = metrics.snapshot().latency.unwrap();
        assert_eq!(latency.p50, Duration::from_millis(50));
        assert_eq!(latency.p95, Duration::from_millis(95));
        assert_eq!(latency.p99, Duration::from_millis(99));
    }
    #[test]
    fn recent_latency_window_is_bounded() {
        let metrics = MetricsHandle::default();
        for _ in 0..LATENCY_WINDOW + 7 {
            metrics.start(0).success();
        }
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.completed, (LATENCY_WINDOW + 7) as u64);
        let latency = snapshot.latency.unwrap();
        assert_eq!(latency.samples, LATENCY_WINDOW);
        assert!(
            latency.min <= latency.p50
                && latency.p50 <= latency.p95
                && latency.p95 <= latency.p99
                && latency.p99 <= latency.max
        );
    }
}
