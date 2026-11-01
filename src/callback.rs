//! Runtime-independent bridge for one-shot Objective-C completion handlers.
use crate::Result;
use std::{
    future::Future,
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Poll, Waker},
};

struct State<T> {
    result: Option<Result<T>>,
    waker: Option<Waker>,
    cancelled: bool,
}
impl<T> Default for State<T> {
    fn default() -> Self {
        Self {
            result: None,
            waker: None,
            cancelled: false,
        }
    }
}

pub(crate) struct Sender<T>(Arc<Mutex<State<T>>>);
pub(crate) struct CallbackFuture<T>(Arc<Mutex<State<T>>>);

pub(crate) fn channel<T>() -> (Sender<T>, CallbackFuture<T>) {
    let state = Arc::new(Mutex::new(State::default()));
    (Sender(state.clone()), CallbackFuture(state))
}
impl<T> Sender<T> {
    pub fn is_cancelled(&self) -> bool {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).cancelled
    }
    pub fn complete(&self, result: Result<T>) {
        let waker = {
            let mut state = self.0.lock().unwrap_or_else(|e| e.into_inner());
            if state.cancelled {
                None
            } else {
                state.result = Some(result);
                state.waker.take()
            }
        };
        if let Some(waker) = waker {
            // A custom executor waker must not unwind through Objective-C.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| waker.wake()));
        }
    }
}
impl<T> Future for CallbackFuture<T> {
    type Output = Result<T>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(result) = state.result.take() {
            Poll::Ready(result)
        } else {
            state.waker = Some(cx.waker().clone());
            Poll::Pending
        }
    }
}
impl<T> Drop for CallbackFuture<T> {
    fn drop(&mut self) {
        let mut state = self.0.lock().unwrap_or_else(|e| e.into_inner());
        state.cancelled = true;
        state.waker = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        sync::atomic::{AtomicUsize, Ordering},
        task::Wake,
    };
    struct WakeCount(AtomicUsize);
    impl Wake for WakeCount {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    #[test]
    fn handles_completion_before_and_after_polling() {
        let (sender, future) = channel();
        let mut future = Box::pin(future);
        let wakes = Arc::new(WakeCount(AtomicUsize::new(0)));
        let waker = Waker::from(wakes.clone());
        let mut cx = Context::from_waker(&waker);
        assert!(future.as_mut().poll(&mut cx).is_pending());
        sender.complete(Ok(42));
        assert_eq!(wakes.0.load(Ordering::SeqCst), 1);
        assert!(matches!(future.as_mut().poll(&mut cx), Poll::Ready(Ok(42))));
        let (sender, future) = channel();
        sender.complete(Ok(9));
        let mut future = Box::pin(future);
        assert!(matches!(future.as_mut().poll(&mut cx), Poll::Ready(Ok(9))));
    }
    #[test]
    fn dropping_future_cancels_delivery() {
        let (sender, future) = channel::<u32>();
        drop(future);
        assert!(sender.is_cancelled());
        sender.complete(Ok(1));
    }
}
