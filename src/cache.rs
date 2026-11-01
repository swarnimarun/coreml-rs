use crate::Model;
use std::{
    collections::HashMap,
    hash::Hash,
    time::{Duration, Instant},
};

struct Entry {
    model: Model,
    inserted_at: Instant,
}

/// An opt-in cache with explicit idle eviction. This cache lives on the same
/// thread as its models and never evicts them on a background thread.
pub struct ModelCache<K> {
    models: HashMap<K, Entry>,
}

impl<K> Default for ModelCache<K> {
    fn default() -> Self {
        Self {
            models: HashMap::new(),
        }
    }
}

impl<K: Eq + Hash> ModelCache<K> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Replace an entry and return the previous model, if any.
    pub fn insert(&mut self, key: K, model: Model) -> Option<Model> {
        self.models
            .insert(
                key,
                Entry {
                    model,
                    inserted_at: Instant::now(),
                },
            )
            .map(|entry| entry.model)
    }

    pub fn get(&self, key: &K) -> Option<&Model> {
        self.models.get(key).map(|entry| &entry.model)
    }

    /// Remove an entry. Other model clones or states can keep it loaded.
    pub fn remove(&mut self, key: &K) -> Option<Model> {
        self.models.remove(key).map(|entry| entry.model)
    }

    pub fn len(&self) -> usize {
        self.models.len()
    }

    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }
}

impl<K: Eq + Hash + Clone> ModelCache<K> {
    /// Drop cache-owned models idle for at least `minimum_idle` and return
    /// their keys. Models with a live clone, state, or request are retained.
    /// Models that have not received a request age from their cache insertion.
    /// Core ML and the OS may retain their own internal caches after eviction.
    pub fn evict_idle(&mut self, minimum_idle: Duration) -> Vec<K> {
        let candidates: Vec<K> = self
            .models
            .iter()
            .filter_map(|(key, entry)| {
                if !entry.model.is_exclusively_owned() {
                    return None;
                }
                let idle_for = entry.model.idle_for_eviction(entry.inserted_at)?;
                (idle_for >= minimum_idle).then(|| key.clone())
            })
            .collect();
        for key in &candidates {
            self.models.remove(key);
        }
        candidates
    }
}
