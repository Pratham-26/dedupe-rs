//! Core scoring utilities.  Ported from `dedupe/core.py`.

use rayon::prelude::*;

use crate::clustering::ScoredPair;
use crate::datamodel::DistanceCalculator;
use crate::logistic::Classifier;
use crate::value::{Data, Record, RecordId, Value};

/// Build a thread pool with exactly `num_cores` threads.
fn pool(num_cores: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_cores)
        .build()
        .expect("thread pool")
}

/// Score ordered record pairs using a reusable per-thread row buffer.
fn score_owned(
    pairs: &[((RecordId, Record), (RecordId, Record))],
    calculator: &DistanceCalculator,
    classifier: &dyn Classifier,
) -> Vec<ScoredPair> {
    let width = calculator.len();
    let mut buf = vec![0f32; width];
    let mut out = Vec::new();
    for pair in pairs {
        calculator.distance_row_into(&pair.0 .1, &pair.1 .1, &mut buf);
        let score = classifier.predict_proba_f32(&buf);
        if score > 0.0 {
            out.push(((pair.0 .0.clone(), pair.1 .0.clone()), score as f32));
        }
    }
    out
}

/// Score pairs of records, keeping only strictly positive predictions.
pub fn score_duplicates(
    pairs: &[((RecordId, Record), (RecordId, Record))],
    calculator: &DistanceCalculator,
    classifier: &dyn Classifier,
    num_cores: usize,
) -> Vec<ScoredPair> {
    if num_cores <= 1 {
        return score_owned(pairs, calculator, classifier);
    }
    let p = pool(num_cores);
    p.install(|| {
        pairs
            .par_iter()
            .map_init(
                || vec![0f32; calculator.len()],
                |buf, pair| -> Option<ScoredPair> {
                    calculator.distance_row_into(&pair.0 .1, &pair.1 .1, buf);
                    let score = classifier.predict_proba_f32(buf);
                    (score > 0.0)
                        .then(|| ((pair.0 .0.clone(), pair.1 .0.clone()), score as f32))
                },
            )
            .flatten()
            .collect()
    })
}

/// Score id pairs by looking records up in `data` (avoids cloning records).
pub fn score_duplicate_ids(
    data: &Data,
    ids: &[(RecordId, RecordId)],
    calculator: &DistanceCalculator,
    classifier: &dyn Classifier,
    num_cores: usize,
) -> Vec<ScoredPair> {
    if num_cores <= 1 {
        let mut buf = vec![0f32; calculator.len()];
        let mut out = Vec::new();
        for pair in ids {
            let (Some(a), Some(b)) = (data.get(&pair.0), data.get(&pair.1)) else {
                continue;
            };
            calculator.distance_row_into(a, b, &mut buf);
            let score = classifier.predict_proba_f32(&buf);
            if score > 0.0 {
                out.push(((pair.0.clone(), pair.1.clone()), score as f32));
            }
        }
        return out;
    }
    let p = pool(num_cores);
    p.install(|| {
        ids.par_iter()
            .map_init(
                || vec![0f32; calculator.len()],
                |buf, pair| -> Option<ScoredPair> {
                    let a = data.get(&pair.0)?;
                    let b = data.get(&pair.1)?;
                    calculator.distance_row_into(a, b, buf);
                    let score = classifier.predict_proba_f32(buf);
                    (score > 0.0)
                        .then(|| ((pair.0.clone(), pair.1.clone()), score as f32))
                },
            )
            .flatten()
            .collect()
    })
}

/// Score linkage id pairs by looking records up in two datasets.
pub fn score_link_ids(
    data_1: &Data,
    data_2: &Data,
    ids: &[(RecordId, RecordId)],
    calculator: &DistanceCalculator,
    classifier: &dyn Classifier,
    num_cores: usize,
) -> Vec<ScoredPair> {
    if num_cores <= 1 {
        let mut buf = vec![0f32; calculator.len()];
        let mut out = Vec::new();
        for pair in ids {
            let (Some(a), Some(b)) = (data_1.get(&pair.0), data_2.get(&pair.1)) else {
                continue;
            };
            calculator.distance_row_into(a, b, &mut buf);
            let score = classifier.predict_proba_f32(&buf);
            if score > 0.0 {
                out.push(((pair.0.clone(), pair.1.clone()), score as f32));
            }
        }
        return out;
    }
    let p = pool(num_cores);
    p.install(|| {
        ids.par_iter()
            .map_init(
                || vec![0f32; calculator.len()],
                |buf, pair| -> Option<ScoredPair> {
                    let a = data_1.get(&pair.0)?;
                    let b = data_2.get(&pair.1)?;
                    calculator.distance_row_into(a, b, buf);
                    let score = classifier.predict_proba_f32(buf);
                    (score > 0.0)
                        .then(|| ((pair.0.clone(), pair.1.clone()), score as f32))
                },
            )
            .flatten()
            .collect()
    })
}

/// Score blocks for gazetteer matching, returning per-block scored pairs.
pub fn score_gazette(
    blocks: &[Vec<((RecordId, Record), (RecordId, Record))>],
    calculator: &DistanceCalculator,
    classifier: &dyn Classifier,
) -> Vec<Vec<ScoredPair>> {
    let mut buf = vec![0f32; calculator.len()];
    blocks
        .iter()
        .map(|block| {
            let mut out = Vec::new();
            for pair in block {
                calculator.distance_row_into(&pair.0 .1, &pair.1 .1, &mut buf);
                let score = classifier.predict_proba_f32(&buf);
                if score > 0.0 {
                    out.push(((pair.0 .0.clone(), pair.1 .0.clone()), score as f32));
                }
            }
            out
        })
        .collect()
}

/// Return the unique elements of a sequence, preserving order.
pub fn unique<T: PartialEq + Clone>(seq: &[T]) -> Vec<T> {
    let mut out: Vec<T> = Vec::new();
    for item in seq {
        if !out.contains(item) {
            out.push(item.clone());
        }
    }
    out
}

/// Re-key a dataset with sequential ids starting at `offset` if needed.
pub fn index(data: &Data, offset: i64) -> Data {
    let n = data.len() as i64;
    let is_indexed = (offset..offset + n).all(|i| data.contains_key(&RecordId::Int(i)));
    if is_indexed {
        return data.clone();
    }
    let mut out = Data::new();
    for (k, v) in data.values().enumerate() {
        out.insert(RecordId::Int(offset + k as i64), v.clone());
    }
    out
}

/// Record id type used for array storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdType {
    Int,
    Str,
}

pub fn sniff_id_type(id: &RecordId) -> IdType {
    match id {
        RecordId::Int(_) => IdType::Int,
        RecordId::Str(_) => IdType::Str,
    }
}

pub fn sqlite_id_type(id: &RecordId) -> &'static str {
    match id {
        RecordId::Int(_) => "integer",
        RecordId::Str(_) => "text",
    }
}

/// A monotonically increasing id generator (Python's `Enumerator`).
#[derive(Debug, Default)]
pub struct Enumerator {
    next: usize,
}

impl Enumerator {
    pub fn new(start: usize) -> Self {
        Self { next: start }
    }
    pub fn next_id(&mut self) -> usize {
        let v = self.next;
        self.next += 1;
        v
    }
}

/// Build a record id from a string or integer field.
pub fn record_id_from_value(v: &Value) -> RecordId {
    match v {
        Value::Int(i) => RecordId::Int(*i),
        Value::Str(s) => RecordId::Str(s.clone()),
        other => RecordId::Str(other.py_str()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use indexmap::IndexMap;

    #[test]
    fn unique_preserves_order() {
        let a: Record = IndexMap::new();
        let mut b: Record = IndexMap::new();
        b.insert("x".to_string(), Value::Int(1));
        let mut a2: Record = IndexMap::new();
        a2.insert("y".to_string(), Value::Int(2));
        let out = unique(&[a.clone(), b.clone(), a2.clone()]);
        assert_eq!(out, vec![a, b, a2]);
    }

    #[test]
    fn index_rekeys() {
        let mut data = Data::new();
        data.insert(RecordId::Str("x".into()), Record::new());
        data.insert(RecordId::Str("y".into()), Record::new());
        let out = index(&data, 5);
        assert!(out.contains_key(&RecordId::Int(5)));
        assert!(out.contains_key(&RecordId::Int(6)));
    }
}
