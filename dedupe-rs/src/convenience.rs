//! Convenience helpers.  Ported from `dedupe/convenience.py`.

use std::collections::{BTreeSet, HashMap};

use rand::{Rng, SeedableRng};

use crate::canonical::get_canonical_rep;
use crate::serializer::TrainingData;
use crate::value::{Data, Record, RecordId, Value};

/// Random combinations of indices for a square matrix of size `n_records`.
pub fn random_pairs(n_records: usize, sample_size: usize) -> Vec<(usize, usize)> {
    let n = n_records * n_records.saturating_sub(1) / 2;
    if sample_size == 0 || n == 0 {
        return Vec::new();
    }
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xDED0);
    let indices: Vec<usize> = if sample_size >= n {
        (0..n).collect()
    } else {
        // Sample without replacement.
        let mut set = BTreeSet::new();
        while set.len() < sample_size {
            set.insert(rng.random_range(0..n));
        }
        set.into_iter().collect()
    };

    let b = 1.0 - 2.0 * n_records as f64;
    indices
        .into_iter()
        .map(|rp| {
            let rp = rp as f64;
            let i = ((-b - 2.0 * (2.0 * (n as f64 - rp) + 0.25).sqrt()) / 2.0).trunc() as i64;
            let j = rp as i64 + i * (b as i64 + i + 2) / 2 + 1;
            (i as usize, j as usize)
        })
        .collect()
}

/// Random combinations of indices for two record lists.
pub fn random_pairs_match(
    n_records_a: usize,
    n_records_b: usize,
    sample_size: usize,
) -> Vec<(usize, usize)> {
    let n = n_records_a * n_records_b;
    if sample_size == 0 || n == 0 {
        return Vec::new();
    }
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x5EED);
    let indices: Vec<usize> = if sample_size >= n {
        (0..n).collect()
    } else {
        let mut set = BTreeSet::new();
        while set.len() < sample_size {
            set.insert(rng.random_range(0..n));
        }
        set.into_iter().collect()
    };
    indices
        .into_iter()
        .map(|idx| (idx / n_records_b, idx % n_records_b))
        .collect()
}

/// Construct training data from already-linked datasets.
pub fn training_data_link(
    data_1: &Data,
    data_2: &Data,
    common_key: &str,
    training_size: usize,
) -> TrainingData {
    let mut identified: HashMap<String, (Vec<RecordId>, Vec<RecordId>)> = HashMap::new();
    let mut matched_pairs: BTreeSet<(RecordId, RecordId)> = BTreeSet::new();

    for (record_id, record) in data_1 {
        let key = record.get(common_key).map(|v| v.py_str()).unwrap_or_default();
        identified.entry(key).or_default().0.push(record_id.clone());
    }
    for (record_id, record) in data_2 {
        let key = record.get(common_key).map(|v| v.py_str()).unwrap_or_default();
        identified.entry(key).or_default().1.push(record_id.clone());
    }
    for (keys_1, keys_2) in identified.values() {
        for a in keys_1 {
            for b in keys_2 {
                matched_pairs.insert((a.clone(), b.clone()));
            }
        }
    }

    let keys_1: Vec<RecordId> = data_1.keys().cloned().collect();
    let keys_2: Vec<RecordId> = data_2.keys().cloned().collect();
    let random = random_pairs_match(data_1.len(), data_2.len(), training_size);

    let mut distinct_pairs: BTreeSet<(RecordId, RecordId)> = BTreeSet::new();
    for (i, j) in random {
        let pair = (keys_1[i].clone(), keys_2[j].clone());
        if !matched_pairs.contains(&pair) {
            distinct_pairs.insert(pair);
        }
    }

    let match_ = matched_pairs
        .iter()
        .filter_map(|(a, b)| Some((data_1.get(a)?.clone(), data_2.get(b)?.clone())))
        .collect();
    let distinct = distinct_pairs
        .iter()
        .filter_map(|(a, b)| Some((data_1.get(a)?.clone(), data_2.get(b)?.clone())))
        .collect();

    TrainingData { match_, distinct }
}

/// Construct training data from an already-deduplicated dataset.
pub fn training_data_dedupe(data: &Data, common_key: &str, training_size: usize) -> TrainingData {
    let mut identified: HashMap<String, Vec<RecordId>> = HashMap::new();
    let mut matched_pairs: BTreeSet<(RecordId, RecordId)> = BTreeSet::new();

    for (record_id, record) in data {
        let key = record.get(common_key).map(|v| v.py_str()).unwrap_or_default();
        identified.entry(key).or_default().push(record_id.clone());
    }
    for record_ids in identified.values() {
        if record_ids.len() > 1 {
            let mut sorted = record_ids.clone();
            sorted.sort();
            for i in 0..sorted.len() {
                for j in (i + 1)..sorted.len() {
                    matched_pairs.insert((sorted[i].clone(), sorted[j].clone()));
                }
            }
        }
    }

    let unique_record_ids: Vec<RecordId> = data.keys().cloned().collect();
    let pair_indices = random_pairs(unique_record_ids.len(), training_size);
    let mut distinct_pairs: BTreeSet<(RecordId, RecordId)> = BTreeSet::new();
    for (i, j) in pair_indices {
        distinct_pairs.insert((
            unique_record_ids[i].clone(),
            unique_record_ids[j].clone(),
        ));
    }
    for pair in &matched_pairs {
        distinct_pairs.remove(pair);
    }

    let match_ = matched_pairs
        .iter()
        .filter_map(|(a, b)| Some((data.get(a)?.clone(), data.get(b)?.clone())))
        .collect();
    let distinct = distinct_pairs
        .iter()
        .filter_map(|(a, b)| Some((data.get(a)?.clone(), data.get(b)?.clone())))
        .collect();

    TrainingData { match_, distinct }
}

/// Construct a canonical representation of a duplicate cluster.
pub fn canonicalize(record_cluster: &[Record]) -> Record {
    get_canonical_rep(record_cluster)
}

/// Convenience wrapper matching Python's `dedupe.console_label` labeling input.
pub fn labeled_pair(
    record_pair: (Record, Record),
    label: &str,
) -> TrainingData {
    let mut training = TrainingData::default();
    match label {
        "match" => training.match_.push(record_pair),
        "distinct" => training.distinct.push(record_pair),
        "unsure" => {
            training.match_.push(record_pair.clone());
            training.distinct.push(record_pair);
        }
        _ => {}
    }
    training
}

#[allow(dead_code)]
fn value_key(v: &Value) -> String {
    v.py_str()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_pairs_count() {
        let pairs = random_pairs(10, 5);
        assert_eq!(pairs.len(), 5);
        for (i, j) in pairs {
            assert!(i < j && j < 10);
        }
    }

    #[test]
    fn training_data_dedupe_finds_matches() {
        let mut data = Data::new();
        for (id, name) in [(1, "bob"), (2, "bob"), (3, "sue")] {
            let mut r = Record::new();
            r.insert("name".to_string(), Value::str(name));
            r.insert("key".to_string(), Value::str(name));
            data.insert(RecordId::Int(id), r);
        }
        let training = training_data_dedupe(&data, "key", 100);
        assert!(!training.match_.is_empty());
    }
}
