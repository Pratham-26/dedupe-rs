//! End-to-end smoke tests for the active-learning API (the Python reference
//! uses a global RNG here, so these assert functional behaviour rather than
//! byte parity).

use std::collections::HashMap;

use dedupe::api::{Dedupe, Gazetteer, JoinConstraint, RecordLink};
use dedupe::serializer::TrainingData;
use dedupe::value::{Data, Record, RecordId, Value};
use dedupe::variables::{self, VariableDef};

fn record(name: &str, age: &str) -> Record {
    let mut r = Record::new();
    r.insert("name".to_string(), Value::str(name));
    r.insert("age".to_string(), Value::str(age));
    r
}

fn dataset() -> Data {
    let rows = [
        (0, "Bob", "51"),
        (1, "Linda", "50"),
        (2, "Gene", "12"),
        (3, "Tina", "15"),
        (4, "Bob B.", "51"),
        (5, "bob belcher", "51"),
        (6, "linda ", "50"),
        (7, "Gene Belcher", "12"),
    ];
    let mut data = Data::new();
    for (id, name, age) in rows {
        data.insert(RecordId::Int(id), record(name, age));
    }
    data
}

fn variables() -> Vec<VariableDef> {
    vec![
        VariableDef::Field(variables::string("name", false)),
        VariableDef::Field(variables::string("age", false)),
    ]
}

fn training_pairs(data: &Data) -> TrainingData {
    let p = |a: i64, b: i64| (data[&RecordId::Int(a)].clone(), data[&RecordId::Int(b)].clone());
    TrainingData {
        match_: vec![p(0, 4), p(0, 5), p(1, 6), p(2, 7), p(4, 5)],
        distinct: vec![p(0, 1), p(2, 3), p(0, 2), p(1, 3), p(4, 1), p(5, 3)],
    }
}

fn cluster_of(clusters: &[dedupe::clustering::Cluster]) -> HashMap<i64, usize> {
    let mut map = HashMap::new();
    for (i, (ids, _)) in clusters.iter().enumerate() {
        for id in ids {
            map.insert(id.as_int().unwrap(), i);
        }
    }
    map
}

#[test]
fn dedupe_active_learning_partition() {
    let data = dataset();
    let mut deduper = Dedupe::new(variables(), 1, true).unwrap().with_seed(42);
    deduper.prepare_training(&data).unwrap();

    // Actively label a handful of pairs, then add the known training data.
    for _ in 0..5 {
        let _ = deduper.uncertain_pairs();
    }
    deduper.mark_pairs(&training_pairs(&data)).unwrap();
    deduper.train(0.9, false).unwrap();

    let clusters = deduper.partition(&data, 0.5).unwrap();
    let map = cluster_of(&clusters);
    for (a, b) in [(0, 4), (0, 5), (1, 6), (2, 7)] {
        assert_eq!(
            map.get(&a),
            map.get(&b),
            "expected {a} and {b} to cluster together; clusters={clusters:?}"
        );
    }
    // Distinct records should not all collapse.
    assert!(
        clusters.len() >= 4,
        "expected several clusters, got {}",
        clusters.len()
    );
}

#[test]
fn record_link_join() {
    let data = dataset();
    let mut data_1 = Data::new();
    let mut data_2 = Data::new();
    for (k, v) in &data {
        if k.as_int().unwrap() % 2 == 0 {
            data_1.insert(k.clone(), v.clone());
        } else {
            data_2.insert(k.clone(), v.clone());
        }
    }

    let mut linker = RecordLink::new(variables(), 1, true).unwrap().with_seed(7);
    linker.prepare_training(&data_1, &data_2).unwrap();

    // Label a few pairs using the data we already have.
    let mut labeled = TrainingData::default();
    labeled
        .match_
        .push((data_1[&RecordId::Int(0)].clone(), data_2[&RecordId::Int(5)].clone()));
    labeled
        .distinct
        .push((data_1[&RecordId::Int(0)].clone(), data_2[&RecordId::Int(3)].clone()));
    linker.mark_pairs(&labeled).unwrap();
    linker.train(0.9, false).unwrap();

    let links = linker
        .join(&data_1, &data_2, 0.5, JoinConstraint::ManyToMany)
        .unwrap();
    assert!(!links.is_empty());
}

#[test]
fn gazetteer_search() {
    let data = dataset();
    let mut gazetteer = Gazetteer::new(variables(), 1, true).unwrap().with_seed(3);
    gazetteer.prepare_training(&data, &data).unwrap();

    let mut labeled = TrainingData::default();
    labeled
        .match_
        .push((data[&RecordId::Int(0)].clone(), data[&RecordId::Int(5)].clone()));
    labeled
        .distinct
        .push((data[&RecordId::Int(0)].clone(), data[&RecordId::Int(3)].clone()));
    gazetteer.mark_pairs(&labeled).unwrap();
    gazetteer.train(0.9, false).unwrap();

    // Index canonical records and search a messy copy.
    gazetteer.index(&data).unwrap();
    let mut query = Data::new();
    query.insert(RecordId::Int(100), record("bob belcher", "51"));
    let results = gazetteer.search(&query, 0.5, 2).unwrap();
    assert!(results.contains_key(&RecordId::Int(100)));
}
