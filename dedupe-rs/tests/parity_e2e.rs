//! End-to-end parity: reproduce Python's trained partition in Rust from a
//! serialized settings file.
mod common;

use std::collections::BTreeSet;

use common::{decode_record, load_golden_value, parse_record_id};
use dedupe::api::{Settings, StaticDedupe};
use dedupe::value::Data;

fn summary(clusters: &[dedupe::clustering::Cluster]) -> BTreeSet<(Vec<i64>, Vec<i64>)> {
    clusters
        .iter()
        .map(|(ids, scores)| {
            (
                ids.iter().map(|i| i.as_int().unwrap()).collect(),
                scores.iter().map(|s| (s * 1_000_000.0).round() as i64).collect(),
            )
        })
        .collect()
}

#[test]
fn end_to_end_partition_matches_python() {
    let golden = load_golden_value("e2e.json");

    let settings: Settings = serde_json::from_value(golden["settings"].clone()).unwrap();
    let matcher = StaticDedupe::from_settings(&settings, 1, true).unwrap();

    let mut data = Data::new();
    for (k, v) in golden["data"].as_object().unwrap() {
        data.insert(parse_record_id(k), decode_record(v));
    }

    let mut failures: Vec<String> = Vec::new();
    for (threshold, expected) in golden["partitions"].as_object().unwrap() {
        let t: f64 = threshold.parse().unwrap();
        let expected: BTreeSet<(Vec<i64>, Vec<i64>)> = expected
            .as_array()
            .unwrap()
            .iter()
            .map(|c| {
                (
                    c["ids"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect(),
                    c["scores"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| (v.as_f64().unwrap() * 1_000_000.0).round() as i64)
                        .collect(),
                )
            })
            .collect();

        let clusters = matcher.partition(&data, t).unwrap();
        let got = summary(&clusters);
        if got != expected {
            failures.push(format!("threshold {t}: got {got:?} expected {expected:?}"));
        }
    }

    if !failures.is_empty() {
        panic!("end-to-end partition failures:\n{}", failures.join("\n"));
    }
}
