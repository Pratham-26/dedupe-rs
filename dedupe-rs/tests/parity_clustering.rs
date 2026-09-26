//! Cross-language parity for clustering (centroid linkage + fcluster).
mod common;

use std::collections::BTreeSet;

use common::{decode, load_golden_value};
use dedupe::clustering::{cluster, connected_components, ScoredPair};
use dedupe::value::RecordId;

fn rid(v: &serde_json::Value) -> RecordId {
    match decode(v) {
        dedupe::Value::Int(i) => RecordId::Int(i),
        dedupe::Value::Str(s) => RecordId::Str(s),
        other => panic!("bad id {other:?}"),
    }
}

fn pairs(v: &serde_json::Value) -> Vec<ScoredPair> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|p| {
            (
                (rid(&p["a"]), rid(&p["b"])),
                p["score"].as_f64().unwrap() as f32,
            )
        })
        .collect()
}

fn summary(clusters: &[dedupe::clustering::Cluster]) -> BTreeSet<(Vec<i64>, Vec<i64>)> {
    clusters
        .iter()
        .map(|(ids, scores)| {
            (
                ids.iter().map(|i| i.as_int().unwrap()).collect(),
                scores.iter().map(|s| (s * 1000.0).round() as i64).collect(),
            )
        })
        .collect()
}

#[test]
fn clustering_matches_python() {
    let golden = load_golden_value("clustering.json");
    let mut failures: Vec<String> = Vec::new();
    let mut total = 0usize;

    for case in golden["cases"].as_array().unwrap() {
        let p = pairs(&case["pairs"]);
        let threshold = case["threshold"].as_f64().unwrap();
        let expected: BTreeSet<(Vec<i64>, Vec<i64>)> = case["expected"]
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
                        .map(|v| (v.as_f64().unwrap() * 1000.0).round() as i64)
                        .collect(),
                )
            })
            .collect();
        let got = summary(&cluster(&p, threshold, 30000));
        total += 1;
        if got != expected {
            failures.push(format!("threshold {threshold}: got {got:?} expected {expected:?}"));
        }
    }

    // connected components
    let comps_case = &golden["components"];
    let comps = connected_components(&pairs(&comps_case["pairs"]));
    let got: BTreeSet<Vec<(i64, i64)>> = comps
        .iter()
        .map(|c| {
            let mut v: Vec<(i64, i64)> = c
                .iter()
                .map(|((a, b), _)| (a.as_int().unwrap(), b.as_int().unwrap()))
                .collect();
            v.sort_unstable();
            v
        })
        .collect();
    let expected: BTreeSet<Vec<(i64, i64)>> = comps_case["expected"]
        .as_array()
        .unwrap()
        .iter()
        .map(|g| {
            g.as_array()
                .unwrap()
                .iter()
                .map(|e| (e[0].as_i64().unwrap(), e[1].as_i64().unwrap()))
                .collect()
        })
        .collect();
    total += 1;
    if got != expected {
        failures.push(format!("components: got {got:?} expected {expected:?}"));
    }

    if !failures.is_empty() {
        let shown: Vec<&String> = failures.iter().take(15).collect();
        panic!(
            "{} clustering failures across {total} cases:\n{}",
            failures.len(),
            shown.iter().map(|s| s.as_str()).collect::<Vec<_>>().join("\n")
        );
    }
}

#[test]
fn linkage_matches_scipy() {
    let golden = load_golden_value("clustering.json");
    let mut failures: Vec<String> = Vec::new();
    let mut total = 0usize;
    for case in golden["linkage"].as_array().unwrap() {
        let n = case["n"].as_u64().unwrap() as usize;
        let condensed: Vec<f64> = case["condensed"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let expected: Vec<(usize, usize, f64, usize)> = case["Z"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| {
                let r = row.as_array().unwrap();
                (
                    r[0].as_f64().unwrap() as usize,
                    r[1].as_f64().unwrap() as usize,
                    r[2].as_f64().unwrap(),
                    r[3].as_f64().unwrap() as usize,
                )
            })
            .collect();
        let got = dedupe::clustering::linkage_centroid(&condensed, n);
        total += 1;
        let ok = got.len() == expected.len()
            && got.iter().zip(expected.iter()).all(|(g, e)| {
                g.0 == e.0
                    && g.1 == e.1
                    && (g.2 - e.2).abs() < 1e-9
                    && g.3 == e.3
            });
        if !ok {
            failures.push(format!(
                "n={n} condensed={condensed:?}\n got {got:?}\n exp {expected:?}"
            ));
        }
    }
    if !failures.is_empty() {
        let shown: Vec<&String> = failures.iter().take(3).collect();
        panic!(
            "{} linkage failures across {total} components:\n{}",
            failures.len(),
            shown.iter().map(|s| s.as_str()).collect::<Vec<_>>().join("\n")
        );
    }
}
