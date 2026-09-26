//! Cross-language parity for the fingerprinter and index predicates.
mod common;

use std::collections::BTreeMap;

use common::{build_vars, decode_record, load_golden_value, parse_record_id};
use dedupe::blocking::Fingerprinter;
use dedupe::datamodel::DataModel;
use dedupe::value::{Data, RecordId};

fn compute_groups(
    fp: &Fingerprinter,
    data: &Data,
    predicate_repr: &str,
    target: bool,
) -> Vec<Vec<i64>> {
    let sorted: Vec<(RecordId, dedupe::Record)> = {
        let mut v: Vec<_> = data.iter().map(|(k, r)| (k.clone(), r.clone())).collect();
        v.sort_by(|a, b| a.0.cmp(&b.0));
        v
    };
    for predicate in &fp.predicates {
        if predicate.0.repr() != predicate_repr {
            continue;
        }
        let mut blocks: BTreeMap<String, Vec<i64>> = BTreeMap::new();
        for (rid, record) in &sorted {
            if let Ok(keys) = predicate.call(record, target) {
                for k in keys {
                    blocks
                        .entry(k)
                        .or_default()
                        .push(rid.as_int().unwrap());
                }
            }
        }
        let mut groups: Vec<Vec<i64>> = blocks
            .into_values()
            .map(|mut g| {
                g.sort_unstable();
                g
            })
            .collect();
        groups.sort();
        return groups;
    }
    panic!("predicate {predicate_repr} not found");
}

#[test]
fn blocking_matches_python() {
    let golden = load_golden_value("blocking.json");
    let defs = golden["defs"].as_array().unwrap().clone();
    let vars = build_vars(&defs);
    let dm = DataModel::new(vars).unwrap();
    let fp = Fingerprinter::new(dm.predicates());

    let mut data = Data::new();
    for (k, v) in golden["data"].as_object().unwrap() {
        data.insert(parse_record_id(k), decode_record(v));
    }
    fp.index_all(&data);

    let mut failures: Vec<String> = Vec::new();
    let mut total = 0usize;
    for case in golden["cases"].as_array().unwrap() {
        let repr = case["predicate"].as_str().unwrap();
        let target = case["target"].as_bool().unwrap();
        let expected: Vec<Vec<i64>> = case["expected_groups"]
            .as_array()
            .unwrap()
            .iter()
            .map(|g| {
                g.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_i64().unwrap())
                    .collect()
            })
            .collect();
        let got = compute_groups(&fp, &data, repr, target);
        total += 1;
        if got != expected {
            failures.push(format!("{repr} target={target}: got {got:?} expected {expected:?}"));
        }
    }

    if !failures.is_empty() {
        let shown: Vec<&String> = failures.iter().take(20).collect();
        panic!(
            "{} blocking failures across {total} cases:\n{}",
            failures.len(),
            shown.iter().map(|s| s.as_str()).collect::<Vec<_>>().join("\n")
        );
    }
}
