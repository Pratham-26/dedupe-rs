//! Parity for the Double Metaphone port against the Python reference.
mod common;

use common::load_golden;
use dedupe::double_metaphone::double_metaphone_codes;

#[test]
fn double_metaphone_matches_python() {
    let cases = load_golden("double_metaphone.json");
    let mut failures = 0usize;
    let mut first: Vec<String> = Vec::new();
    let mut total = 0usize;
    for case in &cases {
        let obj = case.as_object().unwrap();
        let w = obj["w"].as_str().unwrap();
        let expected: Vec<String> = obj["expected"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        let (p, s) = double_metaphone_codes(w);
        total += 1;
        if p != expected[0] || s != expected[1] {
            failures += 1;
            if first.len() < 25 {
                first.push(format!(
                    "{w:?} => got ({p:?}, {s:?}) expected ({:?}, {:?})",
                    expected[0], expected[1]
                ));
            }
        }
    }
    if failures > 0 {
        panic!("{failures}/{total} double-metaphone cases failed:\n{}", first.join("\n"));
    }
    assert!(total > 100_000);
}
