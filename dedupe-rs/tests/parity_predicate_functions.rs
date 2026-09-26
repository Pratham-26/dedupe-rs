//! Cross-language parity for blocking predicate functions.
mod common;

use common::{decode, load_golden};
use dedupe::predicate_functions as pf;

#[test]
fn predicate_functions_match_python() {
    let cases = load_golden("predicate_functions.json");
    let mut failures = 0usize;
    let mut first: Vec<String> = Vec::new();
    let mut total = 0usize;

    for case in &cases {
        let obj = case.as_object().unwrap();
        let name = obj["fn"].as_str().unwrap();
        let input = decode(&obj["input"]);
        let expected: Vec<String> = obj["expected"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();

        let Some(func) = pf::by_name(name) else {
            panic!("no Rust predicate function registered for {name}");
        };
        let got: Vec<String> = func(&input).into_iter().collect();
        total += 1;
        if got != expected {
            failures += 1;
            if first.len() < 25 {
                first.push(format!(
                    "{name}({:?}) => got {:?} expected {:?}",
                    input, got, expected
                ));
            }
        }
    }

    if failures > 0 {
        panic!(
            "{failures}/{total} predicate-function cases failed:\n{}",
            first.join("\n")
        );
    }
    assert!(total > 1000);
}
