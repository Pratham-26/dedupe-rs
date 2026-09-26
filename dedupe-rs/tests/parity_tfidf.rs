//! Cross-language parity for the TF-IDF cosine index.
mod common;

use common::{decode, load_golden_value};
use dedupe::index::{Doc, Index, TfIdfIndex};

#[test]
fn tfidf_matches_python() {
    let golden = load_golden_value("tfidf.json");
    let docs: Vec<Vec<String>> = golden["docs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| {
            d.as_array()
                .unwrap()
                .iter()
                .map(|v| decode(v).as_str().unwrap().to_string())
                .collect()
        })
        .collect();

    let mut index = TfIdfIndex::new();
    for d in &docs {
        index.index_doc(&Doc::Terms(d.clone()));
    }
    index.init_search();

    let mut failures: Vec<String> = Vec::new();
    let mut order_only = 0usize;
    let mut total = 0usize;
    for case in golden["cases"].as_array().unwrap() {
        let query: Vec<String> = case["query"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| decode(v).as_str().unwrap().to_string())
            .collect();
        let threshold = case["threshold"].as_f64().unwrap();
        let expected: Vec<usize> = case["expected"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let got = index.search(&Doc::Terms(query.clone()), threshold);
        total += 1;
        let mut gs = got.clone();
        let mut es = expected.clone();
        gs.sort_unstable();
        es.sort_unstable();
        if gs != es {
            failures.push(format!(
                "MEMBERSHIP query {query:?} t={threshold}: got {got:?} expected {expected:?}"
            ));
        } else if got != expected {
            order_only += 1;
        }
    }

    if !failures.is_empty() {
        let shown: Vec<&String> = failures.iter().take(20).collect();
        panic!(
            "{} tfidf membership failures across {total} cases (order-only diffs: {order_only}):\n{}",
            failures.len(),
            shown.iter().map(|s| s.as_str()).collect::<Vec<_>>().join("\n")
        );
    }
}
