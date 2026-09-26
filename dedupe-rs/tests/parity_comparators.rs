//! Cross-language parity for field comparators.
mod common;

use common::{approx_eq, decode, load_golden};
use dedupe::comparators::{
    CategoricalTypeComparator, Comparator, CosineComparator, CosineKind, ExactComparator,
    ExistsComparator, LatLongComparator, PriceComparator, StringComparator,
    StringComparatorKind,
};
use dedupe::Value;

fn make_case_comparator(name: &str, obj: &serde_json::Map<String, serde_json::Value>) -> Box<dyn Comparator> {
    match name {
        "affinegap" => Box::new(StringComparator {
            kind: StringComparatorKind::AffineGap,
        }),
        "crf" => Box::new(StringComparator {
            kind: StringComparatorKind::Crf,
        }),
        "categorical" => Box::new(CategoricalTypeComparator {
            inner: dedupe::comparators::CategoricalComparator::new(&[
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
            ]),
        }),
        "price" => Box::new(PriceComparator),
        "latlong" => Box::new(LatLongComparator),
        "exact" => Box::new(ExactComparator),
        "exists" => Box::new(ExistsComparator::new()),
        "cosine_text" => {
            let corpus: Vec<Value> = obj["corpus"]
                .as_array()
                .unwrap()
                .iter()
                .map(decode)
                .collect();
            Box::new(CosineComparator::new(CosineKind::Text, &corpus))
        }
        "cosine_set" => {
            let corpus: Vec<Value> = obj["corpus"]
                .as_array()
                .unwrap()
                .iter()
                .map(decode)
                .collect();
            Box::new(CosineComparator::new(CosineKind::Set, &corpus))
        }
        other => panic!("unknown comparator {other}"),
    }
}

#[test]
fn comparators_match_python() {
    let cases = load_golden("comparators.json");
    let mut failures = 0usize;
    let mut first: Vec<String> = Vec::new();
    let mut max_diff: Vec<(String, f64)> = Vec::new();
    let mut total = 0usize;

    for case in &cases {
        let obj = case.as_object().unwrap();
        let name = obj["cmp"].as_str().unwrap();
        let a = decode(&obj["a"]);
        let b = decode(&obj["b"]);
        let expected: Vec<f64> = obj["expected"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap_or(f64::NAN))
            .collect();

        let cmp = make_case_comparator(name, obj);
        let got = cmp.compare(&a, &b);
        total += 1;
        let ok = got.len() == expected.len()
            && got
                .iter()
                .zip(expected.iter())
                .all(|(g, e)| approx_eq(*g as f64, *e, 1e-6));
        if ok {
            let d = got
                .iter()
                .zip(expected.iter())
                .map(|(g, e)| (*g as f64 - *e).abs())
                .fold(0.0f64, f64::max);
            max_diff.push((name.to_string(), d));
        } else {
            failures += 1;
            if first.len() < 25 {
                first.push(format!(
                    "{name}({:?}, {:?}) => got {:?} expected {:?}",
                    a, b, got, expected
                ));
            }
        }
    }

    let worst = max_diff
        .iter()
        .fold(0.0f64, |acc, (_, d)| acc.max(*d));
    eprintln!("comparator parity: {total} cases, {failures} failures, worst diff {worst:e}");
    if failures > 0 {
        panic!(
            "{failures}/{total} comparator cases failed:\n{}",
            first.join("\n")
        );
    }
}
