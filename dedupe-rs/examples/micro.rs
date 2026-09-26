//! Micro-benchmarks for the scoring hot path.
//!
//! `cargo run --release --example micro`

use std::time::Instant;

use dedupe::comparators::{affine_gap_distance, normalized_affine_gap_distance};
use dedupe::datamodel::DataModel;
use dedupe::logistic::{Classifier, LogisticRegression};
use dedupe::value::{Record, Value};
use dedupe::variables::{self, VariableDef};

fn main() {
    let syll = ["ba", "ke", "so", "le", "to", "ma", "for", "ste", "wel", "cro"];
    let mut seed = 0x12345678u64;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as usize
    };
    let n = 200_000;
    let mut names: Vec<(String, String)> = Vec::with_capacity(n);
    for _ in 0..n {
        let mk = |next: &mut dyn FnMut() -> usize| {
            format!(
                "{}{} {}{}",
                syll[next() % syll.len()],
                syll[next() % syll.len()],
                syll[next() % syll.len()],
                syll[next() % syll.len()]
            )
        };
        let a = mk(&mut next);
        let b = mk(&mut next);
        names.push((a, b));
    }
    let iters = 5;

    let t = Instant::now();
    let mut acc = 0.0f64;
    for _ in 0..iters {
        for (a, b) in &names {
            acc += affine_gap_distance(a, b, 1.0, 11.0, 10.0, 7.0, 0.125) as f64;
        }
    }
    println!(
        "affine_gap(size)   : {:8.3}s ({} calls, {:.1}ns/call)",
        t.elapsed().as_secs_f64(),
        iters * n,
        t.elapsed().as_secs_f64() * 1e9 / (iters * n) as f64
    );

    let t = Instant::now();
    let mut acc2 = 0.0f64;
    for _ in 0..iters {
        for (a, b) in &names {
            acc2 += normalized_affine_gap_distance(a, b) as f64;
        }
    }
    println!(
        "normalized         : {:8.3}s ({:.1}ns/call, acc={acc:.0}/{acc2:.0})",
        t.elapsed().as_secs_f64(),
        t.elapsed().as_secs_f64() * 1e9 / (iters * n) as f64
    );

    // distance row + classifier
    let mut records: Vec<(Record, Record)> = Vec::with_capacity(n);
    for (a, b) in &names {
        let mk = |name: &str| {
            let mut r = Record::new();
            r.insert("name".to_string(), Value::str(name));
            r.insert("age".to_string(), Value::str("42"));
            r
        };
        records.push((mk(a), mk(b)));
    }
    let dm = DataModel::new(vec![
        VariableDef::Field(variables::string("name", false)),
        VariableDef::Field(variables::string("age", false)),
    ])
    .unwrap();
    let calc = dm.calculator();
    let width = calc.len();
    let t = Instant::now();
    let mut sink = 0.0f64;
    let mut buf = vec![0f32; width];
    for _ in 0..iters {
        for (a, b) in &records {
            calc.distance_row_into(a, b, &mut buf);
            sink += buf[0] as f64;
        }
    }
    println!(
        "distance_row_into  : {:8.3}s ({:.1}ns/call, sink={sink:.0})",
        t.elapsed().as_secs_f64(),
        t.elapsed().as_secs_f64() * 1e9 / (iters * n) as f64
    );

    let features: Vec<Vec<f64>> = records
        .iter()
        .map(|(a, b)| {
            calc.distance_row(a, b)
                .into_iter()
                .map(|x| x as f64)
                .collect()
        })
        .collect();
    let labels: Vec<usize> = (0..features.len()).map(|i| i % 2).collect();
    let mut lr = LogisticRegression::new(1.0);
    lr.fit(&features, &labels);
    let t = Instant::now();
    let mut s = 0.0f64;
    for _ in 0..iters {
        for (a, b) in &records {
            calc.distance_row_into(a, b, &mut buf);
            s += lr.predict_proba_f32(&buf);
        }
    }
    println!(
        "distance+classifier: {:8.3}s ({:.1}ns/call, s={s:.0})",
        t.elapsed().as_secs_f64(),
        t.elapsed().as_secs_f64() * 1e9 / (iters * n) as f64
    );
}
