//! Cross-language parity for the data model (distance matrix layout + values).
mod common;

use common::{approx_eq, build_vars, decode_record, load_golden};
use dedupe::datamodel::DataModel;

#[test]
fn datamodel_matches_python() {
    let cases = load_golden("datamodel.json");
    let mut failures: Vec<String> = Vec::new();
    let mut total = 0usize;

    for case in &cases {
        let meta = &case["meta"];
        let defs = meta["defs"].as_array().unwrap();
        let vars = build_vars(defs);
        let dm = match DataModel::new(vars) {
            Ok(dm) => dm,
            Err(e) => {
                failures.push(format!("DataModel::new failed: {e}"));
                continue;
            }
        };

        total += 1;
        let mut expect = |got: i64, key: &str| {
            if got != meta[key].as_i64().unwrap() {
                failures.push(format!(
                    "config {defs:?}: {key} got {got} expected {}",
                    meta[key]
                ));
            }
        };
        expect(dm.derived_start as i64, "derived_start");
        expect(dm.len() as i64, "len");

        let got_missing: Vec<i64> = dm.missing_field_indices.iter().map(|x| *x as i64).collect();
        let exp_missing: Vec<i64> = meta["missing_field_indices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        if got_missing != exp_missing {
            failures.push(format!("{defs:?}: missing indices {got_missing:?} != {exp_missing:?}"));
        }

        let got_inter: Vec<Vec<i64>> = dm
            .interaction_indices
            .iter()
            .map(|v| v.iter().map(|x| *x as i64).collect())
            .collect();
        let exp_inter: Vec<Vec<i64>> = meta["interaction_indices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect())
            .collect();
        if got_inter != exp_inter {
            failures.push(format!("{defs:?}: interaction indices {got_inter:?} != {exp_inter:?}"));
        }

        // field variables
        let exp_fv: Vec<(String, String, i64)> = meta["field_variables"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| {
                (
                    v["field"].as_str().unwrap().to_string(),
                    v["name"].as_str().unwrap().to_string(),
                    v["len"].as_i64().unwrap(),
                )
            })
            .collect();
        let got_fv: Vec<(String, String, i64)> = dm
            .field_variables
            .iter()
            .map(|v| (v.field.clone(), v.name.clone(), v.len as i64))
            .collect();
        if got_fv != exp_fv {
            failures.push(format!("{defs:?}: field vars {got_fv:?} != {exp_fv:?}"));
        }

        // predicates (as a sorted set of reprs)
        let mut got_preds: Vec<String> = dm.predicates().iter().map(|p| p.0.repr()).collect();
        got_preds.sort();
        let exp_preds: Vec<String> = meta["predicates"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        if got_preds != exp_preds {
            failures.push(format!("{defs:?}: predicates differ"));
        }

        // distances
        let pairs: Vec<(dedupe::Record, dedupe::Record)> = case["pairs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| (decode_record(&p[0]), decode_record(&p[1])))
            .collect();
        let got = dm.distances(&pairs);
        let expected: Vec<Vec<Option<f64>>> = case["distances"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64())
                    .collect()
            })
            .collect();
        if got.len() != expected.len() {
            failures.push(format!("{defs:?}: distance row count {} != {}", got.len(), expected.len()));
            continue;
        }
        for (i, (grow, erow)) in got.iter().zip(expected.iter()).enumerate() {
            if grow.len() != erow.len() {
                failures.push(format!("{defs:?}: row {i} len {} != {}", grow.len(), erow.len()));
                break;
            }
            for (j, (g, e)) in grow.iter().zip(erow.iter()).enumerate() {
                let e = e.unwrap_or(f64::NAN);
                if !approx_eq(*g as f64, e, 1e-5) {
                    failures.push(format!(
                        "{defs:?}: distance[{i}][{j}] got {g} expected {e}"
                    ));
                }
            }
        }
    }

    if !failures.is_empty() {
        let shown: Vec<&String> = failures.iter().take(30).collect();
        panic!(
            "{} datamodel failures across {total} configs:\n{}",
            failures.len(),
            shown.iter().map(|s| s.as_str()).collect::<Vec<_>>().join("\n")
        );
    }
}
