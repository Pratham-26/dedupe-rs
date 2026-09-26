//! Canonical representations of record clusters.  Ported from
//! `dedupe/canonical.py`.

use indexmap::IndexMap;

use crate::comparators::normalized_affine_gap_distance;
use crate::value::{Record, Value};

/// A comparator used for canonicalization: takes two strings, returns a distance.
pub type CanonicalComparator<'a> = &'a dyn Fn(&str, &str) -> f64;

/// The default comparator (`affinegap.normalizedAffineGapDistance`).
pub fn affine_comparator(a: &str, b: &str) -> f64 {
    normalized_affine_gap_distance(a, b) as f64
}

/// Return the centroid of a list of attribute variants.
pub fn get_centroid(attribute_variants: &[String], comparator: CanonicalComparator) -> String {
    let n = attribute_variants.len();
    if n == 0 {
        return String::new();
    }
    if n == 1 {
        return attribute_variants[0].clone();
    }

    let mut distance_matrix = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..i {
            let d = comparator(&attribute_variants[i], &attribute_variants[j]);
            distance_matrix[i][j] = d;
            distance_matrix[j][i] = d;
        }
    }

    // Column means (`numpy.mean(0)`).
    let average_distance: Vec<f64> = (0..n)
        .map(|j| distance_matrix.iter().map(|row| row[j]).sum::<f64>() / n as f64)
        .collect();

    let min = average_distance
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let idx = average_distance
        .iter()
        .position(|x| *x == min)
        .unwrap_or(0);
    attribute_variants[idx].clone()
}

/// Construct a canonical representation of a cluster by finding canonical
/// values for each field.
pub fn get_canonical_rep(record_cluster: &[Record]) -> Record {
    let mut canonical_rep: IndexMap<String, Value> = IndexMap::new();
    let Some(first) = record_cluster.first() else {
        return canonical_rep;
    };

    for key in first.keys() {
        let key_values: Vec<String> = record_cluster
            .iter()
            .filter_map(|record| record.get(key))
            .filter(|v| v.is_truthy())
            .map(|v| v.py_str())
            .collect();
        if key_values.is_empty() {
            canonical_rep.insert(key.clone(), Value::Str(String::new()));
        } else {
            let centroid = get_centroid(&key_values, &affine_comparator);
            canonical_rep.insert(key.clone(), Value::Str(centroid));
        }
    }

    canonical_rep
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(pairs: &[(&str, &str)]) -> Record {
        let mut r = Record::new();
        for (k, v) in pairs {
            r.insert(k.to_string(), Value::str(*v));
        }
        r
    }

    #[test]
    fn centroid() {
        let variants: Vec<String> = [
            "mary crane center",
            "mary crane center north",
            "mary crane league - mary crane - west",
            "mary crane league mary crane center (east)",
            "mary crane league mary crane center (north)",
            "mary crane league mary crane center (west)",
            "mary crane league - mary crane - east",
            "mary crane family and day care center",
            "mary crane west",
            "mary crane center east",
            "mary crane league mary crane center (east)",
            "mary crane league mary crane center (north)",
            "mary crane league mary crane center (west)",
            "mary crane league",
            "mary crane",
            "mary crane east 0-3",
            "mary crane north",
            "mary crane north 0-3",
            "mary crane league - mary crane - west",
            "mary crane league - mary crane - north",
            "mary crane league - mary crane - east",
            "mary crane league - mary crane - west",
            "mary crane league - mary crane - north",
            "mary crane league - mary crane - east",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        assert_eq!(get_centroid(&variants, &affine_comparator), "mary crane");
    }

    #[test]
    fn canonical_rep() {
        let records = vec![
            rec(&[("name", "mary crane"), ("address", "123 main st"), ("zip", "12345")]),
            rec(&[("name", "mary crane east"), ("address", "123 main street"), ("zip", "")]),
            rec(&[("name", "mary crane west"), ("address", "123 man st"), ("zip", "")]),
        ];
        let rep = get_canonical_rep(&records);
        assert_eq!(rep["name"], Value::str("mary crane"));
        assert_eq!(rep["address"], Value::str("123 main street"));
        assert_eq!(rep["zip"], Value::str("12345"));

        let rep = get_canonical_rep(&records[0..2]);
        assert_eq!(rep["name"], Value::str("mary crane"));
        assert_eq!(rep["address"], Value::str("123 main st"));
        assert_eq!(rep["zip"], Value::str("12345"));
    }
}
