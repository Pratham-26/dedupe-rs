//! Serialization of labeled training pairs.  Ported from
//! `dedupe/serializer.py`.
//!
//! Python's encoder tags tuples and frozensets so they survive a JSON round
//! trip; this module reproduces that wire format.

use indexmap::IndexMap;
use serde_json::{json, Map, Value as J};

use crate::value::{Record, Value};

/// Labeled training data: matched and distinct record pairs.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TrainingData {
    pub match_: Vec<(Record, Record)>,
    pub distinct: Vec<(Record, Record)>,
}

fn value_to_json(v: &Value) -> J {
    match v {
        Value::Null => J::Null,
        Value::Bool(b) => J::Bool(*b),
        Value::Int(i) => json!(i),
        Value::Float(f) => json!(f),
        Value::Str(s) => J::String(s.clone()),
        Value::List(items) => J::Array(items.iter().map(value_to_json).collect()),
        Value::Tuple(items) => json!({
            "__class__": "tuple",
            "__value__": items.iter().map(value_to_json).collect::<Vec<_>>(),
        }),
        Value::Set(items) => {
            let mut encoded: Vec<J> = items.iter().map(value_to_json).collect();
            // Deterministic ordering for sets.
            encoded.sort_by_key(|a| a.to_string());
            json!({ "__class__": "frozenset", "__value__": encoded })
        }
    }
}

fn json_to_value(j: &J) -> Value {
    match j {
        J::Null => Value::Null,
        J::Bool(b) => Value::Bool(*b),
        J::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Int(i)
            } else {
                Value::Float(n.as_f64().unwrap_or(f64::NAN))
            }
        }
        J::String(s) => Value::Str(s.clone()),
        J::Array(items) => Value::List(items.iter().map(json_to_value).collect()),
        J::Object(map) => {
            if let Some(class) = map.get("__class__").and_then(|c| c.as_str()) {
                let inner = map
                    .get("__value__")
                    .and_then(|v| v.as_array())
                    .cloned()
                    .unwrap_or_default();
                let values: Vec<Value> = inner.iter().map(json_to_value).collect();
                match class {
                    "tuple" => return Value::Tuple(values),
                    "frozenset" => return Value::Set(values),
                    _ => {}
                }
            }
            Value::List(Vec::new())
        }
    }
}

fn record_to_json(record: &Record) -> J {
    let mut map = Map::new();
    for (k, v) in record {
        map.insert(k.clone(), value_to_json(v));
    }
    J::Object(map)
}

fn json_to_record(j: &J) -> Record {
    let mut record: Record = IndexMap::new();
    if let Some(map) = j.as_object() {
        for (k, v) in map {
            record.insert(k.clone(), json_to_value(v));
        }
    }
    record
}

fn pair_to_json(pair: &(Record, Record)) -> J {
    J::Array(vec![record_to_json(&pair.0), record_to_json(&pair.1)])
}

fn json_to_pair(j: &J) -> (Record, Record) {
    let arr = j.as_array().cloned().unwrap_or_default();
    let a = arr.first().map(json_to_record).unwrap_or_default();
    let b = arr.get(1).map(json_to_record).unwrap_or_default();
    (a, b)
}

/// Serialize training data to the Python-compatible JSON format.
pub fn write_training(training: &TrainingData) -> J {
    json!({
        "match": training.match_.iter().map(pair_to_json).collect::<Vec<_>>(),
        "distinct": training.distinct.iter().map(pair_to_json).collect::<Vec<_>>(),
    })
}

/// Parse training data from the Python-compatible JSON format.
pub fn read_training(j: &J) -> TrainingData {
    let pairs = |key: &str| -> Vec<(Record, Record)> {
        j.get(key)
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().map(json_to_pair).collect())
            .unwrap_or_default()
    };
    TrainingData {
        match_: pairs("match"),
        distinct: pairs("distinct"),
    }
}

/// Serialize to a string.
pub fn write_training_string(training: &TrainingData) -> String {
    serde_json::to_string(&write_training(training)).expect("serialize")
}

/// Parse from a string.
pub fn read_training_str(s: &str) -> Result<TrainingData, serde_json::Error> {
    let j: J = serde_json::from_str(s)?;
    Ok(read_training(&j))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let mut r1: Record = IndexMap::new();
        r1.insert(
            "bar".to_string(),
            Value::Set(vec![Value::str("bare")]),
        );
        r1.insert(
            "baz".to_string(),
            Value::Tuple(vec![Value::Int(1), Value::Int(2)]),
        );
        r1.insert(
            "bang".to_string(),
            Value::Tuple(vec![Value::Int(1), Value::Int(2)]),
        );
        r1.insert("foo".to_string(), Value::str("baz"));

        let mut r2: Record = IndexMap::new();
        r2.insert("foo".to_string(), Value::str("baz"));

        let training = TrainingData {
            match_: vec![],
            distinct: vec![(r1.clone(), r2.clone())],
        };

        let s = write_training_string(&training);
        let loaded = read_training_str(&s).unwrap();
        assert_eq!(loaded.distinct.len(), 1);
        assert_eq!(loaded.distinct[0].0, r1);
        assert!(matches!(
            loaded.distinct[0].0.get("bar"),
            Some(Value::Set(_))
        ));
        assert!(matches!(
            loaded.distinct[0].0.get("baz"),
            Some(Value::Tuple(_))
        ));
    }
}
