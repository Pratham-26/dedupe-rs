//! Shared helpers for parity integration tests.
#![allow(dead_code)]

use std::path::PathBuf;

use dedupe::value::{Record, RecordId, Value};
use dedupe::variables::{self, InteractionDef, VariableDef};
use serde_json::Value as J;

/// Decode the type-tagged JSON representation produced by `parity/gen_vectors.py`.
pub fn decode(j: &J) -> Value {
    let obj = j.as_object().expect("tagged value object");
    let t = obj.get("t").and_then(|v| v.as_str()).expect("type tag");
    match t {
        "n" => Value::Null,
        "b" => Value::Bool(obj.get("v").unwrap().as_bool().unwrap()),
        "s" => Value::Str(obj.get("v").unwrap().as_str().unwrap().to_string()),
        "i" => Value::Int(obj.get("v").unwrap().as_i64().unwrap()),
        "f" => Value::Float(obj.get("v").unwrap().as_f64().unwrap()),
        "l" => Value::List(obj.get("v").unwrap().as_array().unwrap().iter().map(decode).collect()),
        "tup" => Value::Tuple(obj.get("v").unwrap().as_array().unwrap().iter().map(decode).collect()),
        "set" => Value::Set(obj.get("v").unwrap().as_array().unwrap().iter().map(decode).collect()),
        other => panic!("unknown type tag {other}"),
    }
}

/// Decode a record (`{"t":"d", ...}` or a plain object of tagged values).
pub fn decode_record(j: &J) -> Record {
    let obj = j.as_object().expect("record object");
    let mut rec = Record::new();
    if let Some(inner) = obj.get("v").and_then(|v| v.as_object()) {
        if obj.get("t").and_then(|t| t.as_str()) == Some("d") {
            for (k, v) in inner {
                rec.insert(k.clone(), decode(v));
            }
            return rec;
        }
    }
    for (k, v) in obj {
        rec.insert(k.clone(), if v.get("t").is_some() { decode(v) } else { decode_plain(v) });
    }
    rec
}

/// Decode a plain (untagged) JSON value.
pub fn decode_plain(j: &J) -> Value {
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
        J::Array(items) => Value::List(items.iter().map(decode_plain).collect()),
        J::Object(map) => {
            let mut rec = Record::new();
            for (k, v) in map {
                rec.insert(k.clone(), decode_plain(v));
            }
            Value::List(vec![])
        }
    }
}

/// Build a [`VariableDef`] list from a Python-style config list.
pub fn build_vars(defs: &[J]) -> Vec<VariableDef> {
    let mut out = Vec::new();
    for d in defs {
        let obj = d.as_object().unwrap();
        let t = obj["type"].as_str().unwrap();
        let field = obj.get("field").and_then(|v| v.as_str());
        let has_missing = obj.get("has_missing").and_then(|v| v.as_bool()).unwrap_or(false);
        let name = obj.get("name").and_then(|v| v.as_str());
        let str_list = |k: &str| -> Vec<String> {
            obj.get(k)
                .and_then(|v| v.as_array())
                .map(|a| a.iter().map(|x| x.as_str().unwrap().to_string()).collect())
                .unwrap_or_default()
        };
        let value_list = |k: &str| -> Vec<Value> {
            obj.get(k)
                .and_then(|v| v.as_array())
                .map(|a| a.iter().map(decode).collect())
                .unwrap_or_default()
        };
        let mut var = match t {
            "String" => variables::string(field.unwrap(), false),
            "ShortString" => variables::short_string(field.unwrap(), false),
            "Text" => variables::text(field.unwrap(), &value_list("corpus")),
            "Exact" => variables::exact(field.unwrap()),
            "Categorical" => variables::categorical(field.unwrap(), &str_list("categories")),
            "Exists" => variables::exists(field.unwrap()),
            "Price" => variables::price(field.unwrap()),
            "LatLong" => variables::latlong(field.unwrap()),
            "Set" => variables::set(field.unwrap(), &value_list("corpus")),
            "Interaction" => {
                let refs = str_list("interactions");
                let refs: Vec<&str> = refs.iter().map(|s| s.as_str()).collect();
                out.push(VariableDef::Interaction(InteractionDef::new(&refs)));
                continue;
            }
            other => panic!("unknown variable type {other}"),
        };
        if let Some(n) = name {
            var = variables::with_name(var, n);
        }
        if has_missing {
            var = variables::with_missing(var);
        }
        out.push(VariableDef::Field(var));
    }
    out
}

/// Parse the record-id key used in golden files (always an integer here).
pub fn parse_record_id(k: &str) -> RecordId {
    match k.parse::<i64>() {
        Ok(i) => RecordId::Int(i),
        Err(_) => RecordId::Str(k.to_string()),
    }
}

pub fn golden_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("golden")
        .join(name)
}

pub fn load_golden_value(name: &str) -> J {
    let text = std::fs::read_to_string(golden_path(name))
        .unwrap_or_else(|e| panic!("read golden {name}: {e}"));
    serde_json::from_str(&text).expect("parse golden json")
}

pub fn load_golden(name: &str) -> Vec<J> {
    let text = std::fs::read_to_string(golden_path(name))
        .unwrap_or_else(|e| panic!("read golden {name}: {e}"));
    serde_json::from_str(&text).expect("parse golden json")
}

pub fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
}
