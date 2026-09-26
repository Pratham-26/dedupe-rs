//! Build `dedupe` variable definitions from Python variable objects.
//!
//! The Python-side variable classes (see `python/dedupe_rs/variables.py`) are
//! plain dataclasses exposing a `to_config()` dict; this reads it back into a
//! `VariableDef`, wiring up `Custom` comparators as Python callbacks.

use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};

use dedupe::value::Value;
use dedupe::variables::{self, InteractionDef, VariableDef};

use crate::convert::{py_to_value, value_to_py};

fn opt_string(d: &Bound<'_, PyDict>, k: &str) -> PyResult<Option<String>> {
    match d.get_item(k)? {
        Some(v) if !v.is_none() => Ok(Some(v.extract()?)),
        _ => Ok(None),
    }
}

fn opt_bool(d: &Bound<'_, PyDict>, k: &str) -> PyResult<Option<bool>> {
    match d.get_item(k)? {
        Some(v) if !v.is_none() => Ok(Some(v.extract()?)),
        _ => Ok(None),
    }
}

fn opt_strings(d: &Bound<'_, PyDict>, k: &str) -> PyResult<Vec<String>> {
    match d.get_item(k)? {
        Some(v) if !v.is_none() => Ok(v.extract()?),
        _ => Ok(Vec::new()),
    }
}

/// A variable extracted from Python, plus whether it needs the GIL at call time.
pub struct ExtractedVariable {
    pub def: VariableDef,
    pub is_custom: bool,
}

/// Wrap a Python comparator callable for use by the Rust comparator pipeline.
fn custom_comparator(func: Py<PyAny>) -> Arc<dyn Fn(&Value, &Value) -> f32 + Send + Sync> {
    Arc::new(move |a: &Value, b: &Value| -> f32 {
        Python::attach(|py| {
            let pa = match value_to_py(py, a) {
                Ok(v) => v,
                Err(_) => return 0.0,
            };
            let pb = match value_to_py(py, b) {
                Ok(v) => v,
                Err(_) => return 0.0,
            };
            match func.call1(py, (pa, pb)) {
                Ok(res) => res.extract::<f32>(py).unwrap_or(0.0),
                Err(_) => 0.0,
            }
        })
    })
}

/// Convert a Python variable object into a [`VariableDef`].
pub fn variable_from_py(obj: &Bound<'_, PyAny>) -> PyResult<ExtractedVariable> {
    let cfg_obj = obj.call_method0("to_config").map_err(|_| {
        PyValueError::new_err(
            "variable definitions must be dedupe_rs.variables.* instances (or objects exposing to_config())",
        )
    })?;
    let cfg = cfg_obj.cast::<PyDict>()?;

    let kind: String = opt_string(cfg, "kind")?.unwrap_or_default();
    let field: String = opt_string(cfg, "field")?.unwrap_or_default();
    let name: Option<String> = opt_string(cfg, "name")?;
    let has_missing: bool = opt_bool(cfg, "has_missing")?.unwrap_or(false);
    let crf: bool = opt_bool(cfg, "crf")?.unwrap_or(false);
    let categories: Vec<String> = opt_strings(cfg, "categories")?;
    let interactions: Vec<String> = opt_strings(cfg, "interactions")?;

    let corpus: Vec<Value> = match cfg.get_item("corpus")? {
        Some(v) if !v.is_none() => {
            let list = v.cast::<PyList>()?;
            let mut out = Vec::with_capacity(list.len());
            for item in list.iter() {
                out.push(py_to_value(&item)?);
            }
            out
        }
        _ => Vec::new(),
    };

    if kind == "Interaction" {
        let refs: Vec<&str> = interactions.iter().map(|s| s.as_str()).collect();
        return Ok(ExtractedVariable {
            def: VariableDef::Interaction(InteractionDef::new(&refs)),
            is_custom: false,
        });
    }

    let is_custom = kind == "Custom";
    let mut var = match kind.as_str() {
        "String" => variables::string(&field, crf),
        "ShortString" => variables::short_string(&field, crf),
        "Text" => variables::text(&field, &corpus),
        "Exact" => variables::exact(&field),
        "Categorical" => variables::categorical(&field, &categories),
        "Exists" => variables::exists(&field),
        "Price" => variables::price(&field),
        "LatLong" => variables::latlong(&field),
        "Set" => variables::set(&field, &corpus),
        "Custom" => {
            let comparator = cfg.get_item("comparator")?.ok_or_else(|| {
                PyValueError::new_err("Custom variables require a `comparator` callable")
            })?;
            let label: String = opt_string(cfg, "comparator_label")?
                .unwrap_or_else(|| "custom".to_string());
            let func = custom_comparator(comparator.unbind());
            variables::custom(&field, &label, func)
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown variable kind '{other}'"
            )))
        }
    };

    if let Some(n) = name {
        var = variables::with_name(var, &n);
    }
    if has_missing {
        var = variables::with_missing(var);
    }

    Ok(ExtractedVariable {
        def: VariableDef::Field(var),
        is_custom,
    })
}
