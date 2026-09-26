//! Conversions between Python objects and the Rust `dedupe` value model.

use pyo3::prelude::*;
use pyo3::types::{
    PyAny, PyBool, PyDict, PyFloat, PyFrozenSet, PyInt, PyList, PyListMethods, PySet,
    PySetMethods, PyString, PyTuple,
};

use dedupe::serializer::TrainingData;
use dedupe::value::{Data, Record, RecordId, Value};

/// Convert a Python object to a [`Value`].
pub fn py_to_value(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_none() {
        return Ok(Value::Null);
    }
    if let Ok(b) = obj.cast::<PyBool>() {
        return Ok(Value::Bool(b.is_true()));
    }
    if let Ok(i) = obj.cast::<PyInt>() {
        if let Ok(v) = i.extract::<i64>() {
            return Ok(Value::Int(v));
        }
        // Fall back to the string form for out-of-range integers.
        return Ok(Value::Str(obj.str()?.to_string()));
    }
    if let Ok(f) = obj.cast::<PyFloat>() {
        return Ok(Value::Float(f.value()));
    }
    if let Ok(s) = obj.cast::<PyString>() {
        return Ok(Value::Str(s.to_str()?.to_string()));
    }
    if let Ok(list) = obj.cast::<PyList>() {
        let mut out = Vec::with_capacity(list.len());
        for item in list.iter() {
            out.push(py_to_value(&item)?);
        }
        return Ok(Value::List(out));
    }
    if let Ok(tuple) = obj.cast::<PyTuple>() {
        let mut out = Vec::with_capacity(tuple.len());
        for item in tuple.iter() {
            out.push(py_to_value(&item)?);
        }
        return Ok(Value::Tuple(out));
    }
    if let Ok(set) = obj.cast::<PySet>() {
        let mut out = Vec::with_capacity(set.len());
        for item in set.iter() {
            out.push(py_to_value(&item)?);
        }
        return Ok(Value::Set(out));
    }
    if let Ok(set) = obj.cast::<PyFrozenSet>() {
        let mut out = Vec::with_capacity(set.len());
        for item in set.iter() {
            out.push(py_to_value(&item)?);
        }
        return Ok(Value::Set(out));
    }
    // Anything else is stringified, matching how dedupe's predicates stringify.
    Ok(Value::Str(obj.str()?.to_string()))
}

/// Convert a [`Value`] to a Python object.
pub fn value_to_py(py: Python<'_>, v: &Value) -> PyResult<Py<PyAny>> {
    Ok(match v {
        Value::Null => py.None(),
        Value::Bool(b) => (*b).into_pyobject(py)?.to_owned().into_any().unbind(),
        Value::Int(i) => i.into_pyobject(py)?.into_any().unbind(),
        Value::Float(f) => f.into_pyobject(py)?.into_any().unbind(),
        Value::Str(s) => s.into_pyobject(py)?.into_any().unbind(),
        Value::List(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(value_to_py(py, item)?)?;
            }
            list.into_any().unbind()
        }
        Value::Tuple(items) => {
            let mut objs = Vec::with_capacity(items.len());
            for item in items {
                objs.push(value_to_py(py, item)?);
            }
            PyTuple::new(py, objs)?.into_any().unbind()
        }
        Value::Set(items) => {
            let set = PySet::empty(py)?;
            for item in items {
                set.add(value_to_py(py, item)?)?;
            }
            set.into_any().unbind()
        }
    })
}

/// Convert a Python record id to a [`RecordId`].
pub fn py_to_record_id(obj: &Bound<'_, PyAny>) -> PyResult<RecordId> {
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(RecordId::Int(i));
    }
    if let Ok(s) = obj.extract::<String>() {
        return Ok(RecordId::Str(s));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "record ids must be int or str",
    ))
}

/// Convert a [`RecordId`] to a Python object.
pub fn record_id_to_py(py: Python<'_>, id: &RecordId) -> PyResult<Py<PyAny>> {
    Ok(match id {
        RecordId::Int(i) => i.into_pyobject(py)?.into_any().unbind(),
        RecordId::Str(s) => s.into_pyobject(py)?.into_any().unbind(),
    })
}

/// Convert a Python `dict` to a [`Record`].
pub fn py_to_record(obj: &Bound<'_, PyDict>) -> PyResult<Record> {
    let mut record = Record::new();
    for (k, v) in obj.iter() {
        record.insert(k.extract::<String>()?, py_to_value(&v)?);
    }
    Ok(record)
}

/// Convert a [`Record`] to a Python `dict`.
pub fn record_to_py(py: Python<'_>, record: &Record) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    for (k, v) in record {
        dict.set_item(k, value_to_py(py, v)?)?;
    }
    Ok(dict.into_any().unbind())
}

/// Convert a Python `dict` of records to [`Data`].
pub fn py_to_data(obj: &Bound<'_, PyDict>) -> PyResult<Data> {
    let mut data = Data::new();
    for (k, v) in obj.iter() {
        let id = py_to_record_id(&k)?;
        let record = v.cast::<PyDict>()?;
        data.insert(id, py_to_record(record)?);
    }
    Ok(data)
}

/// Convert [`Data`] to a Python `dict`.
pub fn data_to_py(py: Python<'_>, data: &Data) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    for (id, record) in data {
        dict.set_item(record_id_to_py(py, id)?, record_to_py(py, record)?)?;
    }
    Ok(dict.into_any().unbind())
}

fn pair_from_py(obj: &Bound<'_, PyAny>) -> PyResult<(Record, Record)> {
    let a = obj.get_item(0)?;
    let b = obj.get_item(1)?;
    Ok((
        py_to_record(&a.cast_into::<PyDict>()?)?,
        py_to_record(&b.cast_into::<PyDict>()?)?,
    ))
}

/// Convert Python `{"match": [...], "distinct": [...]}` to [`TrainingData`].
pub fn py_to_training(obj: &Bound<'_, PyDict>) -> PyResult<TrainingData> {
    let mut training = TrainingData::default();
    for (key, target) in [("match", 0u8), ("distinct", 1u8)] {
        let Some(items) = obj.get_item(key)? else {
            continue;
        };
        for item in items.cast::<PyList>()?.iter() {
            let pair = pair_from_py(&item)?;
            if target == 0 {
                training.match_.push(pair);
            } else {
                training.distinct.push(pair);
            }
        }
    }
    Ok(training)
}

/// Convert [`TrainingData`] to a Python `dict`.
pub fn training_to_py(py: Python<'_>, training: &TrainingData) -> PyResult<Py<PyAny>> {
    let out = PyDict::new(py);
    let mut matches: Vec<Py<PyAny>> = Vec::new();
    for (a, b) in &training.match_ {
        matches.push(
            PyTuple::new(py, [record_to_py(py, a)?, record_to_py(py, b)?])?
                .into_any()
                .unbind(),
        );
    }
    let mut distinct: Vec<Py<PyAny>> = Vec::new();
    for (a, b) in &training.distinct {
        distinct.push(
            PyTuple::new(py, [record_to_py(py, a)?, record_to_py(py, b)?])?
                .into_any()
                .unbind(),
        );
    }
    out.set_item("match", PyList::new(py, matches)?)?;
    out.set_item("distinct", PyList::new(py, distinct)?)?;
    Ok(out.into_any().unbind())
}

/// Convert a Python list of `((id, record), (id, record))` pairs.
pub fn py_to_record_pairs(
    obj: &Bound<'_, PyAny>,
) -> PyResult<Vec<((RecordId, Record), (RecordId, Record))>> {
    let list = obj.cast::<PyList>()?;
    let mut out = Vec::with_capacity(list.len());
    for item in list.iter() {
        let a = item.get_item(0)?;
        let b = item.get_item(1)?;
        let a_id = py_to_record_id(&a.get_item(0)?)?;
        let a_rec = py_to_record(&a.get_item(1)?.cast_into::<PyDict>()?)?;
        let b_id = py_to_record_id(&b.get_item(0)?)?;
        let b_rec = py_to_record(&b.get_item(1)?.cast_into::<PyDict>()?)?;
        out.push(((a_id, a_rec), (b_id, b_rec)));
    }
    Ok(out)
}

/// Convert record pairs to a Python list.
pub fn record_pairs_to_py(
    py: Python<'_>,
    pairs: &[((RecordId, Record), (RecordId, Record))],
) -> PyResult<Py<PyAny>> {
    let mut out: Vec<Py<PyAny>> = Vec::with_capacity(pairs.len());
    for ((a_id, a_rec), (b_id, b_rec)) in pairs {
        let a = PyTuple::new(
            py,
            [record_id_to_py(py, a_id)?, record_to_py(py, a_rec)?],
        )?;
        let b = PyTuple::new(
            py,
            [record_id_to_py(py, b_id)?, record_to_py(py, b_rec)?],
        )?;
        out.push(PyTuple::new(py, [a.into_any().unbind(), b.into_any().unbind()])?.into_any().unbind());
    }
    Ok(PyList::new(py, out)?.into_any().unbind())
}

use dedupe::clustering::ScoredPair;

/// Convert a Python list of `((id, id), score)` to scored pairs.
pub fn py_to_scored_pairs(obj: &Bound<'_, PyAny>) -> PyResult<Vec<ScoredPair>> {
    let list = obj.cast::<PyList>()?;
    let mut out = Vec::with_capacity(list.len());
    for item in list.iter() {
        let ids = item.get_item(0)?;
        let a = py_to_record_id(&ids.get_item(0)?)?;
        let b = py_to_record_id(&ids.get_item(1)?)?;
        let score: f32 = item.get_item(1)?.extract()?;
        out.push(((a, b), score));
    }
    Ok(out)
}

/// Convert scored pairs to a Python list of `((id, id), score)`.
pub fn scored_pairs_to_py(py: Python<'_>, pairs: &[ScoredPair]) -> PyResult<Py<PyAny>> {
    let mut out: Vec<Py<PyAny>> = Vec::with_capacity(pairs.len());
    for ((a, b), score) in pairs {
        let ids = PyTuple::new(py, [record_id_to_py(py, a)?, record_id_to_py(py, b)?])?;
        out.push(
            PyTuple::new(py, [ids.into_any().unbind(), score.into_pyobject(py)?.into_any().unbind()])?
                .into_any()
                .unbind(),
        );
    }
    Ok(PyList::new(py, out)?.into_any().unbind())
}

/// Convert a Python list of clusters `(ids, scores)`.
pub fn clusters_to_py(
    py: Python<'_>,
    clusters: &[dedupe::clustering::Cluster],
) -> PyResult<Py<PyAny>> {
    let mut out: Vec<Py<PyAny>> = Vec::with_capacity(clusters.len());
    for (ids, scores) in clusters {
        let mut id_objs: Vec<Py<PyAny>> = Vec::with_capacity(ids.len());
        for id in ids {
            id_objs.push(record_id_to_py(py, id)?);
        }
        out.push(
            PyTuple::new(
                py,
                [
                    PyTuple::new(py, id_objs)?.into_any().unbind(),
                    PyTuple::new(py, scores.clone())?.into_any().unbind(),
                ],
            )?
            .into_any()
            .unbind(),
        );
    }
    Ok(PyList::new(py, out)?.into_any().unbind())
}
