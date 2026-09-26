//! Python bindings for the Rust `dedupe` port.
//!
//! Exposes the matching API (`Dedupe`, `RecordLink`, `Gazetteer`, and their
//! `Static*` variants) plus serializer, canonicalisation and convenience
//! helpers.  The Python package wrapper (`python/dedupe_rs/`) adds the
//! variable classes and `console_label`.

mod convert;
mod variables;

use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyListMethods, PyTuple};

use dedupe::api::{
    self, Dedupe as RsDedupe, Gazetteer as RsGazetteer, JoinConstraint,
    RecordLink as RsRecordLink, Settings, StaticDedupe as RsStaticDedupe,
    StaticRecordLink as RsStaticRecordLink,
};
use dedupe::serializer::TrainingData;

use convert::*;

fn err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// `uncertain_pairs` exhaustion is reported as `IndexError`, like dedupe.
fn err_uncertain<E: std::fmt::Display>(e: E) -> PyErr {
    let msg = e.to_string();
    if msg.contains("No more unlabeled") {
        PyIndexError::new_err(msg)
    } else {
        PyValueError::new_err(msg)
    }
}

fn default_cores() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

fn read_file_text(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    obj.call_method0("read")?.extract::<String>()
}

fn write_file_text(obj: &Bound<'_, PyAny>, text: &str) -> PyResult<()> {
    obj.call_method1("write", (text,))?;
    Ok(())
}

fn read_training_file(obj: &Bound<'_, PyAny>) -> PyResult<TrainingData> {
    let text = read_file_text(obj)?;
    let json: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(dedupe::serializer::read_training(&json))
}

fn read_settings_file(obj: &Bound<'_, PyAny>) -> PyResult<Settings> {
    let text = read_file_text(obj)?;
    serde_json::from_str(&text).map_err(|e| PyValueError::new_err(e.to_string()))
}

fn parse_variables(variable_definition: &Bound<'_, PyAny>) -> PyResult<(Vec<dedupe::variables::VariableDef>, bool)> {
    let items: Vec<Bound<'_, PyAny>> = if let Ok(list) = variable_definition.cast::<PyList>() {
        list.iter().collect()
    } else if let Ok(tuple) = variable_definition.cast::<PyTuple>() {
        tuple.iter().collect()
    } else {
        return Err(PyValueError::new_err(
            "variable_definition must be a list of dedupe_rs.variables.* objects",
        ));
    };
    let mut defs = Vec::new();
    let mut has_custom = false;
    for item in items {
        let extracted = variables::variable_from_py(&item)?;
        has_custom |= extracted.is_custom;
        defs.push(extracted.def);
    }
    Ok((defs, has_custom))
}


fn search_results_to_py(
    py: Python<'_>,
    results: &indexmap::IndexMap<dedupe::RecordId, Vec<(dedupe::RecordId, f32)>>,
) -> PyResult<Py<PyAny>> {
    let out = PyDict::new(py);
    for (id, entries) in results {
        let mut prepared: Vec<Py<PyAny>> = Vec::with_capacity(entries.len());
        for (b, score) in entries {
            prepared.push(
                PyTuple::new(
                    py,
                    [
                        record_id_to_py(py, b)?,
                        score.into_pyobject(py)?.into_any().unbind(),
                    ],
                )?
                .into_any()
                .unbind(),
            );
        }
        out.set_item(record_id_to_py(py, id)?, PyTuple::new(py, prepared)?)?;
    }
    Ok(out.into_any().unbind())
}

fn field_names(inner: &api::Matching) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for var in &inner.data_model.field_variables {
        if seen.insert(var.field.clone()) {
            out.push(var.field.clone());
        }
    }
    out
}

fn join_constraint(s: &str) -> PyResult<JoinConstraint> {
    match s {
        "one-to-one" => Ok(JoinConstraint::OneToOne),
        "many-to-one" => Ok(JoinConstraint::ManyToOne),
        "many-to-many" => Ok(JoinConstraint::ManyToMany),
        other => Err(PyValueError::new_err(format!(
            "{other} is an invalid constraint option. Valid options include one-to-one, many-to-one, or many-to-many"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Dedupe
// ---------------------------------------------------------------------------

#[pyclass(unsendable, name = "Dedupe")]
pub struct Dedupe {
    inner: RsDedupe,
}

#[pymethods]
impl Dedupe {
    #[new]
    #[pyo3(signature = (variable_definition, num_cores=None, in_memory=false, seed=0))]
    fn new(
        variable_definition: &Bound<'_, PyAny>,
        num_cores: Option<usize>,
        in_memory: bool,
        seed: u64,
    ) -> PyResult<Self> {
        let (defs, has_custom) = parse_variables(variable_definition)?;
        // Custom comparators call back into Python, so they cannot be used from
        // the parallel scoring pool without risking a GIL deadlock.
        let cores = if has_custom {
            1
        } else {
            num_cores.unwrap_or_else(default_cores)
        };
        let inner = RsDedupe::new(defs, cores, in_memory)
            .map_err(err)?
            .with_seed(seed);
        Ok(Self { inner })
    }

    #[pyo3(signature = (data, training_file=None, sample_size=1500, blocked_proportion=0.9))]
    fn prepare_training(
        &mut self,
        data: &Bound<'_, PyDict>,
        training_file: Option<Bound<'_, PyAny>>,
        sample_size: usize,
        blocked_proportion: f64,
    ) -> PyResult<()> {
        let _ = (sample_size, blocked_proportion);
        if let Some(file) = training_file {
            let td = read_training_file(&file)?;
            self.inner.training_pairs.match_.extend(td.match_);
            self.inner.training_pairs.distinct.extend(td.distinct);
        }
        let data = py_to_data(data)?;
        self.inner.prepare_training(&data).map_err(err)
    }

    fn uncertain_pairs(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let (a, b) = self.inner.uncertain_pairs().map_err(err_uncertain)?;
        Ok(PyList::new(
            py,
            [record_to_py(py, &a)?, record_to_py(py, &b)?],
        )?
        .into_any()
        .unbind())
    }

    fn mark_pairs(&mut self, labeled_pairs: &Bound<'_, PyDict>) -> PyResult<()> {
        let training = py_to_training(labeled_pairs)?;
        self.inner.mark_pairs(&training).map_err(err)
    }

    #[pyo3(signature = (recall=1.0, index_predicates=true))]
    fn train(&mut self, recall: f64, index_predicates: bool) -> PyResult<()> {
        self.inner.train(recall, index_predicates).map_err(err)
    }

    #[pyo3(signature = (data, threshold=0.5))]
    fn partition(
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyDict>,
        threshold: f64,
    ) -> PyResult<Py<PyAny>> {
        let data = py_to_data(data)?;
        let clusters = self.inner.partition(&data, threshold).map_err(err)?;
        clusters_to_py(py, &clusters)
    }

    fn pairs(&self, py: Python<'_>, data: &Bound<'_, PyDict>) -> PyResult<Py<PyAny>> {
        let fp = self.inner.matching.fingerprinter().map_err(err)?;
        let pairs = api::pairs_dedupe(fp, &py_to_data(data)?);
        record_pairs_to_py(py, &pairs)
    }

    fn score(&self, py: Python<'_>, pairs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let pairs = py_to_record_pairs(pairs)?;
        scored_pairs_to_py(py, &self.inner.score(&pairs))
    }

    #[pyo3(signature = (scores, threshold=0.0))]
    fn one_to_one(&self, py: Python<'_>, scores: &Bound<'_, PyAny>, threshold: f64) -> PyResult<Py<PyAny>> {
        let scores: Vec<_> = py_to_scored_pairs(scores)?
            .into_iter()
            .filter(|(_, s)| (*s as f64) > threshold)
            .collect();
        scored_pairs_to_py(py, &dedupe::clustering::greedy_matching(scores))
    }

    #[pyo3(signature = (scores, threshold=0.0))]
    fn many_to_one(&self, py: Python<'_>, scores: &Bound<'_, PyAny>, threshold: f64) -> PyResult<Py<PyAny>> {
        let scores = py_to_scored_pairs(scores)?;
        scored_pairs_to_py(
            py,
            &dedupe::clustering::pair_gazette_matching(scores, threshold, 1),
        )
    }

    #[pyo3(signature = (scores, threshold=0.5))]
    fn cluster(&self, py: Python<'_>, scores: &Bound<'_, PyAny>, threshold: f64) -> PyResult<Py<PyAny>> {
        let scores = py_to_scored_pairs(scores)?;
        let clusters = dedupe::clustering::cluster(&scores, threshold, 30_000);
        clusters_to_py(py, &clusters)
    }

    fn write_training(&self, file_obj: &Bound<'_, PyAny>) -> PyResult<()> {
        let json = dedupe::serializer::write_training(&self.inner.training_pairs);
        write_file_text(
            file_obj,
            &serde_json::to_string(&json).map_err(err)?,
        )
    }

    fn read_training(&mut self, file_obj: &Bound<'_, PyAny>) -> PyResult<()> {
        let td = read_training_file(file_obj)?;
        self.inner.training_pairs.match_.extend(td.match_);
        self.inner.training_pairs.distinct.extend(td.distinct);
        Ok(())
    }

    #[getter]
    fn training_pairs(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        training_to_py(py, &self.inner.training_pairs)
    }

    fn write_settings(&self, file_obj: &Bound<'_, PyAny>) -> PyResult<()> {
        let settings = Settings::from_matching(&self.inner.matching);
        write_file_text(
            file_obj,
            &serde_json::to_string_pretty(&settings).map_err(err)?,
        )
    }

    #[getter]
    fn predicates(&self) -> Vec<String> {
        self.inner
            .matching
            .predicates
            .iter()
            .map(|p| p.0.repr())
            .collect()
    }

    #[getter]
    fn data_model_len(&self) -> usize {
        self.inner.matching.data_model.len()
    }

    #[getter]
    fn field_names(&self) -> Vec<String> {
        field_names(&self.inner.matching)
    }

    fn __repr__(&self) -> String {
        format!(
            "Dedupe(predicates={}, features={})",
            self.inner.matching.predicates.len(),
            self.inner.matching.data_model.len()
        )
    }
}

// ---------------------------------------------------------------------------
// RecordLink
// ---------------------------------------------------------------------------

#[pyclass(unsendable, name = "RecordLink")]
pub struct RecordLink {
    inner: RsRecordLink,
}

#[pymethods]
impl RecordLink {
    #[new]
    #[pyo3(signature = (variable_definition, num_cores=None, in_memory=false, seed=0))]
    fn new(
        variable_definition: &Bound<'_, PyAny>,
        num_cores: Option<usize>,
        in_memory: bool,
        seed: u64,
    ) -> PyResult<Self> {
        let (defs, has_custom) = parse_variables(variable_definition)?;
        let cores = if has_custom {
            1
        } else {
            num_cores.unwrap_or_else(default_cores)
        };
        Ok(Self {
            inner: RsRecordLink::new(defs, cores, in_memory)
                .map_err(err)?
                .with_seed(seed),
        })
    }

    #[pyo3(signature = (data_1, data_2, training_file=None, sample_size=1500, blocked_proportion=0.9))]
    fn prepare_training(
        &mut self,
        data_1: &Bound<'_, PyDict>,
        data_2: &Bound<'_, PyDict>,
        training_file: Option<Bound<'_, PyAny>>,
        sample_size: usize,
        blocked_proportion: f64,
    ) -> PyResult<()> {
        let _ = (sample_size, blocked_proportion);
        if let Some(file) = training_file {
            let td = read_training_file(&file)?;
            self.inner.training_pairs.match_.extend(td.match_);
            self.inner.training_pairs.distinct.extend(td.distinct);
        }
        self.inner
            .prepare_training(&py_to_data(data_1)?, &py_to_data(data_2)?)
            .map_err(err)
    }

    fn uncertain_pairs(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let (a, b) = self.inner.uncertain_pairs().map_err(err_uncertain)?;
        Ok(PyList::new(py, [record_to_py(py, &a)?, record_to_py(py, &b)?])?
            .into_any()
            .unbind())
    }

    fn mark_pairs(&mut self, labeled_pairs: &Bound<'_, PyDict>) -> PyResult<()> {
        self.inner
            .mark_pairs(&py_to_training(labeled_pairs)?)
            .map_err(err)
    }

    #[pyo3(signature = (recall=1.0, index_predicates=true))]
    fn train(&mut self, recall: f64, index_predicates: bool) -> PyResult<()> {
        self.inner.train(recall, index_predicates).map_err(err)
    }

    #[pyo3(signature = (data_1, data_2, threshold=0.5, constraint="one-to-one"))]
    fn join(
        &self,
        py: Python<'_>,
        data_1: &Bound<'_, PyDict>,
        data_2: &Bound<'_, PyDict>,
        threshold: f64,
        constraint: &str,
    ) -> PyResult<Py<PyAny>> {
        let links = self
            .inner
            .join(
                &py_to_data(data_1)?,
                &py_to_data(data_2)?,
                threshold,
                join_constraint(constraint)?,
            )
            .map_err(err)?;
        scored_pairs_to_py(py, &links)
    }

    fn pairs(
        &self,
        py: Python<'_>,
        data_1: &Bound<'_, PyDict>,
        data_2: &Bound<'_, PyDict>,
    ) -> PyResult<Py<PyAny>> {
        let fp = self.inner.matching.fingerprinter().map_err(err)?;
        let pairs = api::pairs_link(fp, &py_to_data(data_1)?, &py_to_data(data_2)?);
        record_pairs_to_py(py, &pairs)
    }

    fn score(&self, py: Python<'_>, pairs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        scored_pairs_to_py(py, &self.inner.matching.score(&py_to_record_pairs(pairs)?))
    }

    fn write_training(&self, file_obj: &Bound<'_, PyAny>) -> PyResult<()> {
        let json = dedupe::serializer::write_training(&self.inner.training_pairs);
        write_file_text(file_obj, &serde_json::to_string(&json).map_err(err)?)
    }

    fn write_settings(&self, file_obj: &Bound<'_, PyAny>) -> PyResult<()> {
        let settings = Settings::from_matching(&self.inner.matching);
        write_file_text(
            file_obj,
            &serde_json::to_string_pretty(&settings).map_err(err)?,
        )
    }

    #[getter]
    fn training_pairs(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        training_to_py(py, &self.inner.training_pairs)
    }

    #[getter]
    fn predicates(&self) -> Vec<String> {
        self.inner
            .matching
            .predicates
            .iter()
            .map(|p| p.0.repr())
            .collect()
    }

    #[getter]
    fn field_names(&self) -> Vec<String> {
        field_names(&self.inner.matching)
    }
}

// ---------------------------------------------------------------------------
// Gazetteer
// ---------------------------------------------------------------------------

#[pyclass(unsendable, name = "Gazetteer")]
pub struct Gazetteer {
    inner: RsGazetteer,
}

#[pymethods]
impl Gazetteer {
    #[new]
    #[pyo3(signature = (variable_definition, num_cores=None, in_memory=false, seed=0))]
    fn new(
        variable_definition: &Bound<'_, PyAny>,
        num_cores: Option<usize>,
        in_memory: bool,
        seed: u64,
    ) -> PyResult<Self> {
        let (defs, has_custom) = parse_variables(variable_definition)?;
        let cores = if has_custom {
            1
        } else {
            num_cores.unwrap_or_else(default_cores)
        };
        Ok(Self {
            inner: RsGazetteer::new(defs, cores, in_memory)
                .map_err(err)?
                .with_seed(seed),
        })
    }

    #[pyo3(signature = (data_1, data_2, training_file=None, sample_size=1500, blocked_proportion=0.9))]
    fn prepare_training(
        &mut self,
        data_1: &Bound<'_, PyDict>,
        data_2: &Bound<'_, PyDict>,
        training_file: Option<Bound<'_, PyAny>>,
        sample_size: usize,
        blocked_proportion: f64,
    ) -> PyResult<()> {
        let _ = (sample_size, blocked_proportion);
        if let Some(file) = training_file {
            let td = read_training_file(&file)?;
            self.inner.training_pairs.match_.extend(td.match_);
            self.inner.training_pairs.distinct.extend(td.distinct);
        }
        self.inner
            .prepare_training(&py_to_data(data_1)?, &py_to_data(data_2)?)
            .map_err(err)
    }

    fn uncertain_pairs(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let (a, b) = self.inner.uncertain_pairs().map_err(err_uncertain)?;
        Ok(PyList::new(py, [record_to_py(py, &a)?, record_to_py(py, &b)?])?
            .into_any()
            .unbind())
    }

    fn mark_pairs(&mut self, labeled_pairs: &Bound<'_, PyDict>) -> PyResult<()> {
        self.inner
            .mark_pairs(&py_to_training(labeled_pairs)?)
            .map_err(err)
    }

    #[pyo3(signature = (recall=1.0, index_predicates=true))]
    fn train(&mut self, recall: f64, index_predicates: bool) -> PyResult<()> {
        self.inner.train(recall, index_predicates).map_err(err)
    }

    fn index(&mut self, data: &Bound<'_, PyDict>) -> PyResult<()> {
        self.inner.index(&py_to_data(data)?).map_err(err)
    }

    fn unindex(&mut self, data: &Bound<'_, PyDict>) -> PyResult<()> {
        self.inner.unindex(&py_to_data(data)?);
        Ok(())
    }

    #[pyo3(signature = (data, threshold=0.0, n_matches=1))]
    fn search(
        &self,
        py: Python<'_>,
        data: &Bound<'_, PyDict>,
        threshold: f64,
        n_matches: usize,
    ) -> PyResult<Py<PyAny>> {
        let results = self
            .inner
            .search(&py_to_data(data)?, threshold, n_matches)
            .map_err(err)?;
        search_results_to_py(py, &results)
    }

    #[getter]
    fn indexed_len(&self) -> usize {
        self.inner.indexed_data.len()
    }

    #[getter]
    fn field_names(&self) -> Vec<String> {
        field_names(&self.inner.matching)
    }
}

// ---------------------------------------------------------------------------
// Static matchers
// ---------------------------------------------------------------------------

#[pyclass(unsendable, name = "StaticDedupe")]
pub struct StaticDedupe {
    inner: RsStaticDedupe,
}

#[pymethods]
impl StaticDedupe {
    #[staticmethod]
    #[pyo3(signature = (settings_file, num_cores=None, in_memory=false))]
    fn from_settings(
        settings_file: &Bound<'_, PyAny>,
        num_cores: Option<usize>,
        in_memory: bool,
    ) -> PyResult<Self> {
        let settings = read_settings_file(settings_file)?;
        Ok(Self {
            inner: RsStaticDedupe::from_settings(
                &settings,
                num_cores.unwrap_or_else(default_cores),
                in_memory,
            )
            .map_err(err)?,
        })
    }

    #[pyo3(signature = (data, threshold=0.5))]
    fn partition(
        &self,
        py: Python<'_>,
        data: &Bound<'_, PyDict>,
        threshold: f64,
    ) -> PyResult<Py<PyAny>> {
        let clusters = self
            .inner
            .partition(&py_to_data(data)?, threshold)
            .map_err(err)?;
        clusters_to_py(py, &clusters)
    }

    fn pairs(&self, py: Python<'_>, data: &Bound<'_, PyDict>) -> PyResult<Py<PyAny>> {
        let fp = self.inner.matching.fingerprinter().map_err(err)?;
        record_pairs_to_py(py, &api::pairs_dedupe(fp, &py_to_data(data)?))
    }

    fn score(&self, py: Python<'_>, pairs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        scored_pairs_to_py(py, &self.inner.matching.score(&py_to_record_pairs(pairs)?))
    }
}

#[pyclass(unsendable, name = "StaticRecordLink")]
pub struct StaticRecordLink {
    inner: RsStaticRecordLink,
}

#[pymethods]
impl StaticRecordLink {
    #[staticmethod]
    #[pyo3(signature = (settings_file, num_cores=None, in_memory=false))]
    fn from_settings(
        settings_file: &Bound<'_, PyAny>,
        num_cores: Option<usize>,
        in_memory: bool,
    ) -> PyResult<Self> {
        let settings = read_settings_file(settings_file)?;
        Ok(Self {
            inner: RsStaticRecordLink::from_settings(
                &settings,
                num_cores.unwrap_or_else(default_cores),
                in_memory,
            )
            .map_err(err)?,
        })
    }

    #[pyo3(signature = (data_1, data_2, threshold=0.5, constraint="one-to-one"))]
    fn join(
        &self,
        py: Python<'_>,
        data_1: &Bound<'_, PyDict>,
        data_2: &Bound<'_, PyDict>,
        threshold: f64,
        constraint: &str,
    ) -> PyResult<Py<PyAny>> {
        let links = self
            .inner
            .join(
                &py_to_data(data_1)?,
                &py_to_data(data_2)?,
                threshold,
                join_constraint(constraint)?,
            )
            .map_err(err)?;
        scored_pairs_to_py(py, &links)
    }
}

#[pyclass(unsendable, name = "StaticGazetteer")]
pub struct StaticGazetteer {
    inner: RsGazetteer,
}

#[pymethods]
impl StaticGazetteer {
    #[staticmethod]
    #[pyo3(signature = (settings_file, num_cores=None, in_memory=false))]
    fn from_settings(
        settings_file: &Bound<'_, PyAny>,
        num_cores: Option<usize>,
        in_memory: bool,
    ) -> PyResult<Self> {
        let settings = read_settings_file(settings_file)?;
        let matching = settings
            .to_matching(num_cores.unwrap_or_else(default_cores), in_memory)
            .map_err(err)?;
        Ok(Self {
            inner: RsGazetteer::from_matching(matching),
        })
    }

    fn index(&mut self, data: &Bound<'_, PyDict>) -> PyResult<()> {
        self.inner.index(&py_to_data(data)?).map_err(err)
    }

    fn unindex(&mut self, data: &Bound<'_, PyDict>) -> PyResult<()> {
        self.inner.unindex(&py_to_data(data)?);
        Ok(())
    }

    #[pyo3(signature = (data, threshold=0.0, n_matches=1))]
    fn search(
        &self,
        py: Python<'_>,
        data: &Bound<'_, PyDict>,
        threshold: f64,
        n_matches: usize,
    ) -> PyResult<Py<PyAny>> {
        let results = self
            .inner
            .search(&py_to_data(data)?, threshold, n_matches)
            .map_err(err)?;
        search_results_to_py(py, &results)
    }
}

// ---------------------------------------------------------------------------
// Module-level helpers
// ---------------------------------------------------------------------------

#[pyfunction]
fn canonicalize(py: Python<'_>, record_cluster: &Bound<'_, PyList>) -> PyResult<Py<PyAny>> {
    let mut records = Vec::with_capacity(record_cluster.len());
    for item in record_cluster.iter() {
        records.push(py_to_record(&item.cast_into::<PyDict>()?)?);
    }
    let rep = dedupe::canonical::get_canonical_rep(&records);
    record_to_py(py, &rep)
}

#[pyfunction]
#[pyo3(signature = (labeled_pairs, file_obj))]
fn write_training(labeled_pairs: &Bound<'_, PyDict>, file_obj: &Bound<'_, PyAny>) -> PyResult<()> {
    let training = py_to_training(labeled_pairs)?;
    let json = dedupe::serializer::write_training(&training);
    write_file_text(file_obj, &serde_json::to_string(&json).map_err(err)?)
}

#[pyfunction]
fn read_training(py: Python<'_>, file_obj: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let training = read_training_file(file_obj)?;
    training_to_py(py, &training)
}

#[pyfunction]
#[pyo3(signature = (data_1, data_2, common_key, training_size=50000))]
fn training_data_link(
    py: Python<'_>,
    data_1: &Bound<'_, PyDict>,
    data_2: &Bound<'_, PyDict>,
    common_key: &str,
    training_size: usize,
) -> PyResult<Py<PyAny>> {
    let training = dedupe::convenience::training_data_link(
        &py_to_data(data_1)?,
        &py_to_data(data_2)?,
        common_key,
        training_size,
    );
    training_to_py(py, &training)
}

#[pyfunction]
#[pyo3(signature = (data, common_key, training_size=50000))]
fn training_data_dedupe(
    py: Python<'_>,
    data: &Bound<'_, PyDict>,
    common_key: &str,
    training_size: usize,
) -> PyResult<Py<PyAny>> {
    let training = dedupe::convenience::training_data_dedupe(
        &py_to_data(data)?,
        common_key,
        training_size,
    );
    training_to_py(py, &training)
}

#[pyfunction]
fn settings_to_dict(py: Python<'_>, settings: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let settings = read_settings_file(settings)?;
    let json = serde_json::to_value(&settings).map_err(err)?;
    pythonize_json(py, &json)
}

fn pythonize_json(py: Python<'_>, value: &serde_json::Value) -> PyResult<Py<PyAny>> {
    Ok(match value {
        serde_json::Value::Null => py.None(),
        serde_json::Value::Bool(b) => (*b).into_pyobject(py)?.to_owned().into_any().unbind(),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into_pyobject(py)?.into_any().unbind()
            } else {
                n.as_f64()
                    .unwrap_or(0.0)
                    .into_pyobject(py)?
                    .into_any()
                    .unbind()
            }
        }
        serde_json::Value::String(s) => s.into_pyobject(py)?.into_any().unbind(),
        serde_json::Value::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                out.push(pythonize_json(py, item)?);
            }
            PyList::new(py, out)?.into_any().unbind()
        }
        serde_json::Value::Object(map) => {
            let dict = PyDict::new(py);
            let mut sorted: Vec<_> = map.iter().collect();
            sorted.sort_by_key(|(k, _)| *k);
            for (k, v) in sorted {
                dict.set_item(k, pythonize_json(py, v)?)?;
            }
            dict.into_any().unbind()
        }
    })
}

/// Module version.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(canonicalize, m)?)?;
    m.add_function(wrap_pyfunction!(write_training, m)?)?;
    m.add_function(wrap_pyfunction!(read_training, m)?)?;
    m.add_function(wrap_pyfunction!(training_data_link, m)?)?;
    m.add_function(wrap_pyfunction!(training_data_dedupe, m)?)?;
    m.add_function(wrap_pyfunction!(settings_to_dict, m)?)?;

    m.add_class::<Dedupe>()?;
    m.add_class::<RecordLink>()?;
    m.add_class::<Gazetteer>()?;
    m.add_class::<StaticDedupe>()?;
    m.add_class::<StaticRecordLink>()?;
    m.add_class::<StaticGazetteer>()?;

    Ok(())
}
