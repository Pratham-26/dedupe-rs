//! Blocking predicates, ported from `dedupe/predicates.py`.

use std::cell::{Cell, RefCell};
use std::cmp::Ordering;
use std::collections::BTreeSet;

use rustc_hash::FxHashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

use crate::cpredicates::ngrams;
use crate::index::{Doc, LevenshteinIndex, SharedIndex, TfIdfIndex};
use crate::predicate_functions::{self as pf, Keys};
use crate::value::{py_float_repr, Record, Value};

/// Raised when an index predicate is used without indexing records.
#[derive(Debug, Clone)]
pub struct NoIndexError {
    pub message: String,
    pub failing_record: Option<Record>,
}

impl fmt::Display for NoIndexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for NoIndexError {}

/// A blocking predicate.
pub trait Predicate {
    fn type_name(&self) -> &'static str;
    /// The Python `__repr__`, used for equality and hashing.
    fn repr(&self) -> String;
    fn field(&self) -> Option<&str> {
        None
    }
    fn call(&self, record: &Record, target: bool) -> Result<Keys, NoIndexError>;
    fn len(&self) -> usize {
        1
    }
    fn is_empty(&self) -> bool {
        false
    }
    fn is_index(&self) -> bool {
        false
    }
    fn is_canopy(&self) -> bool {
        false
    }
    fn cover_count(&self) -> usize;
    fn set_cover_count(&self, n: usize);
    /// Sub-predicates for compound predicates (used by the fingerprinter).
    fn compound_parts(&self) -> Option<&[PredRef]> {
        None
    }
    fn reset(&self) {}
    fn bust_cache(&self) {}
    fn init_index(&self, _threshold: f64) -> SharedIndex {
        Rc::new(RefCell::new(Box::new(TfIdfIndex::new())))
    }
    /// Preprocess a raw column value into an index document.
    fn preprocess_doc(&self, _value: &Value) -> Option<Doc> {
        None
    }
    fn threshold(&self) -> f64 {
        0.0
    }
    fn freeze(&self, _records_1: &[Record], _records_2: Option<&[Record]>) {}
    /// Set the index used by this predicate (called by the fingerprinter).
    fn set_index(&self, _index: SharedIndex) {}
    fn get_index(&self) -> Option<SharedIndex> {
        None
    }
    /// Serialize this predicate to a portable configuration.
    fn to_config(&self) -> PredicateConfig {
        PredicateConfig::Simple {
            func: "wholeFieldPredicate".to_string(),
            field: self.field().unwrap_or_default().to_string(),
            string_mode: false,
        }
    }
}

/// A portable predicate description, used by the settings file format.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub enum PredicateConfig {
    Simple {
        func: String,
        field: String,
        string_mode: bool,
    },
    Exists {
        field: String,
    },
    Index {
        type_name: String,
        threshold: f64,
        threshold_repr: String,
        field: String,
    },
    Compound {
        parts: Vec<PredicateConfig>,
    },
}

impl PredRef {
    pub fn from_config(config: &PredicateConfig) -> Option<PredRef> {
        match config {
            PredicateConfig::Simple {
                func,
                field,
                string_mode,
            } => Some(PredRef::new(SimplePredicate::new(func, field, *string_mode))),
            PredicateConfig::Exists { field } => Some(PredRef::new(ExistsPredicate::new(field))),
            PredicateConfig::Index {
                type_name,
                threshold,
                threshold_repr,
                field,
            } => IndexPredicate::from_type_name(
                type_name,
                *threshold,
                threshold_repr.clone(),
                field,
            )
            .map(PredRef::new),
            PredicateConfig::Compound { parts } => {
                let mut refs = Vec::new();
                for part in parts {
                    refs.push(PredRef::from_config(part)?);
                }
                Some(PredRef::new(CompoundPredicate::new(refs)))
            }
        }
    }

    pub fn to_config(&self) -> PredicateConfig {
        self.0.to_config()
    }
}

/// A reference-counted predicate with value equality by `repr`.
#[derive(Clone)]
pub struct PredRef(pub Rc<dyn Predicate>);

impl PredRef {
    pub fn new<P: Predicate + 'static>(p: P) -> Self {
        PredRef(Rc::new(p))
    }

    pub fn call(&self, record: &Record, target: bool) -> Result<Keys, NoIndexError> {
        self.0.call(record, target)
    }

    /// Flatten a compound predicate into its atomic parts.
    pub fn flatten(&self) -> Vec<PredRef> {
        match self.0.compound_parts() {
            Some(parts) => parts.to_vec(),
            None => vec![self.clone()],
        }
    }
}

impl fmt::Debug for PredRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.repr())
    }
}

impl PartialEq for PredRef {
    fn eq(&self, other: &Self) -> bool {
        self.0.repr() == other.0.repr()
    }
}
impl Eq for PredRef {}
impl Hash for PredRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.repr().hash(state);
    }
}
impl PartialOrd for PredRef {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for PredRef {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.repr().cmp(&other.0.repr())
    }
}

/// Remove ASCII punctuation, then collapse whitespace (Python
/// `" ".join(strip_punc(s).split())`).
pub fn normalize_string(s: &str) -> String {
    let stripped: String = s
        .chars()
        .filter(|c| !is_ascii_punctuation(*c))
        .collect();
    stripped.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn is_ascii_punctuation(c: char) -> bool {
    matches!(c as u32, 33..=47 | 58..=64 | 91..=96 | 123..=126) && c.is_ascii()
}

fn words(s: &str) -> Vec<String> {
    static RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let re = RE.get_or_init(|| regex::Regex::new(r"[\w']+").unwrap());
    re.find_iter(s).map(|m| m.as_str().to_string()).collect()
}

// ---------------------------------------------------------------------------
// Simple / String / Exists
// ---------------------------------------------------------------------------

pub struct SimplePredicate {
    pub func: fn(&Value) -> Keys,
    pub func_name: String,
    pub field: String,
    /// When true the column is normalized before the function is applied
    /// (`StringPredicate`).
    pub string_mode: bool,
    cover_count: Cell<usize>,
}

impl SimplePredicate {
    pub fn new(func_name: &str, field: &str, string_mode: bool) -> Self {
        let func = pf::by_name(func_name).unwrap_or_else(|| panic!("unknown predicate fn {func_name}"));
        Self {
            func,
            func_name: func_name.to_string(),
            field: field.to_string(),
            string_mode,
            cover_count: Cell::new(0),
        }
    }
}

impl Predicate for SimplePredicate {
    fn type_name(&self) -> &'static str {
        "SimplePredicate"
    }
    fn repr(&self) -> String {
        format!("SimplePredicate: ({}, {})", self.func_name, self.field)
    }
    fn field(&self) -> Option<&str> {
        Some(&self.field)
    }
    fn call(&self, record: &Record, _target: bool) -> Result<Keys, NoIndexError> {
        let column = record.get(&self.field).cloned().unwrap_or(Value::Null);
        if !column.is_truthy() {
            return Ok(Keys::new());
        }
        if self.string_mode {
            let normalized = normalize_string(column.as_str().unwrap_or(&column.py_str()));
            Ok((self.func)(&Value::Str(normalized)))
        } else {
            Ok((self.func)(&column))
        }
    }
    fn cover_count(&self) -> usize {
        self.cover_count.get()
    }
    fn set_cover_count(&self, n: usize) {
        self.cover_count.set(n);
    }
    fn to_config(&self) -> PredicateConfig {
        PredicateConfig::Simple {
            func: self.func_name.clone(),
            field: self.field.clone(),
            string_mode: self.string_mode,
        }
    }
}

pub struct ExistsPredicate {
    pub field: String,
    cover_count: Cell<usize>,
}

impl ExistsPredicate {
    pub fn new(field: &str) -> Self {
        Self {
            field: field.to_string(),
            cover_count: Cell::new(0),
        }
    }

    pub fn func(column: &Value) -> Keys {
        let mut k = Keys::new();
        k.insert(if column.is_truthy() { "1" } else { "0" }.to_string());
        k
    }
}

impl Predicate for ExistsPredicate {
    fn type_name(&self) -> &'static str {
        "ExistsPredicate"
    }
    fn repr(&self) -> String {
        format!("ExistsPredicate: (Exists, {})", self.field)
    }
    fn field(&self) -> Option<&str> {
        Some(&self.field)
    }
    fn call(&self, record: &Record, _target: bool) -> Result<Keys, NoIndexError> {
        let column = record.get(&self.field).cloned().unwrap_or(Value::Null);
        Ok(Self::func(&column))
    }
    fn cover_count(&self) -> usize {
        self.cover_count.get()
    }
    fn set_cover_count(&self, n: usize) {
        self.cover_count.set(n);
    }
    fn to_config(&self) -> PredicateConfig {
        PredicateConfig::Exists {
            field: self.field.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Index predicates
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexFamily {
    Tfidf,
    Levenshtein,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Preprocess {
    Text,
    Set,
    NGram,
    Levenshtein,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexMode {
    Canopy,
    Search,
}

#[derive(Default)]
struct IndexState {
    index: Option<SharedIndex>,
    /// Search cache keyed by `(column, target)`.
    cache: FxHashMap<(Value, bool), Keys>,
    /// Canopy cache keyed by `column`.
    canopy_cache: FxHashMap<Value, Keys>,
    canopy: FxHashMap<usize, Option<usize>>,
}

pub struct IndexPredicate {
    pub family: IndexFamily,
    pub preprocess: Preprocess,
    pub mode: IndexMode,
    pub threshold: f64,
    pub threshold_repr: String,
    pub type_name: &'static str,
    pub field: String,
    state: RefCell<IndexState>,
    cover_count: Cell<usize>,
}

impl IndexPredicate {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        family: IndexFamily,
        preprocess: Preprocess,
        mode: IndexMode,
        threshold: f64,
        threshold_repr: String,
        type_name: &'static str,
        field: &str,
    ) -> Self {
        Self {
            family,
            preprocess,
            mode,
            threshold,
            threshold_repr,
            type_name,
            field: field.to_string(),
            state: RefCell::new(IndexState::default()),
            cover_count: Cell::new(0),
        }
    }

    /// Build a predicate from its Python class name.
    pub fn from_type_name(type_name: &str, threshold: f64, threshold_repr: String, field: &str) -> Option<Self> {
        let (family, preprocess, mode) = match type_name {
            "TfidfTextCanopyPredicate" => (IndexFamily::Tfidf, Preprocess::Text, IndexMode::Canopy),
            "TfidfTextSearchPredicate" => (IndexFamily::Tfidf, Preprocess::Text, IndexMode::Search),
            "TfidfSetCanopyPredicate" => (IndexFamily::Tfidf, Preprocess::Set, IndexMode::Canopy),
            "TfidfSetSearchPredicate" => (IndexFamily::Tfidf, Preprocess::Set, IndexMode::Search),
            "TfidfNGramCanopyPredicate" => (IndexFamily::Tfidf, Preprocess::NGram, IndexMode::Canopy),
            "TfidfNGramSearchPredicate" => (IndexFamily::Tfidf, Preprocess::NGram, IndexMode::Search),
            "LevenshteinCanopyPredicate" => {
                (IndexFamily::Levenshtein, Preprocess::Levenshtein, IndexMode::Canopy)
            }
            "LevenshteinSearchPredicate" => {
                (IndexFamily::Levenshtein, Preprocess::Levenshtein, IndexMode::Search)
            }
            _ => return None,
        };
        // SAFETY: type_name is one of the literals matched above.
        let static_name: &'static str = match type_name {
            "TfidfTextCanopyPredicate" => "TfidfTextCanopyPredicate",
            "TfidfTextSearchPredicate" => "TfidfTextSearchPredicate",
            "TfidfSetCanopyPredicate" => "TfidfSetCanopyPredicate",
            "TfidfSetSearchPredicate" => "TfidfSetSearchPredicate",
            "TfidfNGramCanopyPredicate" => "TfidfNGramCanopyPredicate",
            "TfidfNGramSearchPredicate" => "TfidfNGramSearchPredicate",
            "LevenshteinCanopyPredicate" => "LevenshteinCanopyPredicate",
            "LevenshteinSearchPredicate" => "LevenshteinSearchPredicate",
            _ => unreachable!(),
        };
        Some(Self::new(
            family,
            preprocess,
            mode,
            threshold,
            threshold_repr,
            static_name,
            field,
        ))
    }

    pub fn threshold_str(&self) -> &str {
        &self.threshold_repr
    }

    fn preprocess_value(&self, column: &Value) -> Doc {
        match self.preprocess {
            Preprocess::Text => Doc::Terms(words(column.as_str().unwrap_or(&column.py_str()))),
            Preprocess::Set => Doc::Terms(
                column
                    .as_seq()
                    .map(|s| s.iter().map(|v| v.py_str()).collect())
                    .unwrap_or_default(),
            ),
            Preprocess::NGram => {
                let normalized = normalize_string(column.as_str().unwrap_or(&column.py_str()));
                let mut grams: Vec<String> = ngrams(&normalized, 2).into_iter().collect();
                grams.sort();
                Doc::Terms(grams)
            }
            Preprocess::Levenshtein => {
                Doc::Text(normalize_string(column.as_str().unwrap_or(&column.py_str())))
            }
        }
    }
}

impl Predicate for IndexPredicate {
    fn type_name(&self) -> &'static str {
        self.type_name
    }
    fn repr(&self) -> String {
        format!("{}: ({}, {})", self.type_name, self.threshold_repr, self.field)
    }
    fn field(&self) -> Option<&str> {
        Some(&self.field)
    }
    fn is_index(&self) -> bool {
        true
    }
    fn is_canopy(&self) -> bool {
        self.mode == IndexMode::Canopy
    }
    fn call(&self, record: &Record, target: bool) -> Result<Keys, NoIndexError> {
        let column = record.get(&self.field).cloned().unwrap_or(Value::Null);
        if !column.is_truthy() {
            return Ok(Keys::new());
        }

        match self.mode {
            IndexMode::Canopy => {
                if let Some(cached) = self.state.borrow().canopy_cache.get(&column) {
                    return Ok(cached.clone());
                }
                let index = self.state.borrow().index.clone().ok_or_else(|| NoIndexError {
                    message: "Attempting to block with an index predicate without indexing records"
                        .to_string(),
                    failing_record: Some(record.clone()),
                })?;
                let doc = self.preprocess_value(&column);
                let doc_id = index.borrow_mut().get_or_create_id(&doc);

                let mut block_key: Option<usize> = None;
                let existing = self.state.borrow().canopy.get(&doc_id).copied();
                if let Some(c) = existing {
                    block_key = c;
                } else {
                    let members = index.borrow().search(&doc, self.threshold);
                    {
                        let mut st = self.state.borrow_mut();
                        for m in &members {
                            st.canopy.entry(*m).or_insert(Some(doc_id));
                        }
                        if members.is_empty() {
                            st.canopy.insert(doc_id, None);
                        } else {
                            st.canopy.insert(doc_id, Some(doc_id));
                            block_key = Some(doc_id);
                        }
                    }
                }

                Ok(match block_key {
                    Some(b) => BTreeSet::from([b.to_string()]),
                    None => Keys::new(),
                })
            }
            IndexMode::Search => {
                let key = (column.clone(), target);
                if let Some(cached) = self.state.borrow().cache.get(&key) {
                    return Ok(cached.clone());
                }
                let index = self.state.borrow().index.clone().ok_or_else(|| NoIndexError {
                    message: "Attempting to block with an index predicate without indexing records"
                        .to_string(),
                    failing_record: Some(record.clone()),
                })?;
                let doc = self.preprocess_value(&column);
                let centers: Vec<usize> = if target {
                    vec![index.borrow_mut().get_or_create_id(&doc)]
                } else {
                    index.borrow().search(&doc, self.threshold)
                };
                let result: Keys = centers.into_iter().map(|c| c.to_string()).collect();
                self.state.borrow_mut().cache.insert(key, result.clone());
                Ok(result)
            }
        }
    }
    fn cover_count(&self) -> usize {
        self.cover_count.get()
    }
    fn set_cover_count(&self, n: usize) {
        self.cover_count.set(n);
    }
    fn reset(&self) {
        let mut st = self.state.borrow_mut();
        st.cache.clear();
        st.canopy_cache.clear();
        st.canopy.clear();
        st.index = None;
    }
    fn bust_cache(&self) {
        let mut st = self.state.borrow_mut();
        st.cache.clear();
        st.canopy_cache.clear();
    }
    fn init_index(&self, _threshold: f64) -> SharedIndex {
        match self.family {
            IndexFamily::Tfidf => Rc::new(RefCell::new(Box::new(TfIdfIndex::new()))),
            IndexFamily::Levenshtein => Rc::new(RefCell::new(Box::new(LevenshteinIndex::new()))),
        }
    }
    fn preprocess_doc(&self, value: &Value) -> Option<Doc> {
        Some(self.preprocess_value(value))
    }
    fn threshold(&self) -> f64 {
        self.threshold
    }
    fn set_index(&self, index: SharedIndex) {
        self.state.borrow_mut().index = Some(index);
        self.bust_cache();
    }
    fn get_index(&self) -> Option<SharedIndex> {
        self.state.borrow().index.clone()
    }
    fn to_config(&self) -> PredicateConfig {
        PredicateConfig::Index {
            type_name: self.type_name.to_string(),
            threshold: self.threshold,
            threshold_repr: self.threshold_repr.clone(),
            field: self.field.clone(),
        }
    }
    fn freeze(&self, records_1: &[Record], records_2: Option<&[Record]>) {
        match self.mode {
            IndexMode::Canopy => {
                let mut cache = FxHashMap::default();
                for record in records_1 {
                    let column = record.get(&self.field).cloned().unwrap_or(Value::Null);
                    if let Ok(keys) = self.call(record, false) {
                        cache.insert(column, keys);
                    }
                }
                let mut st = self.state.borrow_mut();
                st.canopy_cache = cache;
                st.canopy.clear();
                st.index = None;
            }
            IndexMode::Search => {
                let mut cache = FxHashMap::default();
                for record in records_1 {
                    let column = record.get(&self.field).cloned().unwrap_or(Value::Null);
                    if let Ok(keys) = self.call(record, false) {
                        cache.insert((column, false), keys);
                    }
                }
                if let Some(records_2) = records_2 {
                    for record in records_2 {
                        let column = record.get(&self.field).cloned().unwrap_or(Value::Null);
                        if let Ok(keys) = self.call(record, true) {
                            cache.insert((column, true), keys);
                        }
                    }
                }
                let mut st = self.state.borrow_mut();
                st.cache = cache;
                st.index = None;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Compound predicate
// ---------------------------------------------------------------------------

pub struct CompoundPredicate {
    pub parts: Vec<PredRef>,
    cover_count: Cell<usize>,
}

impl CompoundPredicate {
    pub fn new(parts: Vec<PredRef>) -> Self {
        Self {
            parts,
            cover_count: Cell::new(0),
        }
    }
}

impl Predicate for CompoundPredicate {
    fn type_name(&self) -> &'static str {
        "CompoundPredicate"
    }
    fn repr(&self) -> String {
        let inner: Vec<String> = self.parts.iter().map(|p| p.0.repr()).collect();
        format!("CompoundPredicate: [{}]", inner.join(", "))
    }
    fn call(&self, record: &Record, target: bool) -> Result<Keys, NoIndexError> {
        let mut per_predicate: Vec<Vec<String>> = Vec::with_capacity(self.parts.len());
        for p in &self.parts {
            let keys = p.call(record, target)?;
            per_predicate.push(keys.into_iter().collect());
        }
        let mut out = Keys::new();
        let mut indices = vec![0usize; per_predicate.len()];
        if per_predicate.iter().any(|p| p.is_empty()) {
            return Ok(out);
        }
        loop {
            let parts: Vec<String> = per_predicate
                .iter()
                .zip(indices.iter())
                .map(|(p, i)| p[*i].replace(':', "\\:"))
                .collect();
            out.insert(parts.join(":"));
            // increment mixed-radix counter
            let mut k = 0;
            loop {
                if k >= indices.len() {
                    return Ok(out);
                }
                indices[k] += 1;
                if indices[k] < per_predicate[k].len() {
                    break;
                }
                indices[k] = 0;
                k += 1;
            }
        }
    }
    fn len(&self) -> usize {
        self.parts.len()
    }
    fn cover_count(&self) -> usize {
        self.cover_count.get()
    }
    fn set_cover_count(&self, n: usize) {
        self.cover_count.set(n);
    }
    fn compound_parts(&self) -> Option<&[PredRef]> {
        Some(&self.parts)
    }
    fn to_config(&self) -> PredicateConfig {
        PredicateConfig::Compound {
            parts: self.parts.iter().map(|p| p.to_config()).collect(),
        }
    }
}

/// Build the `(type_name, threshold, threshold_repr, field)` tuples for the
/// index predicates of a variable, mirroring `variables.base.indexPredicates`.
pub fn index_predicate_types(
    type_names: &[&'static str],
    thresholds: &[(f64, &str)],
    field: &str,
) -> Vec<PredRef> {
    let mut out = Vec::new();
    for type_name in type_names {
        for (threshold, repr) in thresholds {
            if let Some(p) = IndexPredicate::from_type_name(
                type_name,
                *threshold,
                (*repr).to_string(),
                field,
            ) {
                out.push(PredRef::new(p));
            }
        }
    }
    out
}

/// Format a threshold the way Python's `str()` would.
pub fn threshold_repr(x: f64, is_int: bool) -> String {
    if is_int {
        format!("{}", x as i64)
    } else {
        py_float_repr(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use indexmap::IndexMap;

    fn rec(pairs: &[(&str, Value)]) -> Record {
        let mut r = IndexMap::new();
        for (k, v) in pairs {
            r.insert(k.to_string(), v.clone());
        }
        r
    }

    #[test]
    fn normalize() {
        assert_eq!(normalize_string("fo,18v*1vaad80"), "fo18v1vaad80");
        assert_eq!(normalize_string(" cip   ciop "), "cip ciop");
    }

    #[test]
    fn simple_predicate() {
        let p = SimplePredicate::new("wholeFieldPredicate", "foo", false);
        let r = rec(&[("foo", Value::str("bar"))]);
        assert_eq!(p.call(&r, false).unwrap(), BTreeSet::from(["bar".to_string()]));
        assert_eq!(p.repr(), "SimplePredicate: (wholeFieldPredicate, foo)");
    }

    #[test]
    fn string_predicate_normalizes() {
        let p = SimplePredicate::new("sameSevenCharStartPredicate", "foo", true);
        let a = rec(&[("foo", Value::str("fo,18v*1vaad80"))]);
        let b = rec(&[("foo", Value::str("fo18v1vaad80"))]);
        assert_eq!(p.call(&a, false).unwrap(), p.call(&b, false).unwrap());
    }

    #[test]
    fn exists_predicate() {
        let p = ExistsPredicate::new("foo");
        assert_eq!(
            p.call(&rec(&[("foo", Value::Null)]), false).unwrap(),
            BTreeSet::from(["0".to_string()])
        );
        assert_eq!(
            p.call(&rec(&[("foo", Value::Int(3))]), false).unwrap(),
            BTreeSet::from(["1".to_string()])
        );
    }

    #[test]
    fn compound_escapes_colon() {
        let p1 = PredRef::new(SimplePredicate::new("commonSetElementPredicate", "col_1", false));
        let p2 = PredRef::new(SimplePredicate::new("commonSetElementPredicate", "col_2", false));
        let c = CompoundPredicate::new(vec![p1, p2]);
        let r = rec(&[
            (
                "col_1",
                Value::List(vec![Value::str("foo:"), Value::str("foo")]),
            ),
            (
                "col_2",
                Value::List(vec![Value::str(":bar"), Value::str("bar")]),
            ),
        ]);
        let got = c.call(&r, false).unwrap();
        assert_eq!(
            got,
            BTreeSet::from([
                "foo\\::\\:bar".to_string(),
                "foo\\::bar".to_string(),
                "foo:\\:bar".to_string(),
                "foo:bar".to_string(),
            ])
        );
    }
}
