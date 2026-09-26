//! Field comparators, ported from the third-party libraries used by dedupe:
//! `affinegap`, `categorical-distance`, `simplecosine`, `haversine`, plus the
//! inline comparators in `dedupe.variables`.

use std::collections::BTreeSet;

use rustc_hash::FxHashMap;
use std::sync::Arc;

use crate::value::{cmp_values, Value};

/// A comparator maps a pair of field values to a fixed-length distance vector.
pub trait Comparator: Send + Sync + std::fmt::Debug {
    fn name(&self) -> String;
    fn compare(&self, a: &Value, b: &Value) -> Vec<f32>;
    /// Write this comparator's distance vector into `out` without allocating.
    fn compare_into(&self, a: &Value, b: &Value, out: &mut [f32]) {
        let v = self.compare(a, b);
        let n = v.len().min(out.len());
        out[..n].copy_from_slice(&v[..n]);
    }
    /// How many distance columns this comparator produces.
    fn len(&self) -> usize {
        self.compare(&Value::Null, &Value::Null).len().max(1)
    }
    fn is_empty(&self) -> bool {
        false
    }
    /// Whether missing (`None`) values should be passed to the comparator.
    fn handles_missing(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// Affine gap (affinegap 1.12)
// ---------------------------------------------------------------------------

/// Affine gap distance with Monge-Elkan default weights, matching
/// `affinegap.affineGapDistance`.
#[allow(clippy::too_many_arguments)]
pub fn affine_gap_distance(
    a: &str,
    b: &str,
    match_weight: f32,
    mismatch_weight: f32,
    gap_weight: f32,
    space_weight: f32,
    abbreviation_scale: f32,
) -> f32 {
    // Fast path: identical strings (and match weight is the cheapest).
    if a == b && match_weight == match_weight.min(mismatch_weight).min(gap_weight) {
        return match_weight * a.chars().count() as f32;
    }

    let len_a = a.chars().count();
    let len_b = b.chars().count();
    let (long, short) = if len_a < len_b { (b, a) } else { (a, b) };
    let length1 = long.chars().count();

    if long.is_ascii() && short.is_ascii() {
        with_ag_buffers(length1, |d, vc, vp| {
            ag_core(
                long.as_bytes(),
                short.as_bytes(),
                d,
                vc,
                vp,
                match_weight,
                mismatch_weight,
                gap_weight,
                space_weight,
                abbreviation_scale,
            )
        })
    } else {
        let long: Vec<char> = long.chars().collect();
        let short: Vec<char> = short.chars().collect();
        with_ag_buffers(length1, |d, vc, vp| {
            ag_core(
                &long,
                &short,
                d,
                vc,
                vp,
                match_weight,
                mismatch_weight,
                gap_weight,
                space_weight,
                abbreviation_scale,
            )
        })
    }
}

/// Thread-local affine-gap workspace, reused across calls to avoid allocation
/// and initialisation on the hot path.
struct AgScratch {
    d: Vec<f32>,
    v_current: Vec<f32>,
    v_previous: Vec<f32>,
}

thread_local! {
    static AG_SCRATCH: std::cell::RefCell<AgScratch> =
        const { std::cell::RefCell::new(AgScratch {
            d: Vec::new(),
            v_current: Vec::new(),
            v_previous: Vec::new(),
        }) };
}

#[inline]
fn with_ag_buffers<R>(
    length1: usize,
    f: impl FnOnce(&mut [f32], &mut [f32], &mut [f32]) -> R,
) -> R {
    AG_SCRATCH.with(|cell| {
        let mut s = cell.borrow_mut();
        let n = length1 + 1;
        if s.d.len() < n {
            s.d.resize(n, 0.0);
            s.v_current.resize(n, 0.0);
            s.v_previous.resize(n, 0.0);
        }
        let AgScratch {
            d,
            v_current,
            v_previous,
        } = &mut *s;
        f(&mut d[..n], &mut v_current[..n], &mut v_previous[..n])
    })
}

/// The affine-gap recurrence, generic over the symbol type (`u8` for ASCII,
/// `char` otherwise).  `a` must be at least as long as `b`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn ag_core<T: PartialEq + Copy>(
    a: &[T],
    b: &[T],
    d: &mut [f32],
    v_current: &mut [f32],
    v_previous: &mut [f32],
    match_weight: f32,
    mismatch_weight: f32,
    gap_weight: f32,
    space_weight: f32,
    abbreviation_scale: f32,
) -> f32 {
    let length1 = a.len();
    let length2 = b.len();
    // `(float) INT_MAX` in C, i.e. 2^31.
    let inf: f32 = 2147483647i32 as f32;

    v_current[0] = 0.0;
    for j in 1..=length1 {
        v_current[j] = gap_weight + space_weight * j as f32;
        d[j] = inf;
    }

    for i in 1..=length2 {
        let symbol2 = b[i - 1];
        v_previous[..=length1].copy_from_slice(&v_current[..=length1]);
        v_current[0] = gap_weight + space_weight * i as f32;
        let mut insertion = inf;

        // The abbreviation discount only applies once we run past the length
        // of the shorter string; splitting the loop removes the branch.
        for j in 1..=length2 {
            let symbol1 = a[j - 1];
            insertion = insertion.min(v_current[j - 1] + gap_weight) + space_weight;
            let dj = d[j].min(v_previous[j] + gap_weight) + space_weight;
            d[j] = dj;
            let m = v_previous[j - 1]
                + if symbol1 == symbol2 {
                    match_weight
                } else {
                    mismatch_weight
                };
            v_current[j] = insertion.min(dj).min(m);
        }
        for j in (length2 + 1)..=length1 {
            let symbol1 = a[j - 1];
            insertion = insertion
                .min(v_current[j - 1] + gap_weight * abbreviation_scale)
                + space_weight * abbreviation_scale;
            let dj = d[j].min(v_previous[j] + gap_weight) + space_weight;
            d[j] = dj;
            let m = v_previous[j - 1]
                + if symbol1 == symbol2 {
                    match_weight
                } else {
                    mismatch_weight
                };
            v_current[j] = insertion.min(dj).min(m);
        }
    }

    v_current[length1]
}

/// `affinegap.normalizedAffineGapDistance`.
pub fn normalized_affine_gap_distance(a: &str, b: &str) -> f32 {
    let normalizer = (a.chars().count() + b.chars().count()) as f32;
    if normalizer == 0.0 {
        return f32::NAN;
    }
    affine_gap_distance(a, b, 1.0, 11.0, 10.0, 7.0, 0.125) / normalizer
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StringComparatorKind {
    AffineGap,
    Crf,
}

#[derive(Debug)]
pub struct StringComparator {
    pub kind: StringComparatorKind,
}

impl Comparator for StringComparator {
    fn name(&self) -> String {
        match self.kind {
            StringComparatorKind::AffineGap => "affinegap".to_string(),
            StringComparatorKind::Crf => "crf".to_string(),
        }
    }
    fn compare(&self, a: &Value, b: &Value) -> Vec<f32> {
        let mut out = [0f32; 1];
        self.compare_into(a, b, &mut out);
        out.to_vec()
    }
    fn compare_into(&self, a: &Value, b: &Value, out: &mut [f32]) {
        let a = a.as_str().unwrap_or("");
        let b = b.as_str().unwrap_or("");
        out[0] = match self.kind {
            StringComparatorKind::AffineGap => normalized_affine_gap_distance(a, b),
            StringComparatorKind::Crf => crate::comparators::crf::crf_edit_distance(a, b),
        };
    }
}

// ---------------------------------------------------------------------------
// Exact
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct ExactComparator;

impl Comparator for ExactComparator {
    fn name(&self) -> String {
        "exact".to_string()
    }
    fn compare(&self, a: &Value, b: &Value) -> Vec<f32> {
        vec![if a == b { 1.0 } else { 0.0 }]
    }
    fn compare_into(&self, a: &Value, b: &Value, out: &mut [f32]) {
        out[0] = if a == b { 1.0 } else { 0.0 };
    }
}

// ---------------------------------------------------------------------------
// Categorical (categorical-distance)
// ---------------------------------------------------------------------------

/// Port of `categorical.CategoricalComparator`.
#[derive(Debug)]
pub struct CategoricalComparator {
    /// Python `repr` of the dummy variable tuples, e.g. `("('b', 'b')", ...)`.
    pub dummy_names: Vec<String>,
    responses: FxHashMap<(String, String), Vec<f32>>,
    vector_length: usize,
}

fn vector_length(n: usize) -> usize {
    (n + 1) * n / 2 - 1
}

fn response_vector(index: usize, vector_length: usize) -> Vec<f32> {
    let mut r = vec![0.0f32; vector_length];
    if index > 0 {
        r[index - 1] = 1.0;
    }
    r
}

impl CategoricalComparator {
    pub fn new(category_names: &[String]) -> Self {
        let n = category_names.len();
        let vector_length = vector_length(n);

        let mut categories: Vec<(String, String)> = category_names
            .iter()
            .map(|c| (c.clone(), c.clone()))
            .collect();
        for i in 0..n {
            for j in (i + 1)..n {
                categories.push((category_names[i].clone(), category_names[j].clone()));
            }
        }

        let dummy_names: Vec<String> = categories[1..]
            .iter()
            .map(|(a, b)| {
                format!(
                    "({}, {})",
                    Value::Str(a.clone()).py_str(),
                    Value::Str(b.clone()).py_str()
                )
            })
            .collect();

        let mut responses = FxHashMap::default();
        for (i, (a, b)) in categories.iter().enumerate() {
            let response = response_vector(i, vector_length);
            responses.insert((a.clone(), b.clone()), response.clone());
            responses.insert((b.clone(), a.clone()), response);
        }

        Self {
            dummy_names,
            responses,
            vector_length,
        }
    }

    pub fn compare_str(&self, a: &str, b: &str) -> Option<Vec<f32>> {
        self.responses
            .get(&(a.to_string(), b.to_string()))
            .cloned()
    }
}

/// Comparator used by `CategoricalType`.
#[derive(Debug)]
pub struct CategoricalTypeComparator {
    pub inner: CategoricalComparator,
}

impl Comparator for CategoricalTypeComparator {
    fn name(&self) -> String {
        "categorical".to_string()
    }
    fn compare(&self, a: &Value, b: &Value) -> Vec<f32> {
        let sa = a.py_str();
        let sb = b.py_str();
        self.inner
            .compare_str(&sa, &sb)
            .unwrap_or_else(|| vec![0.0; self.inner.vector_length])
    }
    fn len(&self) -> usize {
        self.inner.vector_length
    }
}

// ---------------------------------------------------------------------------
// Exists
// ---------------------------------------------------------------------------

/// Comparator used by `ExistsType`.  Handles missing values.
#[derive(Debug)]
pub struct ExistsComparator {
    inner: CategoricalComparator,
}

impl ExistsComparator {
    pub fn new() -> Self {
        Self {
            inner: CategoricalComparator::new(&["0".to_string(), "1".to_string()]),
        }
    }

    /// Python `repr` of the dummy variable tuples, e.g. `"(1, 1)"`.
    pub fn dummy_names(&self) -> Vec<String> {
        // Exists uses integer categories [0, 1].
        vec!["(1, 1)".to_string(), "(0, 1)".to_string()]
    }
}

impl Default for ExistsComparator {
    fn default() -> Self {
        Self::new()
    }
}

impl Comparator for ExistsComparator {
    fn name(&self) -> String {
        "exists".to_string()
    }
    fn compare(&self, a: &Value, b: &Value) -> Vec<f32> {
        let av = a.is_truthy();
        let bv = b.is_truthy();
        let (x, y) = match (av, bv) {
            (true, true) => ("1", "1"),
            (false, false) => ("0", "0"),
            _ => ("0", "1"),
        };
        self.inner.compare_str(x, y).unwrap_or_default()
    }
    fn len(&self) -> usize {
        self.inner.vector_length
    }
    fn handles_missing(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// Price
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct PriceComparator;

impl Comparator for PriceComparator {
    fn name(&self) -> String {
        "price".to_string()
    }
    fn compare(&self, a: &Value, b: &Value) -> Vec<f32> {
        let mut out = [0f32; 1];
        self.compare_into(a, b, &mut out);
        out.to_vec()
    }
    fn compare_into(&self, a: &Value, b: &Value, out: &mut [f32]) {
        let p1 = a.as_f64().unwrap_or(f64::NAN);
        let p2 = b.as_f64().unwrap_or(f64::NAN);
        out[0] = if p1 <= 0.0 || p2 <= 0.0 {
            f32::NAN
        } else {
            (p1.log10() - p2.log10()).abs() as f32
        };
    }
}

// ---------------------------------------------------------------------------
// LatLong
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct LatLongComparator;

/// Great-circle distance in kilometres, matching the `haversine` package.
pub fn haversine_km(a: (f64, f64), b: (f64, f64)) -> f64 {
    const R: f64 = 6371.0;
    let (lat1, lon1) = a;
    let (lat2, lon2) = b;
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();
    let h = (dlat / 2.0).sin().powi(2)
        + lat1.to_radians().cos() * lat2.to_radians().cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * h.sqrt().asin();
    R * c
}

fn as_latlong(v: &Value) -> Option<(f64, f64)> {
    let s = v.as_seq()?;
    if s.len() != 2 {
        return None;
    }
    Some((s[0].as_f64()?, s[1].as_f64()?))
}

impl Comparator for LatLongComparator {
    fn name(&self) -> String {
        "latlong".to_string()
    }
    fn compare(&self, a: &Value, b: &Value) -> Vec<f32> {
        let mut out = [0f32; 1];
        self.compare_into(a, b, &mut out);
        out.to_vec()
    }
    fn compare_into(&self, a: &Value, b: &Value, out: &mut [f32]) {
        out[0] = match (as_latlong(a), as_latlong(b)) {
            (Some(x), Some(y)) => haversine_km(x, y).sqrt() as f32,
            _ => f32::NAN,
        };
    }
}

// ---------------------------------------------------------------------------
// Cosine similarity (simplecosine)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CosineKind {
    Text,
    Set,
}

/// Port of `simplecosine.cosine.CosineSimilarity`.
#[derive(Debug)]
pub struct CosineComparator {
    kind: CosineKind,
    doc_freq: FxHashMap<String, f64>,
    default_score: f64,
}

impl CosineComparator {
    pub fn new(kind: CosineKind, corpus: &[Value]) -> Self {
        let mut doc_freq: FxHashMap<String, i64> = FxHashMap::default();
        let mut num_docs = 0.0f64;
        for document in corpus {
            let words = list_words(kind, document);
            if words.is_empty() && !document.is_truthy() {
                continue;
            }
            if !document.is_truthy() {
                continue;
            }
            let unique: BTreeSet<&String> = words.iter().collect();
            for word in unique {
                *doc_freq.entry(word.clone()).or_insert(0) += 1;
            }
            num_docs += 1.0;
        }

        let mut df = FxHashMap::default();
        for (word, count) in doc_freq {
            df.insert(word, (num_docs / count as f64).ln());
        }
        let default_score = if num_docs > 0.0 { num_docs.ln() } else { 1.0 };

        Self {
            kind,
            doc_freq: df,
            default_score,
        }
    }

    fn vectorize(&self, field: &Value) -> (FxHashMap<String, f64>, f64) {
        let words = list_words(self.kind, field);
        let mut vector: FxHashMap<String, f64> = FxHashMap::default();
        for word in words {
            let w = *self.doc_freq.get(&word).unwrap_or(&self.default_score);
            *vector.entry(word).or_insert(0.0) += w;
        }
        let norm = vector.values().map(|w| w * w).sum::<f64>().sqrt();
        (vector, norm)
    }
}

fn list_words(kind: CosineKind, document: &Value) -> Vec<String> {
    match kind {
        CosineKind::Text => document
            .py_str()
            .split_whitespace()
            .map(|s| s.to_string())
            .collect(),
        CosineKind::Set => document
            .as_seq()
            .map(|s| s.iter().map(|v| v.py_str()).collect())
            .unwrap_or_default(),
    }
}

impl Comparator for CosineComparator {
    fn name(&self) -> String {
        match self.kind {
            CosineKind::Text => "cosine_text".to_string(),
            CosineKind::Set => "cosine_set".to_string(),
        }
    }
    fn compare(&self, a: &Value, b: &Value) -> Vec<f32> {
        let mut out = [0f32; 1];
        self.compare_into(a, b, &mut out);
        out.to_vec()
    }
    fn compare_into(&self, a: &Value, b: &Value, out: &mut [f32]) {
        let (v1, n1) = self.vectorize(a);
        let (v2, n2) = self.vectorize(b);
        out[0] = if n1 != 0.0 && n2 != 0.0 {
            let mut numerator = 0.0;
            let (small, big) = if v1.len() <= v2.len() { (&v1, &v2) } else { (&v2, &v1) };
            for (word, w) in small {
                if let Some(other) = big.get(word) {
                    numerator += w * other;
                }
            }
            (numerator / (n1 * n2)) as f32
        } else {
            f32::NAN
        };
    }
}

// ---------------------------------------------------------------------------
// Custom
// ---------------------------------------------------------------------------

/// A user-supplied comparator, mirroring `dedupe.variables.Custom`.
pub struct CustomComparator {
    pub label: String,
    pub func: Arc<dyn Fn(&Value, &Value) -> f32 + Send + Sync>,
    pub len: usize,
}

impl std::fmt::Debug for CustomComparator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CustomComparator({})", self.label)
    }
}

impl Comparator for CustomComparator {
    fn name(&self) -> String {
        self.label.clone()
    }
    fn compare(&self, a: &Value, b: &Value) -> Vec<f32> {
        vec![(self.func)(a, b)]
    }
    fn len(&self) -> usize {
        self.len
    }
}

/// A comparator over the derived dummy variables of categorical / exists types
/// that are produced by an interaction.  It is never called directly.
#[derive(Debug)]
pub struct DummyComparator;

impl Comparator for DummyComparator {
    fn name(&self) -> String {
        "dummy".to_string()
    }
    fn compare(&self, _a: &Value, _b: &Value) -> Vec<f32> {
        vec![0.0]
    }
}

/// Compare two values for the exact comparator with Python `==` semantics.
pub fn values_equal(a: &Value, b: &Value) -> bool {
    cmp_values(a, b) == std::cmp::Ordering::Equal && a == b
}

pub mod crf;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine_gap_matches_reference() {
        assert!((normalized_affine_gap_distance("a", "a") - 0.5).abs() < 1e-6);
        assert!((normalized_affine_gap_distance("ab", "ab") - 0.5).abs() < 1e-6);
        assert!((normalized_affine_gap_distance("", "abc") - affine_gap_distance("", "abc", 1.0, 11.0, 10.0, 7.0, 0.125) / 3.0).abs() < 1e-6);
    }

    #[test]
    fn exact() {
        let c = ExactComparator;
        assert_eq!(c.compare(&Value::str("a"), &Value::str("a")), vec![1.0]);
        assert_eq!(c.compare(&Value::str("a"), &Value::str("b")), vec![0.0]);
    }

    #[test]
    fn categorical() {
        let c = CategoricalComparator::new(&["a".into(), "b".into(), "c".into()]);
        let cc = CategoricalTypeComparator { inner: c };
        assert_eq!(cc.compare(&Value::str("a"), &Value::str("b")), vec![0.0, 0.0, 1.0, 0.0, 0.0]);
        assert_eq!(cc.compare(&Value::str("a"), &Value::str("c")), vec![0.0, 0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn exists() {
        let c = ExistsComparator::new();
        assert_eq!(c.compare(&Value::Null, &Value::Null), vec![0.0, 0.0]);
        assert_eq!(c.compare(&Value::Int(1), &Value::Int(1)), vec![1.0, 0.0]);
        assert_eq!(c.compare(&Value::Int(1), &Value::Int(0)), vec![0.0, 1.0]);
    }

    #[test]
    fn price() {
        assert_eq!(PriceComparator.compare(&Value::Int(1), &Value::Int(10)), vec![1.0]);
        assert_eq!(PriceComparator.compare(&Value::Int(10), &Value::Int(1)), vec![1.0]);
    }

    #[test]
    fn cosine_text() {
        let c = CosineComparator::new(CosineKind::Text, &[]);
        // identical non-empty strings are maximally similar
        let r = c.compare(&Value::str("hello world"), &Value::str("hello world"));
        assert!((r[0] - 1.0).abs() < 1e-6);
        let r = c.compare(&Value::str("hello"), &Value::str("world"));
        assert!(r[0].is_nan() || r[0] <= 1.0);
    }
}
