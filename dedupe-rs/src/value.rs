//! Record and value types.
//!
//! Mirrors the dynamically typed `dict[str, Any]` records used by the Python
//! library with a small, explicit enum.  Rendering helpers reproduce Python's
//! `str()` / `repr()` output for the values that blocking predicates stringify.

use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};

use indexmap::IndexMap;

/// A record identifier.  Python allows record ids to be either `int` or `str`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum RecordId {
    Int(i64),
    Str(String),
}

impl RecordId {
    pub fn is_int(&self) -> bool {
        matches!(self, RecordId::Int(_))
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            RecordId::Int(i) => Some(*i),
            RecordId::Str(_) => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            RecordId::Int(_) => None,
            RecordId::Str(s) => Some(s),
        }
    }
}

impl fmt::Display for RecordId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RecordId::Int(i) => write!(f, "{i}"),
            RecordId::Str(s) => write!(f, "{s}"),
        }
    }
}

impl From<i64> for RecordId {
    fn from(v: i64) -> Self {
        RecordId::Int(v)
    }
}

impl From<i32> for RecordId {
    fn from(v: i32) -> Self {
        RecordId::Int(v as i64)
    }
}

impl From<&str> for RecordId {
    fn from(v: &str) -> Self {
        RecordId::Str(v.to_string())
    }
}

impl From<String> for RecordId {
    fn from(v: String) -> Self {
        RecordId::Str(v)
    }
}

/// A dynamically typed value, roughly equivalent to a Python object restricted
/// to the types that appear in records.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    List(Vec<Value>),
    Tuple(Vec<Value>),
    Set(Vec<Value>),
}

impl Value {
    pub fn str(s: impl Into<String>) -> Value {
        Value::Str(s.into())
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    /// Truthiness following Python semantics.
    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Null => false,
            Value::Bool(b) => *b,
            Value::Int(i) => *i != 0,
            Value::Float(f) => *f != 0.0,
            Value::Str(s) => !s.is_empty(),
            Value::List(v) | Value::Tuple(v) | Value::Set(v) => !v.is_empty(),
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Int(i) => Some(*i as f64),
            Value::Float(f) => Some(*f),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_seq(&self) -> Option<&[Value]> {
        match self {
            Value::List(v) | Value::Tuple(v) | Value::Set(v) => Some(v),
            _ => None,
        }
    }

    /// Python `str(value)`.
    pub fn py_str(&self) -> String {
        match self {
            Value::Null => "None".to_string(),
            Value::Bool(b) => if *b { "True" } else { "False" }.to_string(),
            Value::Int(i) => i.to_string(),
            Value::Float(f) => py_float_repr(*f),
            Value::Str(s) => s.clone(),
            Value::List(items) => {
                let inner: Vec<String> = items.iter().map(py_repr).collect();
                format!("[{}]", inner.join(", "))
            }
            Value::Tuple(items) => {
                if items.len() == 1 {
                    format!("({},)", py_repr(&items[0]))
                } else {
                    let inner: Vec<String> = items.iter().map(py_repr).collect();
                    format!("({})", inner.join(", "))
                }
            }
            Value::Set(items) => {
                if items.is_empty() {
                    "set()".to_string()
                } else {
                    // Python set iteration order is hash-dependent and therefore
                    // process dependent; we render deterministically instead.
                    let mut reprs: Vec<String> = items.iter().map(py_repr).collect();
                    reprs.sort();
                    format!("{{{}}}", reprs.join(", "))
                }
            }
        }
    }

    /// A total-ish ordering used by set element predicates (`min`/`max`).
    pub fn sort_key(&self) -> NotNanKey<'_> {
        NotNanKey(self)
    }
}

/// Python `repr(value)`.
pub fn py_repr(v: &Value) -> String {
    match v {
        Value::Str(s) => format!("'{}'", s.replace('\\', "\\\\").replace('\'', "\\'")),
        other => other.py_str(),
    }
}

/// Render an `f64` the way Python's `repr`/`str` would for the common range.
pub fn py_float_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    if x == x.trunc() && x.abs() < 1e16 {
        // Integral: Python always shows a trailing `.0`.
        return format!("{:.1}", x);
    }
    let abs = x.abs();
    if abs != 0.0 && !(1e-4..1e16).contains(&abs) {
        // Python switches to exponential notation.
        let s = format!("{:e}", x);
        // Rust renders `1e20`, Python renders `1e+20`.
        if let Some(pos) = s.find('e') {
            let (mantissa, exp) = s.split_at(pos);
            let exp = &exp[1..];
            let exp = if let Some(stripped) = exp.strip_prefix('-') {
                format!("-{stripped}")
            } else {
                format!("+{exp}")
            };
            return format!("{mantissa}e{exp}");
        }
    }
    format!("{}", x)
}

/// Wrapper providing `Ord` for `Value` by comparing Python-like sort keys.
pub struct NotNanKey<'a>(pub &'a Value);

impl PartialEq for NotNanKey<'_> {
    fn eq(&self, other: &Self) -> bool {
        cmp_values(self.0, other.0) == Ordering::Equal
    }
}
impl Eq for NotNanKey<'_> {}
impl PartialOrd for NotNanKey<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for NotNanKey<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        cmp_values(self.0, other.0)
    }
}

/// Best-effort ordering matching Python's ordering of the supported value types.
pub fn cmp_values(a: &Value, b: &Value) -> Ordering {
    fn rank(v: &Value) -> u8 {
        match v {
            Value::Null => 0,
            Value::Bool(_) => 1,
            Value::Int(_) | Value::Float(_) => 2,
            Value::Str(_) => 3,
            Value::List(_) | Value::Tuple(_) => 4,
            Value::Set(_) => 5,
        }
    }
    match rank(a).cmp(&rank(b)) {
        Ordering::Equal => match (a, b) {
            (Value::Null, Value::Null) => Ordering::Equal,
            (Value::Bool(x), Value::Bool(y)) => x.cmp(y),
            (Value::Int(x), Value::Int(y)) => x.cmp(y),
            (Value::Float(x), Value::Float(y)) => x.partial_cmp(y).unwrap_or(Ordering::Equal),
            (Value::Int(x), Value::Float(y)) => (*x as f64).partial_cmp(y).unwrap_or(Ordering::Equal),
            (Value::Float(x), Value::Int(y)) => x.partial_cmp(&(*y as f64)).unwrap_or(Ordering::Equal),
            (Value::Str(x), Value::Str(y)) => x.cmp(y),
            (Value::List(x) | Value::Tuple(x), Value::List(y) | Value::Tuple(y)) => {
                cmp_seqs(x, y)
            }
            (Value::Set(x), Value::Set(y)) => cmp_seqs(x, y),
            _ => Ordering::Equal,
        },
        other => other,
    }
}

fn cmp_seqs(a: &[Value], b: &[Value]) -> Ordering {
    for (x, y) in a.iter().zip(b.iter()) {
        match cmp_values(x, y) {
            Ordering::Equal => continue,
            other => return other,
        }
    }
    a.len().cmp(&b.len())
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Null, Value::Null) => true,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Int(a), Value::Int(b)) => a == b,
            (Value::Int(a), Value::Float(b)) | (Value::Float(b), Value::Int(a)) => {
                (*a as f64) == *b
            }
            (Value::Float(a), Value::Float(b)) => a == b,
            (Value::Str(a), Value::Str(b)) => a == b,
            (Value::List(a), Value::List(b)) => a == b,
            (Value::Tuple(a), Value::Tuple(b)) => a == b,
            (Value::Set(a), Value::Set(b)) => {
                if a.len() != b.len() {
                    return false;
                }
                // Order insensitive comparison.
                let mut b_remaining: Vec<&Value> = b.iter().collect();
                for x in a {
                    if let Some(pos) = b_remaining.iter().position(|y| *y == x) {
                        b_remaining.swap_remove(pos);
                    } else {
                        return false;
                    }
                }
                true
            }
            _ => false,
        }
    }
}
impl Eq for Value {}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Value::Null => 0u8.hash(state),
            Value::Bool(b) => {
                1u8.hash(state);
                b.hash(state);
            }
            Value::Int(i) => {
                2u8.hash(state);
                i.hash(state);
            }
            Value::Float(f) => {
                2u8.hash(state);
                // Hash integral floats the same as the equivalent int.
                if f.fract() == 0.0 && f.abs() < 9.007_199_254_740_992e15 {
                    (*f as i64).hash(state);
                } else {
                    f.to_bits().hash(state);
                }
            }
            Value::Str(s) => {
                3u8.hash(state);
                s.hash(state);
            }
            Value::List(v) => {
                4u8.hash(state);
                v.hash(state);
            }
            Value::Tuple(v) => {
                5u8.hash(state);
                v.hash(state);
            }
            Value::Set(v) => {
                6u8.hash(state);
                // Order insensitive.
                let mut h: u64 = 0;
                for item in v {
                    let mut sub = rustc_hash::FxHasher::default();
                    item.hash(&mut sub);
                    h ^= sub.finish();
                }
                h.hash(state);
            }
        }
    }
}

/// A record is an ordered map of field name to value.
pub type Record = IndexMap<String, Value>;

/// A dataset is an ordered map of record id to record.
pub type Data = IndexMap<RecordId, Record>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn py_str_set_sorted() {
        let v = Value::Set(vec![
            Value::str("red"),
            Value::str("blue"),
            Value::str("green"),
        ]);
        assert_eq!(v.py_str(), "{'blue', 'green', 'red'}");
    }

    #[test]
    fn py_str_tuple() {
        let v = Value::Tuple(vec![Value::Float(1.1), Value::Float(-5.0)]);
        assert_eq!(v.py_str(), "(1.1, -5.0)");
        assert_eq!(Value::Tuple(vec![Value::Int(1)]).py_str(), "(1,)");
    }

    #[test]
    fn float_repr() {
        assert_eq!(py_float_repr(1.1), "1.1");
        assert_eq!(py_float_repr(-5.0), "-5.0");
        assert_eq!(py_float_repr(42.0), "42.0");
    }
}
