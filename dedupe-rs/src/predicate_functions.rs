//! Blocking predicate functions, ported from `dedupe/predicate_functions.py`.
//!
//! In the Python library these accept arbitrary objects.  The Rust port takes a
//! [`Value`] and coerces it, mirroring how `SimplePredicate` / `StringPredicate`
//! feed values in.

use std::collections::BTreeSet;
use std::sync::OnceLock;

use regex::Regex;

use crate::cpredicates::{initials, unique_ngrams};
use crate::value::{py_float_repr, Value};

pub type Keys = BTreeSet<String>;

fn re(cell: &'static OnceLock<Regex>, pattern: &str) -> &'static Regex {
    cell.get_or_init(|| Regex::new(pattern).expect("valid regex"))
}

fn words_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    re(&R, r"[\w']+")
}

fn integers_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    re(&R, r"\d+")
}

fn start_word_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    re(&R, r"^([\w']+)")
}

fn two_start_words_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    re(&R, r"^([\w']+\W+[\w']+)")
}

fn start_integer_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    re(&R, r"^(\d+)")
}

fn as_str(v: &Value) -> String {
    v.as_str().map(|s| s.to_string()).unwrap_or_else(|| v.py_str())
}

fn set1(s: impl Into<String>) -> Keys {
    let mut k = BTreeSet::new();
    k.insert(s.into());
    k
}

/// `str(int(i))`: strips leading zeros from a run of ASCII digits.
fn normalize_int_str(s: &str) -> String {
    let t = s.trim_start_matches('0');
    if t.is_empty() {
        "0".to_string()
    } else {
        t.to_string()
    }
}

fn parse_i128(s: &str) -> Option<i128> {
    s.parse::<i128>().ok()
}

/// Return the whole field as a string.
pub fn whole_field_predicate(field: &Value) -> Keys {
    set1(field.py_str())
}

/// Return the whitespace/word tokens.
pub fn token_field_predicate(field: &Value) -> Keys {
    words_re()
        .find_iter(&as_str(field))
        .map(|m| m.as_str().to_string())
        .collect()
}

pub fn first_token_predicate(field: &Value) -> Keys {
    let s = as_str(field);
    match start_word_re().captures(&s) {
        Some(c) => set1(c.get(1).unwrap().as_str()),
        None => Keys::new(),
    }
}

pub fn first_two_tokens_predicate(field: &Value) -> Keys {
    let s = as_str(field);
    match two_start_words_re().captures(&s) {
        Some(c) => set1(c.get(1).unwrap().as_str()),
        None => Keys::new(),
    }
}

/// Return any integers with leading zeros removed.
pub fn common_integer_predicate(field: &Value) -> Keys {
    integers_re()
        .find_iter(&as_str(field))
        .map(|m| normalize_int_str(m.as_str()))
        .collect()
}

/// Maximal alphanumeric runs that contain at least one digit.
pub fn alpha_numeric_predicate(field: &Value) -> Keys {
    let s = as_str(field);
    let mut out = Keys::new();
    let bytes = s.as_bytes();
    let is_alnum = |b: u8| b.is_ascii_alphanumeric();
    let mut i = 0;
    while i < bytes.len() {
        if is_alnum(bytes[i]) {
            let start = i;
            let mut has_digit = false;
            while i < bytes.len() && is_alnum(bytes[i]) {
                if bytes[i].is_ascii_digit() {
                    has_digit = true;
                }
                i += 1;
            }
            if has_digit {
                out.insert(s[start..i].to_string());
            }
        } else {
            i += 1;
        }
    }
    out
}

/// For any integer N in field return the integers N-1, N and N+1.
pub fn near_integers_predicate(field: &Value) -> Keys {
    let mut out = Keys::new();
    for m in integers_re().find_iter(&as_str(field)) {
        let s = m.as_str();
        if let Some(n) = parse_i128(s) {
            out.insert((n - 1).to_string());
            out.insert(n.to_string());
            out.insert((n + 1).to_string());
        } else {
            out.insert(normalize_int_str(s));
        }
    }
    out
}

pub fn hundred_integer_predicate(field: &Value) -> Keys {
    integers_re()
        .find_iter(&as_str(field))
        .map(|m| {
            let s = normalize_int_str(m.as_str());
            let mut r: String = s.chars().take(s.chars().count().saturating_sub(2)).collect();
            r.push_str("00");
            r
        })
        .collect()
}

pub fn hundred_integers_odd_predicate(field: &Value) -> Keys {
    integers_re()
        .find_iter(&as_str(field))
        .map(|m| {
            let s = normalize_int_str(m.as_str());
            let mut r: String = s.chars().take(s.chars().count().saturating_sub(2)).collect();
            r.push('0');
            if let Some(n) = parse_i128(&s) {
                r.push_str(&(n % 2).to_string());
            } else {
                let last = s.chars().last().unwrap_or('0');
                let odd = last.to_digit(10).map(|d| d % 2).unwrap_or(0);
                r.push_str(&odd.to_string());
            }
            r
        })
        .collect()
}

pub fn first_integer_predicate(field: &Value) -> Keys {
    let s = as_str(field);
    match start_integer_re().captures(&s) {
        Some(c) => set1(c.get(1).unwrap().as_str()),
        None => Keys::new(),
    }
}

/// Contiguous `n`-token grams of a token sequence.
pub fn ngrams_tokens(field: &[Value], n: usize) -> Keys {
    let n_tokens = field.len();
    if n == 0 || n_tokens < n {
        return Keys::new();
    }
    let mut grams = Keys::new();
    for i in 0..=(n_tokens - n) {
        let joined = field[i..i + n]
            .iter()
            .map(|v| v.py_str())
            .collect::<Vec<_>>()
            .join(" ");
        grams.insert(joined);
    }
    grams
}

pub fn common_two_tokens(field: &Value) -> Keys {
    let tokens: Vec<Value> = as_str(field)
        .split_whitespace()
        .map(|s| Value::Str(s.to_string()))
        .collect();
    ngrams_tokens(&tokens, 2)
}

pub fn common_three_tokens(field: &Value) -> Keys {
    let tokens: Vec<Value> = as_str(field)
        .split_whitespace()
        .map(|s| Value::Str(s.to_string()))
        .collect();
    ngrams_tokens(&tokens, 3)
}

pub fn fingerprint(field: &Value) -> Keys {
    let s = as_str(field);
    let mut toks: Vec<&str> = s.split_whitespace().collect();
    toks.sort_unstable();
    set1(toks.concat())
}

pub fn one_gram_fingerprint(field: &Value) -> Keys {
    let s = as_str(field).replace(' ', "");
    let set: BTreeSet<char> = s.chars().collect();
    set1(set.into_iter().collect::<String>())
}

pub fn two_gram_fingerprint(field: &Value) -> Keys {
    let original = as_str(field);
    let s = original.replace(' ', "");
    if original.chars().count() > 1 {
        let mut grams: Vec<String> = unique_ngrams(&s, 2).into_iter().collect();
        grams.sort();
        set1(grams.concat())
    } else {
        Keys::new()
    }
}

pub fn common_four_gram(field: &Value) -> Keys {
    unique_ngrams(&as_str(field).replace(' ', ""), 4)
}

pub fn common_six_gram(field: &Value) -> Keys {
    unique_ngrams(&as_str(field).replace(' ', ""), 6)
}

pub fn same_three_char_start_predicate(field: &Value) -> Keys {
    initials(&as_str(field).replace(' ', ""), 3)
}

pub fn same_five_char_start_predicate(field: &Value) -> Keys {
    initials(&as_str(field).replace(' ', ""), 5)
}

pub fn same_seven_char_start_predicate(field: &Value) -> Keys {
    initials(&as_str(field).replace(' ', ""), 7)
}

pub fn suffix_array(field: &Value) -> Keys {
    let s = as_str(field);
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len() as i64 - 4;
    if n > 0 {
        (0..n)
            .map(|i| chars[i as usize..].iter().collect())
            .collect()
    } else {
        Keys::new()
    }
}

pub fn sorted_acronym(field: &Value) -> Keys {
    let mut firsts: Vec<char> = as_str(field)
        .split_whitespace()
        .filter_map(|t| t.chars().next())
        .collect();
    firsts.sort_unstable();
    set1(firsts.into_iter().collect::<String>())
}

pub fn double_metaphone(field: &Value) -> Keys {
    let s = as_str(field);
    let (primary, secondary) = crate::double_metaphone::double_metaphone_codes(&s);
    let mut out = Keys::new();
    for code in [primary, secondary] {
        if !code.is_empty() {
            out.insert(code);
        }
    }
    out
}

pub fn metaphone_token(field: &Value) -> Keys {
    let s = as_str(field);
    let mut out = Keys::new();
    for token in s.split_whitespace() {
        let (primary, secondary) = crate::double_metaphone::double_metaphone_codes(token);
        for code in [primary, secondary] {
            if !code.is_empty() {
                out.insert(code);
            }
        }
    }
    out
}

pub fn whole_set_predicate(field_set: &Value) -> Keys {
    set1(field_set.py_str())
}

pub fn common_set_element_predicate(field_set: &Value) -> Keys {
    match field_set.as_seq() {
        Some(items) => items.iter().map(|i| i.py_str()).collect(),
        None => Keys::new(),
    }
}

pub fn common_two_elements_predicate(field: &Value) -> Keys {
    let mut items: Vec<Value> = field.as_seq().map(|s| s.to_vec()).unwrap_or_default();
    items.sort_by(crate::value::cmp_values);
    ngrams_tokens(&items, 2)
}

pub fn common_three_elements_predicate(field: &Value) -> Keys {
    let mut items: Vec<Value> = field.as_seq().map(|s| s.to_vec()).unwrap_or_default();
    items.sort_by(crate::value::cmp_values);
    ngrams_tokens(&items, 3)
}

pub fn last_set_element_predicate(field_set: &Value) -> Keys {
    match field_set.as_seq() {
        Some(items) if !items.is_empty() => {
            let max = items
                .iter()
                .max_by(|a, b| crate::value::cmp_values(a, b))
                .unwrap();
            set1(max.py_str())
        }
        _ => Keys::new(),
    }
}

pub fn first_set_element_predicate(field_set: &Value) -> Keys {
    match field_set.as_seq() {
        Some(items) if !items.is_empty() => {
            let min = items
                .iter()
                .min_by(|a, b| crate::value::cmp_values(a, b))
                .unwrap();
            set1(min.py_str())
        }
        _ => Keys::new(),
    }
}

pub fn magnitude_of_cardinality(field_set: &Value) -> Keys {
    match field_set.as_seq() {
        Some(items) => order_of_magnitude(&Value::Int(items.len() as i64)),
        None => Keys::new(),
    }
}

/// Round a float to `n` decimal places using round-half-to-even, matching
/// Python's `round`.
pub fn round_half_even(x: f64, ndigits: i32) -> f64 {
    if !x.is_finite() {
        return x;
    }
    if ndigits >= 0 {
        let s = format!("{:.*}", ndigits as usize, x);
        s.parse().unwrap_or(x)
    } else {
        let scale = 10f64.powi(-ndigits);
        let scaled = x / scale;
        let r = scaled.round_ties_even();
        r * scale
    }
}

pub fn lat_long_grid_predicate(field: &Value) -> Keys {
    let items = match field.as_seq() {
        Some(s) if s.len() >= 2 => s,
        _ => return Keys::new(),
    };
    let any_truthy = items.iter().any(|v| v.is_truthy());
    if !any_truthy {
        return Keys::new();
    }
    let rounded: Vec<Value> = items
        .iter()
        .map(|v| match v {
            // Python's `round(int, ndigits)` returns an int unchanged.
            Value::Int(i) => Value::Int(*i),
            other => Value::Float(round_half_even(other.as_f64().unwrap_or(0.0), 1)),
        })
        .collect();
    set1(Value::Tuple(rounded).py_str())
}

pub fn order_of_magnitude(field: &Value) -> Keys {
    let x = match field.as_f64() {
        Some(x) => x,
        None => return Keys::new(),
    };
    if x > 0.0 {
        let m = x.log10().round_ties_even();
        set1(format!("{}", m as i64))
    } else {
        Keys::new()
    }
}

/// Round to one significant figure, returning an integer string.
pub fn round_to_1(field: &Value) -> Keys {
    let x = match field.as_f64() {
        Some(x) => x,
        None => return Keys::new(),
    };
    let abs_num = x.abs();
    if abs_num == 0.0 {
        return set1("0");
    }
    let order = abs_num.log10().floor() as i32;
    let rounded = round_half_even(abs_num, -order);
    let signed = if x.is_sign_negative() {
        -rounded
    } else {
        rounded
    };
    set1(format!("{}", signed as i64))
}

/// Deterministic ordering helper used by predicate caches.
pub fn py_float_string(x: f64) -> String {
    py_float_repr(x)
}

/// Look up a predicate function by its Python name.  Used for parity testing
/// and for reconstructing variables from serialized settings.
pub fn by_name(name: &str) -> Option<fn(&Value) -> Keys> {
    Some(match name {
        "wholeFieldPredicate" => whole_field_predicate,
        "tokenFieldPredicate" => token_field_predicate,
        "firstTokenPredicate" => first_token_predicate,
        "firstTwoTokensPredicate" => first_two_tokens_predicate,
        "commonIntegerPredicate" => common_integer_predicate,
        "alphaNumericPredicate" => alpha_numeric_predicate,
        "nearIntegersPredicate" => near_integers_predicate,
        "hundredIntegerPredicate" => hundred_integer_predicate,
        "hundredIntegersOddPredicate" => hundred_integers_odd_predicate,
        "firstIntegerPredicate" => first_integer_predicate,
        "commonTwoTokens" => common_two_tokens,
        "commonThreeTokens" => common_three_tokens,
        "fingerprint" => fingerprint,
        "oneGramFingerprint" => one_gram_fingerprint,
        "twoGramFingerprint" => two_gram_fingerprint,
        "commonFourGram" => common_four_gram,
        "commonSixGram" => common_six_gram,
        "sameThreeCharStartPredicate" => same_three_char_start_predicate,
        "sameFiveCharStartPredicate" => same_five_char_start_predicate,
        "sameSevenCharStartPredicate" => same_seven_char_start_predicate,
        "suffixArray" => suffix_array,
        "sortedAcronym" => sorted_acronym,
        "doubleMetaphone" => double_metaphone,
        "metaphoneToken" => metaphone_token,
        "wholeSetPredicate" => whole_set_predicate,
        "commonSetElementPredicate" => common_set_element_predicate,
        "commonTwoElementsPredicate" => common_two_elements_predicate,
        "commonThreeElementsPredicate" => common_three_elements_predicate,
        "lastSetElementPredicate" => last_set_element_predicate,
        "firstSetElementPredicate" => first_set_element_predicate,
        "magnitudeOfCardinality" => magnitude_of_cardinality,
        "latLongGridPredicate" => lat_long_grid_predicate,
        "orderOfMagnitude" => order_of_magnitude,
        "roundTo1" => round_to_1,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(s: &str) -> Value {
        Value::Str(s.to_string())
    }

    #[test]
    fn whole_field() {
        assert_eq!(whole_field_predicate(&v("donald")), set1("donald"));
        assert_eq!(whole_field_predicate(&v("go-of,y  ")), set1("go-of,y  "));
    }

    #[test]
    fn token_field() {
        assert_eq!(token_field_predicate(&v("donald")), set1("donald"));
        assert_eq!(
            token_field_predicate(&v("do\nal d")),
            BTreeSet::from(["al".into(), "d".into(), "do".into()])
        );
        assert_eq!(
            token_field_predicate(&v("go-of y  ")),
            BTreeSet::from(["go".into(), "of".into(), "y".into()])
        );
    }

    #[test]
    fn first_token() {
        assert_eq!(first_token_predicate(&v("donald")), set1("donald"));
        assert_eq!(first_token_predicate(&v("don ald")), set1("don"));
        assert_eq!(first_token_predicate(&v(" cip   ciop ")), Keys::new());
    }

    #[test]
    fn first_two_tokens() {
        assert_eq!(first_two_tokens_predicate(&v("donald")), Keys::new());
        assert_eq!(first_two_tokens_predicate(&v("don ald")), set1("don ald"));
        assert_eq!(first_two_tokens_predicate(&v("go-of y  ")), set1("go-of"));
        assert_eq!(first_two_tokens_predicate(&v(" cip   ciop ")), Keys::new());
    }

    #[test]
    fn integers() {
        assert_eq!(common_integer_predicate(&v("g00fy  ")), set1("0"));
        assert_eq!(
            common_integer_predicate(&v(" c1p   c10p ")),
            BTreeSet::from(["1".into(), "10".into()])
        );
        assert_eq!(
            near_integers_predicate(&v("don4ld")),
            BTreeSet::from(["3".into(), "4".into(), "5".into()])
        );
        assert_eq!(hundred_integer_predicate(&v("don456ld")), set1("400"));
        assert_eq!(hundred_integer_predicate(&v("g00fy  ")), set1("00"));
        assert_eq!(
            hundred_integers_odd_predicate(&v(" c111p   c1230p ")),
            BTreeSet::from(["101".into(), "1200".into()])
        );
        assert_eq!(first_integer_predicate(&v("00fy  ")), set1("00"));
        assert_eq!(first_integer_predicate(&v("donald 456")), Keys::new());
    }

    #[test]
    fn alnum() {
        assert_eq!(alpha_numeric_predicate(&v("don4ld")), set1("don4ld"));
        assert_eq!(alpha_numeric_predicate(&v("1a")), set1("1a"));
        assert_eq!(alpha_numeric_predicate(&v("asdf")), Keys::new());
        assert_eq!(alpha_numeric_predicate(&v("a_1")), set1("1"));
        assert_eq!(
            alpha_numeric_predicate(&v("773-555-1676")),
            BTreeSet::from(["1676".into(), "555".into(), "773".into()])
        );
    }

    #[test]
    fn tokens() {
        assert_eq!(
            common_two_tokens(&v("d on 456 ld")),
            BTreeSet::from(["456 ld".into(), "d on".into(), "on 456".into()])
        );
        assert_eq!(
            common_three_tokens(&v("d on 456 ld")),
            BTreeSet::from(["d on 456".into(), "on 456 ld".into()])
        );
        assert_eq!(fingerprint(&v("don 456 ld ")), set1("456donld"));
        assert_eq!(one_gram_fingerprint(&v(" g00fy  ")), set1("0fgy"));
        assert_eq!(two_gram_fingerprint(&v("don4ld")), set1("4ldoldn4on"));
        assert_eq!(two_gram_fingerprint(&v("7")), Keys::new());
    }

    #[test]
    fn grams() {
        assert_eq!(
            common_four_gram(&v("don4ld")),
            BTreeSet::from(["don4".into(), "n4ld".into(), "on4l".into()])
        );
        assert_eq!(common_six_gram(&v("g00fy  ")), Keys::new());
        assert_eq!(same_three_char_start_predicate(&v(" c1p   c10p ")), set1("c1p"));
        assert_eq!(same_five_char_start_predicate(&v("donald 1992")), set1("donal"));
        assert_eq!(same_seven_char_start_predicate(&v(" g00fy  ")), set1("g00fy"));
        assert_eq!(
            suffix_array(&v("g00fy  ")),
            BTreeSet::from(["0fy  ".into(), "00fy  ".into(), "g00fy  ".into()])
        );
        assert_eq!(sorted_acronym(&v("donald 19 92")), set1("19d"));
    }

    #[test]
    fn metaphone() {
        assert_eq!(double_metaphone(&v("donald")), set1("TNLT"));
        assert_eq!(
            double_metaphone(&v("cipciop")),
            BTreeSet::from(["SPSP".into(), "SPXP".into()])
        );
        // token-level metaphone (issue covered by test_predicates.py)
        assert_eq!(
            metaphone_token(&v("9301 S. State St. ")),
            BTreeSet::from(["S".into(), "ST".into(), "STT".into()])
        );
        assert_eq!(metaphone_token(&v("don ald")), BTreeSet::from(["ALT".into(), "TN".into()]));
    }

    #[test]
    fn set_predicates() {
        let s = Value::Set(vec![v("red"), v("blue"), v("green")]);
        assert_eq!(common_set_element_predicate(&s), BTreeSet::from(["blue".into(), "green".into(), "red".into()]));
        assert_eq!(last_set_element_predicate(&s), set1("red"));
        assert_eq!(first_set_element_predicate(&s), set1("blue"));
        assert_eq!(magnitude_of_cardinality(&s), set1("0"));
        assert_eq!(magnitude_of_cardinality(&Value::Set(vec![])), Keys::new());
        assert_eq!(
            common_two_elements_predicate(&Value::Tuple(vec![Value::Int(1), Value::Int(2), Value::Int(3)])),
            BTreeSet::from(["1 2".into(), "2 3".into()])
        );
    }

    #[test]
    fn numeric_predicates() {
        assert_eq!(order_of_magnitude(&Value::Int(10)), set1("1"));
        assert_eq!(order_of_magnitude(&Value::Int(9)), set1("1"));
        assert_eq!(order_of_magnitude(&Value::Int(2)), set1("0"));
        assert_eq!(order_of_magnitude(&Value::Int(-2)), Keys::new());
        assert_eq!(round_to_1(&Value::Int(22315)), set1("20000"));
        assert_eq!(round_to_1(&Value::Int(-22315)), set1("-20000"));
    }

    #[test]
    fn latlong() {
        let f = Value::Tuple(vec![Value::Float(42.535), Value::Float(-5.012)]);
        assert_eq!(lat_long_grid_predicate(&f), set1("(42.5, -5.0)"));
        assert_eq!(
            lat_long_grid_predicate(&Value::Tuple(vec![Value::Float(0.0), Value::Float(0.0)])),
            Keys::new()
        );
        assert_eq!(
            lat_long_grid_predicate(&Value::Tuple(vec![Value::Float(1.11), Value::Float(2.27)])),
            set1("(1.1, 2.3)")
        );
    }
}
