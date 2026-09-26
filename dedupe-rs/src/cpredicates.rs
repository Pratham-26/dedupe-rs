//! Character-level predicates, ported from `dedupe/cpredicates.pyx`.

use std::collections::BTreeSet;

/// All contiguous sequences of `n` characters of `field`.
///
/// ```
/// # use dedupe::cpredicates::ngrams;
/// assert_eq!(ngrams("deduplicate", 3), vec!["ded", "edu", "dup", "upl", "pli", "lic", "ica", "cat", "ate"]);
/// ```
pub fn ngrams(field: &str, n: usize) -> Vec<String> {
    if n == 0 {
        return Vec::new();
    }
    let chars: Vec<char> = field.chars().collect();
    let n_char = chars.len();
    if n_char + 1 < n {
        return Vec::new();
    }
    let n_grams = n_char + 1 - n;
    (0..n_grams)
        .map(|i| chars[i..i + n].iter().collect())
        .collect()
}

/// All contiguous unique sequences of `n` characters of `field`.
pub fn unique_ngrams(field: &str, n: usize) -> BTreeSet<String> {
    ngrams(field, n).into_iter().collect()
}

/// The first `n` characters of `field`, or the whole field if shorter.
pub fn initials(field: &str, n: usize) -> BTreeSet<String> {
    let mut set = BTreeSet::new();
    set.insert(field.chars().take(n).collect::<String>());
    set
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ngrams_matches_python() {
        assert_eq!(ngrams("deduplicate", 1)[0], "d");
        assert_eq!(
            ngrams("deduplicate", 11),
            vec!["deduplicate"],
            "11-grams"
        );
        assert!(ngrams("deduplicate", 12).is_empty());
        assert!(ngrams("deduplicate", 100).is_empty());
    }

    #[test]
    fn ngrams_spaces() {
        assert_eq!(
            ngrams("123 16th st", 3),
            vec!["123", "23 ", "3 1", " 16", "16t", "6th", "th ", "h s", " st"]
        );
    }

    #[test]
    fn unique_ngrams_mississippi() {
        let got = unique_ngrams("mississippi", 2);
        let want: BTreeSet<String> = ["mi", "is", "ss", "si", "ip", "pp", "pi"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(got, want);
        assert!(unique_ngrams("mississippi", 12).is_empty());
    }

    #[test]
    fn initials_short() {
        assert_eq!(initials("deduplicate", 7), BTreeSet::from(["dedupli".to_string()]));
        assert_eq!(initials("deduplicate", 12), BTreeSet::from(["deduplicate".to_string()]));
        assert_eq!(initials("", 3), BTreeSet::from([String::new()]));
    }

    #[test]
    fn unicode_is_code_point_based() {
        assert_eq!(ngrams("héllo", 2), vec!["hé", "él", "ll", "lo"]);
    }
}
