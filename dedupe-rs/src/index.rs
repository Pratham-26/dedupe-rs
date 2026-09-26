//! Search indexes used by blocking index predicates.
//!
//! Mirrors `dedupe.index`, `dedupe.tfidf` (a port of Zope's `CosineIndex`) and
//! `dedupe.levenshtein` (a port of `Levenshtein_search`).

use std::cell::RefCell;
use std::collections::BTreeMap;

use rustc_hash::FxHashMap;
use std::rc::Rc;

/// A document as seen by an index.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Doc {
    /// A single normalized string (Levenshtein index).
    Text(String),
    /// A sequence of terms (TF-IDF index).
    Terms(Vec<String>),
}

impl Doc {
    pub fn terms(&self) -> &[String] {
        match self {
            Doc::Terms(t) => t,
            Doc::Text(s) => std::slice::from_ref(s),
        }
    }
}

pub trait Index {
    fn index_doc(&mut self, doc: &Doc);
    fn unindex_doc(&mut self, doc: &Doc);
    fn init_search(&mut self);
    fn search(&self, doc: &Doc, threshold: f64) -> Vec<usize>;
    /// `_doc_to_id[doc]` with `defaultdict` semantics (create on access).
    fn get_or_create_id(&mut self, doc: &Doc) -> usize;
    fn get_id(&self, doc: &Doc) -> Option<usize>;
    fn num_docs(&self) -> usize;
}

/// A shared, interior-mutable index, as owned by index predicates.
pub type SharedIndex = Rc<RefCell<Box<dyn Index>>>;

// ---------------------------------------------------------------------------
// TF-IDF cosine index (Zope `CosineIndex` + dedupe `CanopyIndex`)
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct CanopyIndex {
    lexicon_words: FxHashMap<String, usize>,
    lexicon_wids: FxHashMap<usize, String>,
    next_wid: usize,
    wordinfo: FxHashMap<usize, FxHashMap<usize, f32>>,
    docweight: FxHashMap<usize, f32>,
    docwords: FxHashMap<usize, Vec<usize>>,
    wids_dict: FxHashMap<String, (usize, f64)>,
}

impl CanopyIndex {
    pub fn new() -> Self {
        Self {
            next_wid: 1,
            ..Default::default()
        }
    }

    fn get_or_create_wid(&mut self, word: &str) -> usize {
        if let Some(w) = self.lexicon_words.get(word) {
            return *w;
        }
        let wid = self.next_wid;
        self.next_wid += 1;
        self.lexicon_words.insert(word.to_string(), wid);
        self.lexicon_wids.insert(wid, word.to_string());
        wid
    }

    fn get_frequencies(&self, wids: &[usize]) -> (FxHashMap<usize, f32>, f32) {
        let mut counts: FxHashMap<usize, i64> = FxHashMap::default();
        for w in wids {
            *counts.entry(*w).or_insert(0) += 1;
        }
        let mut wsquares = 0.0f64;
        let mut weights: FxHashMap<usize, f64> = FxHashMap::default();
        for (wid, count) in &counts {
            let w = 1.0 + (*count as f64).ln();
            wsquares += w * w;
            weights.insert(*wid, w);
        }
        let docweight = wsquares.sqrt();
        let normalized = weights
            .into_iter()
            .map(|(wid, w)| (wid, (w / docweight) as f32))
            .collect();
        (normalized, docweight as f32)
    }

    /// `CanopyIndex.initSearch`: remove stop words and precompute idf weights.
    pub fn init_search(&mut self) {
        let n = self.docweight.len() as f64;
        let threshold = (n * 0.05).max(1000.0) as i64;

        let mut stop_words = Vec::new();
        self.wids_dict.clear();

        let mut wids: Vec<usize> = self.wordinfo.keys().copied().collect();
        wids.sort_unstable();
        for wid in wids {
            let docs = &self.wordinfo[&wid];
            if docs.len() as i64 > threshold {
                stop_words.push(wid);
                continue;
            }
            let idf = (1.0 + n / docs.len() as f64).ln();
            if let Some(term) = self.lexicon_wids.get(&wid) {
                self.wids_dict.insert(term.clone(), (wid, idf));
            }
        }

        for wid in stop_words {
            if let Some(word) = self.lexicon_wids.remove(&wid) {
                self.lexicon_words.remove(&word);
            }
            self.wordinfo.remove(&wid);
        }
    }

    /// `CanopyIndex.apply`: weighted union over query terms, filtered by
    /// `qw * threshold` and sorted by descending score.
    pub fn apply(&self, query: &[String], threshold: f64) -> Vec<usize> {
        let mut l: Vec<(FxHashMap<usize, f32>, f64)> = Vec::new();
        let mut qw_sq = 0.0f64;
        for term in query {
            if let Some((wid, weight)) = self.wids_dict.get(term) {
                if let Some(docs) = self.wordinfo.get(wid) {
                    l.push((docs.clone(), *weight));
                    qw_sq += weight * weight;
                }
            }
        }
        if l.is_empty() {
            return Vec::new();
        }
        let scores = mass_weighted_union(&l);
        let qw = qw_sq.sqrt();
        // BTrees `byValue` compares against `min` narrowed to the tree's f32
        // value type.
        let min = (qw * threshold) as f32;
        let mut results: Vec<(usize, f32)> = scores
            .into_iter()
            .filter(|(_, s)| *s >= min)
            .collect();
        // byValue order: descending score, and descending key on ties.
        results.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.0.cmp(&a.0))
        });
        results.into_iter().map(|(docid, _)| docid).collect()
    }
}

/// A borrow-or-owned partial weighted union.
enum Partial<'a> {
    Ref(&'a FxHashMap<usize, f32>),
    Owned(FxHashMap<usize, f32>),
}

impl Partial<'_> {
    fn len(&self) -> usize {
        match self {
            Partial::Ref(m) => m.len(),
            Partial::Owned(m) => m.len(),
        }
    }

    fn get(&self, d: usize) -> Option<f32> {
        match self {
            Partial::Ref(m) => m.get(&d).copied(),
            Partial::Owned(m) => m.get(&d).copied(),
        }
    }

    fn for_each(&self, mut f: impl FnMut(usize, f32)) {
        match self {
            Partial::Ref(m) => {
                for (d, v) in *m {
                    f(*d, *v);
                }
            }
            Partial::Owned(m) => {
                for (d, v) in m {
                    f(*d, *v);
                }
            }
        }
    }
}

/// Port of `zope.index.text.setops.mass_weightedUnion` using f32 value
/// arithmetic, exactly like an `IFBTree`.  Borrows the expensive input maps
/// and only allocates for merged results.
fn mass_weighted_union(l: &[(FxHashMap<usize, f32>, f64)]) -> FxHashMap<usize, f32> {
    if l.is_empty() {
        return FxHashMap::default();
    }
    if l.len() == 1 {
        let (m, w) = &l[0];
        return m.iter().map(|(d, v)| (*d, (*w * *v as f64) as f32)).collect();
    }

    // Balanced union: repeatedly merge the two smallest mappings.
    let mut queue: Vec<(Partial, f64, usize)> = l
        .iter()
        .enumerate()
        .map(|(i, (m, w))| (Partial::Ref(m), *w, i))
        .collect();
    let mut seq = queue.len();

    while queue.len() > 1 {
        let mut order: Vec<usize> = (0..queue.len()).collect();
        order.sort_by_key(|&i| (queue[i].0.len(), queue[i].2));
        let (i0, i1) = (order[0], order[1]);
        let wa = queue[i0].1;
        let wb = queue[i1].1;

        let mut z: FxHashMap<usize, f32> = FxHashMap::with_capacity_and_hasher(queue[i0].0.len().max(queue[i1].0.len()), Default::default());
        {
            let a = &queue[i0].0;
            let b = &queue[i1].0;
            a.for_each(|d, va| {
                let vb = b.get(d).unwrap_or(0.0);
                z.insert(d, (wa * va as f64 + wb * vb as f64) as f32);
            });
            b.for_each(|d, vb| {
                z.entry(d).or_insert((wb * vb as f64) as f32);
            });
        }

        let (hi, lo) = if i0 > i1 { (i0, i1) } else { (i1, i0) };
        queue.swap_remove(hi);
        queue.swap_remove(lo);
        queue.push((Partial::Owned(z), 1.0, seq));
        seq += 1;
    }

    match queue.pop().unwrap().0 {
        Partial::Ref(m) => m.clone(),
        Partial::Owned(m) => m,
    }
}

#[derive(Debug)]
pub struct TfIdfIndex {
    index: CanopyIndex,
    doc_to_id: FxHashMap<Doc, usize>,
    next_id: usize,
}

impl Default for TfIdfIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl TfIdfIndex {
    pub fn new() -> Self {
        Self {
            index: CanopyIndex::new(),
            doc_to_id: FxHashMap::default(),
            next_id: 1,
        }
    }
}

impl Index for TfIdfIndex {
    fn index_doc(&mut self, doc: &Doc) {
        if self.doc_to_id.contains_key(doc) {
            return;
        }
        let id = self.get_or_create_id(doc);
        let terms = doc.terms().to_vec();
        let wids: Vec<usize> = terms
            .iter()
            .map(|t| self.index.get_or_create_wid(t))
            .collect();
        let (wid2weight, docweight) = self.index.get_frequencies(&wids);
        for (wid, w) in wid2weight {
            self.index
                .wordinfo
                .entry(wid)
                .or_default()
                .insert(id, w);
        }
        self.index.docweight.insert(id, docweight);
        self.index.docwords.insert(id, wids);
    }

    fn unindex_doc(&mut self, doc: &Doc) {
        let Some(id) = self.doc_to_id.remove(doc) else {
            return;
        };
        if let Some(wids) = self.index.docwords.remove(&id) {
            let unique: std::collections::BTreeSet<usize> = wids.into_iter().collect();
            for wid in unique {
                if let Some(docs) = self.index.wordinfo.get_mut(&wid) {
                    docs.remove(&id);
                    if docs.is_empty() {
                        self.index.wordinfo.remove(&wid);
                        if let Some(word) = self.index.lexicon_wids.remove(&wid) {
                            self.index.lexicon_words.remove(&word);
                        }
                    }
                }
            }
        }
        self.index.docweight.remove(&id);
        self.init_search();
    }

    fn init_search(&mut self) {
        self.index.init_search();
    }

    fn search(&self, doc: &Doc, threshold: f64) -> Vec<usize> {
        let query = doc.terms();
        if query.is_empty() {
            return Vec::new();
        }
        self.index.apply(query, threshold)
    }

    fn get_or_create_id(&mut self, doc: &Doc) -> usize {
        if let Some(id) = self.doc_to_id.get(doc) {
            return *id;
        }
        let id = self.next_id;
        self.next_id += 1;
        self.doc_to_id.insert(doc.clone(), id);
        id
    }

    fn get_id(&self, doc: &Doc) -> Option<usize> {
        self.doc_to_id.get(doc).copied()
    }

    fn num_docs(&self) -> usize {
        self.index.docweight.len()
    }
}

// ---------------------------------------------------------------------------
// Levenshtein index (BK-tree)
// ---------------------------------------------------------------------------

/// Plain Levenshtein distance over Unicode scalar values.
pub fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for i in 1..=a.len() {
        cur[0] = i;
        for j in 1..=b.len() {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

#[derive(Debug, Default)]
struct BkNode {
    word: String,
    children: BTreeMap<usize, BkNode>,
}

impl BkNode {
    fn insert(&mut self, word: &str) {
        let d = levenshtein(&self.word, word);
        if d == 0 {
            return;
        }
        match self.children.get_mut(&d) {
            Some(child) => child.insert(word),
            None => {
                self.children.insert(
                    d,
                    BkNode {
                        word: word.to_string(),
                        children: BTreeMap::new(),
                    },
                );
            }
        }
    }

    fn search(&self, query: &str, max: usize, out: &mut Vec<String>) {
        // BK-tree navigation requires the exact distance to the node.
        let d = levenshtein(&self.word, query);
        if d <= max {
            out.push(self.word.clone());
        }
        if self.children.is_empty() {
            return;
        }
        let low = d.saturating_sub(max);
        let high = d.saturating_add(max);
        for (dist, child) in self.children.range(low..=high) {
            let _ = dist;
            child.search(query, max, out);
        }
    }
}

#[derive(Debug)]
pub struct LevenshteinIndex {
    root: Option<BkNode>,
    doc_to_id: FxHashMap<String, usize>,
    next_id: usize,
}

impl Default for LevenshteinIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl LevenshteinIndex {
    pub fn new() -> Self {
        Self {
            root: None,
            doc_to_id: FxHashMap::default(),
            next_id: 1,
        }
    }

    fn rebuild(&mut self) {
        let mut root = BkNode {
            word: String::new(),
            children: BTreeMap::new(),
        };
        let mut first = true;
        let words: Vec<String> = self.doc_to_id.keys().cloned().collect();
        for w in words {
            if first {
                root = BkNode {
                    word: w,
                    children: BTreeMap::new(),
                };
                first = false;
            } else {
                root.insert(&w);
            }
        }
        self.root = if first { None } else { Some(root) };
    }
}

impl Index for LevenshteinIndex {
    fn index_doc(&mut self, doc: &Doc) {
        let Doc::Text(s) = doc else {
            return;
        };
        if self.doc_to_id.contains_key(s) {
            return;
        }
        self.get_or_create_id(doc);
        match &mut self.root {
            Some(root) => root.insert(s),
            None => {
                self.root = Some(BkNode {
                    word: s.clone(),
                    children: BTreeMap::new(),
                })
            }
        }
    }

    fn unindex_doc(&mut self, doc: &Doc) {
        let Doc::Text(s) = doc else {
            return;
        };
        self.doc_to_id.remove(s);
        self.rebuild();
    }

    fn init_search(&mut self) {}

    fn search(&self, doc: &Doc, threshold: f64) -> Vec<usize> {
        let Doc::Text(query) = doc else {
            return Vec::new();
        };
        let max = threshold.max(0.0) as usize;
        let mut matched = Vec::new();
        if let Some(root) = &self.root {
            root.search(query, max, &mut matched);
        }
        matched
            .into_iter()
            .filter_map(|w| self.doc_to_id.get(&w).copied())
            .collect()
    }

    fn get_or_create_id(&mut self, doc: &Doc) -> usize {
        let key = match doc {
            Doc::Text(s) => s.clone(),
            Doc::Terms(t) => t.join(" "),
        };
        if let Some(id) = self.doc_to_id.get(&key) {
            return *id;
        }
        let id = self.next_id;
        self.next_id += 1;
        self.doc_to_id.insert(key, id);
        id
    }

    fn get_id(&self, doc: &Doc) -> Option<usize> {
        let key = match doc {
            Doc::Text(s) => s.clone(),
            Doc::Terms(t) => t.join(" "),
        };
        self.doc_to_id.get(&key).copied()
    }

    fn num_docs(&self) -> usize {
        self.doc_to_id.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levenshtein_basic() {
        assert_eq!(levenshtein("kitten", "sitting"), 3);
        assert_eq!(levenshtein("", "abc"), 3);
        assert_eq!(levenshtein("abc", "abc"), 0);
    }

    #[test]
    fn bk_search() {
        let mut idx = LevenshteinIndex::new();
        for w in ["thomas", "tomas", "thoms", "bob"] {
            idx.index_doc(&Doc::Text(w.to_string()));
        }
        let got = idx.search(&Doc::Text("thomas".into()), 1.0);
        let mut got = got;
        got.sort_unstable();
        assert_eq!(got, vec![1, 2, 3]);
    }

    #[test]
    fn tfidf_basic() {
        let mut idx = TfIdfIndex::new();
        idx.index_doc(&Doc::Terms(vec![
            "AND".into(),
            "OR".into(),
            "EOF".into(),
            "NOT".into(),
        ]));
        idx.init_search();
        assert_eq!(
            idx.search(
                &Doc::Terms(vec!["AND".into(), "OR".into(), "EOF".into(), "NOT".into()]),
                0.0
            )[0],
            1
        );
    }
}
