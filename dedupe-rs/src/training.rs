//! Blocking-rule learning.  Ported from `dedupe/training.py`.

use std::collections::{BTreeSet, HashSet};

use indexmap::IndexMap;
use rand::seq::IndexedRandom;
use rand::Rng;

use crate::blocking::Fingerprinter;
use crate::branch_and_bound::{self, Cover};
use crate::predicates::PredRef;
use crate::value::{Data, Record, RecordId};

/// Predicate to covered example-pair indices.
pub type MatchCover = Cover;
/// Predicate to covered record pairs.
pub type ComparisonCover = IndexMap<PredRef, BTreeSet<(RecordId, RecordId)>>;

/// The kind of candidate conjunctions to generate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateTypes {
    Simple,
    RandomForest,
}

pub trait BlockLearner {
    fn blocker(&self) -> &Fingerprinter;
    fn comparison_cover(&self) -> &ComparisonCover;

    /// Compute the examples covered by each predicate.
    fn cover(
        &self,
        pairs: &[(Record, Record)],
        index_predicates: bool,
    ) -> MatchCover {
        let blocker = self.blocker();
        let mut predicate_cover: MatchCover = IndexMap::new();

        let predicates: Vec<PredRef> = if index_predicates {
            blocker.predicates.clone()
        } else {
            blocker
                .predicates
                .iter()
                .filter(|p| !p.0.is_index())
                .cloned()
                .collect()
        };

        for predicate in predicates {
            let mut coverage: BTreeSet<usize> = BTreeSet::new();
            for (i, (record_1, record_2)) in pairs.iter().enumerate() {
                let keys_1 = predicate.call(record_1, false).unwrap_or_default();
                let keys_2 = predicate.call(record_2, true).unwrap_or_default();
                if !keys_1.is_disjoint(&keys_2) {
                    coverage.insert(i);
                }
            }
            if !coverage.is_empty() {
                predicate_cover.insert(predicate, coverage);
            }
        }

        predicate_cover
    }

    /// Learn a set of blocking predicates.
    fn learn(
        &self,
        matches: &[(Record, Record)],
        recall: f64,
        index_predicates: bool,
        candidate_types: CandidateTypes,
        rng: &mut impl Rng,
    ) -> Vec<PredRef> {
        assert!(
            !matches.is_empty(),
            "You must supply at least one pair of matching records to learn blocking rules."
        );

        let comparison_cover = self.comparison_cover();
        let mut match_cover = self.cover(matches, index_predicates);

        // Drop predicates that are not in the comparison cover.
        let keep: HashSet<PredRef> = comparison_cover.keys().cloned().collect();
        match_cover.retain(|k, _| keep.contains(k));

        let mut coverable_dupes: BTreeSet<usize> = BTreeSet::new();
        for cover in match_cover.values() {
            coverable_dupes.extend(cover.iter().copied());
        }

        let mut target_cover = (recall * matches.len() as f64) as usize;
        if coverable_dupes.len() < target_cover {
            target_cover = coverable_dupes.len();
        }

        let mut candidate_cover = simple_candidates(&match_cover, comparison_cover);

        if candidate_types == CandidateTypes::RandomForest {
            let k = (matches.len() as f64).log10().floor().max(1.0) as usize;
            if k > 1 {
                let rf = random_forest_candidates(
                    &match_cover,
                    comparison_cover,
                    k,
                    rng,
                );
                for (pred, cover) in rf {
                    candidate_cover.insert(pred, cover);
                }
            }
        }

        branch_and_bound::search(&candidate_cover, target_cover, 2500)
    }
}

pub struct DedupeBlockLearner {
    pub blocker: Fingerprinter,
    pub comparison_cover: ComparisonCover,
}

impl DedupeBlockLearner {
    pub fn new(predicates: Vec<PredRef>, sampled_records: &Data, data: &Data) -> Self {
        let blocker = Fingerprinter::new(predicates);
        blocker.index_all(data);
        let comparison_cover = Self::covered_pairs(&blocker, sampled_records);
        Self {
            blocker,
            comparison_cover,
        }
    }

    pub fn covered_pairs(blocker: &Fingerprinter, records: &Data) -> ComparisonCover {
        let mut cover: ComparisonCover = IndexMap::new();
        let n_records = records.len();

        for predicate in &blocker.predicates {
            let mut pred_cover: IndexMap<String, BTreeSet<RecordId>> = IndexMap::new();
            for (id, record) in records {
                let blocks = predicate.call(record, false).unwrap_or_default();
                for block in blocks {
                    pred_cover.entry(block).or_default().insert(id.clone());
                }
            }

            if pred_cover.is_empty() {
                continue;
            }
            let max_cover = pred_cover.values().map(|v| v.len()).max().unwrap_or(0);
            if max_cover == n_records {
                continue;
            }

            let mut pairs: BTreeSet<(RecordId, RecordId)> = BTreeSet::new();
            for block in pred_cover.values() {
                let sorted: Vec<&RecordId> = block.iter().collect();
                for i in 0..sorted.len() {
                    for j in (i + 1)..sorted.len() {
                        pairs.insert((sorted[i].clone(), sorted[j].clone()));
                    }
                }
            }
            if !pairs.is_empty() {
                cover.insert(predicate.clone(), pairs);
            }
        }

        cover
    }
}

impl BlockLearner for DedupeBlockLearner {
    fn blocker(&self) -> &Fingerprinter {
        &self.blocker
    }
    fn comparison_cover(&self) -> &ComparisonCover {
        &self.comparison_cover
    }
}

pub struct RecordLinkBlockLearner {
    pub blocker: Fingerprinter,
    pub comparison_cover: ComparisonCover,
}

impl RecordLinkBlockLearner {
    pub fn new(
        predicates: Vec<PredRef>,
        sampled_records_1: &Data,
        sampled_records_2: &Data,
        data_2: &Data,
    ) -> Self {
        let blocker = Fingerprinter::new(predicates);
        blocker.index_all(data_2);
        let comparison_cover = Self::covered_pairs(
            &blocker,
            sampled_records_1,
            sampled_records_2,
        );
        Self {
            blocker,
            comparison_cover,
        }
    }

    pub fn covered_pairs(
        blocker: &Fingerprinter,
        records_1: &Data,
        records_2: &Data,
    ) -> ComparisonCover {
        let n_records_1 = records_1.len();
        let n_records_2 = records_2.len();
        let mut pair_cover: ComparisonCover = IndexMap::new();

        for predicate in &blocker.predicates {
            // block -> (set of ids from records_1, set from records_2)
            let mut blocks: IndexMap<String, (BTreeSet<RecordId>, BTreeSet<RecordId>)> =
                IndexMap::new();
            for (id, record) in records_2 {
                let keys = predicate.call(record, true).unwrap_or_default();
                for block in keys {
                    blocks.entry(block).or_default().1.insert(id.clone());
                }
            }
            let current_blocks: HashSet<String> = blocks.keys().cloned().collect();
            for (id, record) in records_1 {
                let keys = predicate.call(record, false).unwrap_or_default();
                for block in keys {
                    if current_blocks.contains(&block) {
                        blocks.entry(block).or_default().0.insert(id.clone());
                    }
                }
            }

            let dominated = blocks.values().any(|(a, b)| {
                a.len() == n_records_1 && b.len() == n_records_2
            });
            if dominated {
                continue;
            }

            let mut pairs: BTreeSet<(RecordId, RecordId)> = BTreeSet::new();
            for (a, b) in blocks.values() {
                for id_a in a {
                    for id_b in b {
                        pairs.insert((id_a.clone(), id_b.clone()));
                    }
                }
            }
            if !pairs.is_empty() {
                pair_cover.insert(predicate.clone(), pairs);
            }
        }

        pair_cover
    }
}

impl BlockLearner for RecordLinkBlockLearner {
    fn blocker(&self) -> &Fingerprinter {
        &self.blocker
    }
    fn comparison_cover(&self) -> &ComparisonCover {
        &self.comparison_cover
    }
}

/// Set `cover_count` on each candidate and copy its coverage.
pub fn simple_candidates(
    match_cover: &MatchCover,
    comparison_cover: &ComparisonCover,
) -> MatchCover {
    let mut candidates = MatchCover::new();
    for (predicate, coverage) in match_cover {
        predicate
            .0
            .set_cover_count(comparison_cover.get(predicate).map(|c| c.len()).unwrap_or(0));
        candidates.insert(predicate.clone(), coverage.clone());
    }
    candidates
}

/// Resamples match coverage (Python's `Resampler`).
pub struct Resampler {
    replacements: std::collections::HashMap<usize, Vec<usize>>,
}

impl Resampler {
    pub fn new(sequence: &[usize], rng: &mut impl Rng) -> Self {
        let sampled: Vec<usize> = (0..sequence.len())
            .map(|_| sequence[rng.random_range(0..sequence.len())])
            .collect();
        let mut counts: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for s in sampled {
            *counts.entry(s).or_insert(0) += 1;
        }
        let mut max_value = sequence.len() + 1;
        let mut replacements = std::collections::HashMap::new();
        for (k, v) in counts {
            let mut vals = vec![v];
            if v > 1 {
                for _ in 0..(v - 1) {
                    vals.push(max_value);
                    max_value += 1;
                }
            }
            replacements.insert(k, vals);
        }
        Self { replacements }
    }

    pub fn resample(&self, iterable: impl IntoIterator<Item = usize>) -> BTreeSet<usize> {
        let mut out = BTreeSet::new();
        for k in iterable {
            if let Some(vals) = self.replacements.get(&k) {
                out.extend(vals.iter().copied());
            }
        }
        out
    }
}

/// Generate random-forest conjunction candidates.
pub fn random_forest_candidates(
    match_cover: &MatchCover,
    comparison_cover: &ComparisonCover,
    k_conj: usize,
    rng: &mut impl Rng,
) -> MatchCover {
    let predicates: Vec<PredRef> = match_cover.keys().cloned().collect();
    if predicates.is_empty() {
        return MatchCover::new();
    }
    let mut matches: Vec<usize> = Vec::new();
    for cover in match_cover.values() {
        matches.extend(cover.iter().copied());
    }
    let matches: BTreeSet<usize> = matches.into_iter().collect();
    let matches_vec: Vec<usize> = matches.iter().copied().collect();

    let pred_sample_size = ((predicates.len() as f64).sqrt() as usize).max(5).min(predicates.len());
    let mut candidates = MatchCover::new();

    let n_samples = 5000;
    for _ in 0..n_samples {
        let sample_predicates: Vec<PredRef> =
            predicates.choose_multiple(rng, pred_sample_size).cloned().collect();
        let mut sample_predicates = sample_predicates;
        let resampler = Resampler::new(&matches_vec, rng);
        let sample_match_cover: IndexMap<PredRef, BTreeSet<usize>> = match_cover
            .iter()
            .map(|(p, pairs)| (p.clone(), resampler.resample(pairs.iter().copied())))
            .collect();

        let mut candidate: Option<PredRef> = None;
        let mut covered_comparisons: Option<BTreeSet<(RecordId, RecordId)>> = None;
        let mut covered_matches: Option<BTreeSet<usize>> = None;
        let mut covered_sample_matches: Option<BTreeSet<usize>> = None;

        for _ in 0..k_conj {
            if sample_predicates.is_empty() {
                break;
            }
            let score = |predicate: &PredRef| -> f64 {
                let s = covered_sample_matches
                    .as_ref()
                    .map(|c| c.intersection(&sample_match_cover[predicate]).count())
                    .unwrap_or(sample_match_cover[predicate].len());
                let cmp = covered_comparisons
                    .as_ref()
                    .map(|c| c.intersection(&comparison_cover[predicate]).count())
                    .unwrap_or(comparison_cover[predicate].len());
                if cmp == 0 {
                    0.0
                } else {
                    s as f64 / cmp as f64
                }
            };
            let next = sample_predicates
                .iter()
                .cloned()
                .reduce(|acc, p| if score(&p) > score(&acc) { p } else { acc })
                .unwrap();
            candidate = Some(match &candidate {
                None => next.clone(),
                Some(c) => {
                    // Python's `candidate += next_predicate` builds a compound.
                    let mut parts = c.0.compound_parts().map(|p| p.to_vec()).unwrap_or_else(|| vec![c.clone()]);
                    parts.push(next.clone());
                    PredRef::new(crate::predicates::CompoundPredicate::new(parts))
                }
            });

            let cmp = comparison_cover[&next].clone();
            covered_comparisons = Some(match covered_comparisons {
                None => cmp,
                Some(c) => c.intersection(&cmp).cloned().collect(),
            });
            let cand = candidate.clone().unwrap();
            cand.0.set_cover_count(covered_comparisons.as_ref().unwrap().len());

            let m = match_cover[&next].clone();
            covered_matches = Some(match covered_matches {
                None => m,
                Some(c) => c.intersection(&m).cloned().collect(),
            });
            candidates.insert(cand, covered_matches.clone().unwrap());

            let sm = sample_match_cover[&next].clone();
            covered_sample_matches = Some(match covered_sample_matches {
                None => sm,
                Some(c) => c.intersection(&sm).cloned().collect(),
            });

            sample_predicates.retain(|p| p != &next);
        }
    }

    candidates
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datamodel::DataModel;
    use crate::variables::{self, VariableDef};

    fn record(name: &str, age: &str) -> Record {
        let mut r = Record::new();
        r.insert("name".to_string(), crate::value::Value::str(name));
        r.insert("age".to_string(), crate::value::Value::str(age));
        r
    }

    #[test]
    fn dedupe_coverage_matches_python() {
        // Mirrors tests/test_training.py::TrainingTest::test_dedupe_coverage
        let dm = DataModel::new(vec![VariableDef::Field(variables::string("name", false))]).unwrap();
        let match_pairs: Vec<(Record, Record)> = vec![
            (record("Bob", "50"), record("Bob", "75")),
            (record("Meredith", "40"), record("Sue", "10")),
        ];
        let distinct_pairs: Vec<(Record, Record)> = vec![
            (record("Jimmy", "20"), record("Jimbo", "21")),
            (record("Willy", "35"), record("William", "35")),
            (record("William", "36"), record("William", "35")),
        ];
        let training: Vec<(Record, Record)> = match_pairs
            .iter()
            .chain(distinct_pairs.iter())
            .cloned()
            .collect();

        let mut training_records: Vec<Record> = Vec::new();
        for (a, b) in &training {
            for r in [a, b] {
                if !training_records.contains(r) {
                    training_records.push(r.clone());
                }
            }
        }
        let mut data = Data::new();
        for (i, r) in training_records.iter().enumerate() {
            data.insert(RecordId::Int(i as i64), r.clone());
        }

        let learner = DedupeBlockLearner::new(dm.predicates(), &data, &data);
        let cover = learner.cover(&training, true);
        let reprs: std::collections::BTreeSet<String> =
            cover.keys().map(|p| p.0.repr()).collect();

        let expected = [
            "SimplePredicate: (tokenFieldPredicate, name)",
            "SimplePredicate: (commonSixGram, name)",
            "TfidfTextCanopyPredicate: (0.4, name)",
            "SimplePredicate: (sortedAcronym, name)",
            "SimplePredicate: (sameThreeCharStartPredicate, name)",
            "TfidfTextCanopyPredicate: (0.2, name)",
            "SimplePredicate: (sameFiveCharStartPredicate, name)",
            "TfidfTextCanopyPredicate: (0.6, name)",
            "SimplePredicate: (wholeFieldPredicate, name)",
            "TfidfTextCanopyPredicate: (0.8, name)",
            "SimplePredicate: (commonFourGram, name)",
            "SimplePredicate: (firstTokenPredicate, name)",
            "SimplePredicate: (sameSevenCharStartPredicate, name)",
        ];
        let missing: Vec<&str> = expected
            .iter()
            .copied()
            .filter(|e| !reprs.contains(*e))
            .collect();
        assert!(missing.is_empty(), "missing predicates: {missing:?}");
    }
}
