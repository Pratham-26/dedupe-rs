//! Active-learning disagreement learners.  Ported from `dedupe/labeler.py`.

use std::collections::HashSet;

use rand::seq::IndexedRandom;
use rand::{Rng, SeedableRng};

use crate::core;
use crate::datamodel::DistanceCalculator;
use crate::logistic::{Classifier, LogisticRegression};
use crate::predicates::PredRef;
use crate::training::{self, BlockLearner as _, CandidateTypes};
use crate::value::{Data, Record, RecordId};

/// A learner that can score the current candidate pairs.
pub trait CandidateLearner {
    fn candidates(&self) -> &[(Record, Record)];
    /// A single score column for each candidate.
    fn candidate_scores(&self) -> Vec<f64>;
    fn remove(&mut self, index: usize);
    fn fit(&mut self, pairs: &[(Record, Record)], y: &[usize]);
}

fn verify_fit_args(pairs: &[(Record, Record)], y: &[usize]) -> Result<(), String> {
    if pairs.is_empty() {
        return Err("pairs must have length of at least 1".to_string());
    }
    if pairs.len() != y.len() {
        return Err(format!(
            "pairs and y must be same length. Got {} and {}",
            pairs.len(),
            y.len()
        ));
    }
    Ok(())
}

/// A match learner that owns the distance calculator so it can featurize
/// freshly labeled pairs.
pub struct DistanceMatchLearner {
    calculator: DistanceCalculator,
    candidates: Vec<(Record, Record)>,
    features: Vec<Vec<f64>>,
    classifier: LogisticRegression,
    fitted: bool,
}

impl DistanceMatchLearner {
    pub fn new(calculator: DistanceCalculator, candidates: Vec<(Record, Record)>) -> Self {
        let features = featurize(&calculator, &candidates);
        Self {
            calculator,
            candidates,
            features,
            classifier: LogisticRegression::default(),
            fitted: false,
        }
    }
}

fn featurize(calculator: &DistanceCalculator, pairs: &[(Record, Record)]) -> Vec<Vec<f64>> {
    pairs
        .iter()
        .map(|(a, b)| {
            calculator
                .distance_row(a, b)
                .into_iter()
                .map(|x| x as f64)
                .collect()
        })
        .collect()
}

impl CandidateLearner for DistanceMatchLearner {
    fn candidates(&self) -> &[(Record, Record)] {
        &self.candidates
    }

    fn candidate_scores(&self) -> Vec<f64> {
        if !self.fitted {
            panic!("Must call fit() before candidate_scores()");
        }
        self.classifier.predict_proba(&self.features)
    }

    fn remove(&mut self, index: usize) {
        self.candidates.remove(index);
        self.features.remove(index);
    }

    fn fit(&mut self, pairs: &[(Record, Record)], y: &[usize]) {
        if verify_fit_args(pairs, y).is_err() {
            return;
        }
        let features = featurize(&self.calculator, pairs);
        self.classifier.fit(&features, y);
        self.fitted = true;
    }
}

/// A learner based on blocking-rule coverage.
pub struct BlockingLearner {
    pub block_learner: TrainingBlockLearner,
    pub current_predicates: Vec<PredRef>,
    cached_scores: Option<Vec<f64>>,
    old_dupes: Vec<(Record, Record)>,
    pub candidates: Vec<(Record, Record)>,
    pub rng_seed: u64,
    fitted: bool,
}

/// Either a dedupe or record-linkage training block learner.
pub enum TrainingBlockLearner {
    Dedupe(training::DedupeBlockLearner),
    RecordLink(training::RecordLinkBlockLearner),
}

impl TrainingBlockLearner {
    fn learn(
        &self,
        matches: &[(Record, Record)],
        recall: f64,
        index_predicates: bool,
        candidate_types: CandidateTypes,
        rng: &mut impl Rng,
    ) -> Vec<PredRef> {
        match self {
            TrainingBlockLearner::Dedupe(l) => {
                training::BlockLearner::learn(l, matches, recall, index_predicates, candidate_types, rng)
            }
            TrainingBlockLearner::RecordLink(l) => {
                training::BlockLearner::learn(l, matches, recall, index_predicates, candidate_types, rng)
            }
        }
    }
}

impl BlockingLearner {
    pub fn new(block_learner: TrainingBlockLearner, candidates: Vec<(Record, Record)>, rng_seed: u64) -> Self {
        Self {
            block_learner,
            current_predicates: Vec::new(),
            cached_scores: None,
            old_dupes: Vec::new(),
            candidates,
            rng_seed,
            fitted: false,
        }
    }

    pub fn learn_predicates(
        &self,
        dupes: &[(Record, Record)],
        recall: f64,
        index_predicates: bool,
    ) -> Vec<PredRef> {
        let mut rng = crate::logistic::seeded_rng(self.rng_seed);
        self.block_learner.learn(
            dupes,
            recall,
            index_predicates,
            CandidateTypes::RandomForest,
            &mut rng,
        )
    }

    fn predict(&self, pairs: &[(Record, Record)]) -> Vec<usize> {
        let mut labels = Vec::with_capacity(pairs.len());
        for (record_1, record_2) in pairs {
            let mut matched = 0;
            for predicate in &self.current_predicates {
                let keys_2 = predicate.call(record_2, true).unwrap_or_default();
                let keys_1 = predicate.call(record_1, false).unwrap_or_default();
                if !keys_1.is_disjoint(&keys_2) {
                    matched = 1;
                    break;
                }
            }
            labels.push(matched);
        }
        labels
    }
}

impl CandidateLearner for BlockingLearner {
    fn candidates(&self) -> &[(Record, Record)] {
        &self.candidates
    }

    fn candidate_scores(&self) -> Vec<f64> {
        if !self.fitted {
            panic!("Must call fit() before candidate_scores()");
        }
        if let Some(cached) = &self.cached_scores {
            return cached.clone();
        }
        self.predict(&self.candidates)
            .into_iter()
            .map(|x| x as f64)
            .collect()
    }

    fn remove(&mut self, index: usize) {
        self.candidates.remove(index);
        if let Some(cached) = &mut self.cached_scores {
            cached.remove(index);
        }
    }

    fn fit(&mut self, pairs: &[(Record, Record)], y: &[usize]) {
        if verify_fit_args(pairs, y).is_err() {
            return;
        }
        let dupes: Vec<(Record, Record)> = y
            .iter()
            .zip(pairs.iter())
            .filter(|(label, _)| **label == 1)
            .map(|(_, pair)| pair.clone())
            .collect();

        let new_dupes: Vec<(Record, Record)> = dupes
            .iter()
            .filter(|pair| !self.old_dupes.contains(pair))
            .cloned()
            .collect();
        let predictions = self.predict(&new_dupes);
        let new_uncovered = !predictions.iter().all(|x| *x == 1);

        if new_uncovered {
            let mut rng = crate::logistic::seeded_rng(self.rng_seed);
            self.current_predicates = self.block_learner.learn(
                &dupes,
                1.0,
                true,
                CandidateTypes::Simple,
                &mut rng,
            );
            self.cached_scores = None;
            self.old_dupes = dupes;
        }
        self.fitted = true;
    }
}

/// A disagreement learner over two candidate learners.
pub struct DisagreementLearner {
    pub y: Vec<usize>,
    pub pairs: Vec<(Record, Record)>,
    pub matcher: DistanceMatchLearner,
    pub blocker: BlockingLearner,
    rng_state: rand::rngs::StdRng,
}

impl DisagreementLearner {
    pub fn new(
        matcher: DistanceMatchLearner,
        blocker: BlockingLearner,
        rng_seed: u64,
    ) -> Self {
        Self {
            y: Vec::new(),
            pairs: Vec::new(),
            matcher,
            blocker,
            rng_state: rand::rngs::StdRng::seed_from_u64(rng_seed),
        }
    }

    pub fn len(&self) -> usize {
        self.matcher.candidates().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Pick the next most informative pair to label.
    pub fn pop(&mut self) -> Result<(Record, Record), String> {
        let n_candidates = self.matcher.candidates().len();
        if n_candidates == 0 {
            return Err("No more unlabeled examples to label".to_string());
        }

        let p0 = self.matcher.candidate_scores();
        let p1 = self.blocker.candidate_scores();

        let uncovered: Vec<bool> = (0..n_candidates)
            .map(|i| ((p0[i] > 0.5) != (p1[i] > 0.5)) && p1[i] == 0.0)
            .collect();

        let uncertain_index = if uncovered.iter().any(|x| *x) {
            let mut weights: Vec<f64> = (0..n_candidates)
                .map(|i| if uncovered[i] { p0[i] } else { 0.0 })
                .collect();
            let sum: f64 = weights.iter().sum();
            for w in weights.iter_mut() {
                *w /= sum;
            }
            weighted_choice(&mut self.rng_state, &weights)
        } else if p1.contains(&1.0) {
            let covered: Vec<usize> = (0..n_candidates).filter(|&i| p1[i] == 1.0).collect();
            let target: f64 = self.rng_state.random::<f64>();
            *covered
                .iter()
                .min_by(|&&i, &&j| {
                    (p0[i] - target)
                        .abs()
                        .partial_cmp(&(p0[j] - target).abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(i.cmp(&j))
                })
                .unwrap()
        } else {
            let mut weights: Vec<f64> = (0..n_candidates)
                .map(|i| {
                    let mean = (p0[i] + p1[i]) / 2.0;
                    let var = ((p0[i] - mean).powi(2) + (p1[i] - mean).powi(2)) / 2.0;
                    var.sqrt()
                })
                .collect();
            let sum: f64 = weights.iter().sum();
            if sum == 0.0 {
                for w in weights.iter_mut() {
                    *w = 1.0 / n_candidates as f64;
                }
            } else {
                for w in weights.iter_mut() {
                    *w /= sum;
                }
            }
            weighted_choice(&mut self.rng_state, &weights)
        };

        let pair = self.matcher.candidates()[uncertain_index].clone();
        self.remove(uncertain_index);
        Ok(pair)
    }

    fn remove(&mut self, index: usize) {
        self.matcher.remove(index);
        self.blocker.remove(index);
    }

    pub fn mark(&mut self, pairs: &[(Record, Record)], y: &[usize]) {
        self.y.extend_from_slice(y);
        self.pairs.extend_from_slice(pairs);
        let pairs = self.pairs.clone();
        let y = self.y.clone();
        self.matcher.fit(&pairs, &y);
        self.blocker.fit(&pairs, &y);
    }

    pub fn learn_predicates(&self, recall: f64, index_predicates: bool) -> Vec<PredRef> {
        let dupes: Vec<(Record, Record)> = self
            .y
            .iter()
            .zip(self.pairs.iter())
            .filter(|(label, _)| **label == 1)
            .map(|(_, pair)| pair.clone())
            .collect();
        self.blocker.learn_predicates(&dupes, recall, index_predicates)
    }
}

fn weighted_choice(rng: &mut impl Rng, weights: &[f64]) -> usize {
    let total: f64 = weights.iter().sum();
    let mut r: f64 = rng.random::<f64>() * total;
    for (i, w) in weights.iter().enumerate() {
        if r < *w {
            return i;
        }
        r -= *w;
    }
    weights.len() - 1
}

/// Sample records from a dataset (Python's `sample_records`).
pub fn sample_records(data: &Data, sample_size: usize, rng: &mut impl Rng) -> Data {
    let mut out = Data::new();
    if data.len() <= sample_size {
        for (k, v) in data {
            out.insert(k.clone(), v.clone());
        }
        return out;
    }
    let keys: Vec<RecordId> = data.keys().cloned().collect();
    let sampled: Vec<&RecordId> = keys.choose_multiple(rng, sample_size).collect();
    for k in sampled {
        out.insert(k.clone(), data[k].clone());
    }
    out
}

/// Filter canopy / non-canopy index predicates.
pub fn filter_canopy_predicates(predicates: &[PredRef], canopies: bool) -> Vec<PredRef> {
    let mut out = Vec::new();
    for predicate in predicates {
        if predicate.0.is_index() {
            if predicate.0.is_canopy() == canopies {
                out.push(predicate.clone());
            }
        } else {
            out.push(predicate.clone());
        }
    }
    out
}

/// Sample weighted record-pair indices (Python's `BlockLearner._sample_indices`).
pub fn sample_indices(
    weights: &[(RecordId, RecordId, f64)],
    sample_size: usize,
    max_cover: usize,
    rng: &mut impl Rng,
) -> Vec<(RecordId, RecordId)> {
    let mut map: indexmap::IndexMap<(RecordId, RecordId), f64> = indexmap::IndexMap::new();
    for (a, b, w) in weights {
        if *w < max_cover as f64 {
            *map.entry((a.clone(), b.clone())).or_insert(0.0) += 1.0 / *w;
        }
    }
    if sample_size < map.len() {
        let keys: Vec<(RecordId, RecordId)> = map.keys().cloned().collect();
        let vals: Vec<f64> = map.values().copied().collect();
        let sum: f64 = vals.iter().sum();
        let probs: Vec<f64> = vals.iter().map(|v| v / sum).collect();
        // Weighted sampling without replacement (Efraimidis-Spirakis).
        let mut keyed: Vec<(f64, usize)> = (0..keys.len())
            .map(|i| {
                let u: f64 = rng.random::<f64>().max(f64::MIN_POSITIVE);
                (u.ln() / probs[i], i)
            })
            .collect();
        keyed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        keyed
            .into_iter()
            .take(sample_size)
            .map(|(_, i)| keys[i].clone())
            .collect()
    } else {
        map.keys().cloned().collect()
    }
}

/// Dedupe disagreement learner (active learning for deduplication).
pub struct DedupeDisagreementLearner {
    pub inner: DisagreementLearner,
    pub calculator: DistanceCalculator,
}

impl DedupeDisagreementLearner {
    pub fn new(
        candidate_predicates: Vec<PredRef>,
        calculator: DistanceCalculator,
        data: &Data,
        index_include: Vec<(Record, Record)>,
        rng_seed: u64,
    ) -> Result<Self, String> {
        let mut rng = crate::logistic::seeded_rng(rng_seed);
        let data = core::index(data, 0);

        let values: Vec<Record> = data.values().cloned().collect();
        let random_pair = (
            values[rng.random_range(0..values.len())].clone(),
            values[rng.random_range(0..values.len())].clone(),
        );
        let exact_match = (random_pair.0.clone(), random_pair.0.clone());

        let mut index_include = index_include;
        index_include.push(exact_match.clone());

        // Build the training block learner.
        const N_SAMPLED_RECORDS: usize = 5000;
        let index_data = sample_records(&data, 50_000, &mut rng);
        let sampled_records = sample_records(&index_data, N_SAMPLED_RECORDS, &mut rng);

        let preds = filter_canopy_predicates(&candidate_predicates, true);
        let block_learner = training::DedupeBlockLearner::new(preds, &sampled_records, &index_data);

        let candidates = {
            let mut weights: Vec<(RecordId, RecordId, f64)> = Vec::new();
            for (_, covered) in block_learner.comparison_cover.iter() {
                if covered.len() < sampled_records.len() * sampled_records.len().saturating_sub(1) / 2
                {
                    let w = covered.len() as f64;
                    for (a, b) in covered {
                        weights.push((a.clone(), b.clone(), w));
                    }
                }
            }
            let indices = sample_indices(&weights, 10_000, usize::MAX, &mut rng);
            indices
                .into_iter()
                .map(|(a, b)| (sampled_records[&a].clone(), sampled_records[&b].clone()))
                .collect::<Vec<_>>()
        };

        let mut examples_to_index = candidates.clone();
        examples_to_index.extend(index_include.iter().cloned());

        // Index the predicates on the candidate records.
        {
            let blocker = block_learner.blocker();
            let mut records: Vec<Record> = Vec::new();
            for (a, b) in &examples_to_index {
                records.push(a.clone());
                records.push(b.clone());
            }
            let records = core::unique(&records);
            for field in blocker.index_fields.keys() {
                let mut unique_fields: Vec<crate::value::Value> = Vec::new();
                let mut seen = HashSet::new();
                for record in &records {
                    let v = record.get(field).cloned().unwrap_or(crate::value::Value::Null);
                    if seen.insert(v.clone()) {
                        unique_fields.push(v);
                    }
                }
                blocker.index(&unique_fields, field);
            }
            for pred in &blocker.index_predicates {
                pred.0.freeze(&records, None);
            }
        }

        let matcher = DistanceMatchLearner::new(calculator.clone(), candidates.clone());
        let blocker = BlockingLearner::new(TrainingBlockLearner::Dedupe(block_learner), candidates, rng_seed);
        let mut inner = DisagreementLearner::new(matcher, blocker, rng_seed);

        let mut examples = vec![exact_match.clone(); 4];
        examples.push(random_pair);
        let labels = vec![1usize, 1, 1, 1, 0];
        inner.mark(&examples, &labels);

        Ok(Self { inner, calculator })
    }
}

/// Record-linkage disagreement learner.
pub struct RecordLinkDisagreementLearner {
    pub inner: DisagreementLearner,
    pub calculator: DistanceCalculator,
}

impl RecordLinkDisagreementLearner {
    pub fn new(
        candidate_predicates: Vec<PredRef>,
        calculator: DistanceCalculator,
        data_1: &Data,
        data_2: &Data,
        index_include: Vec<(Record, Record)>,
        rng_seed: u64,
    ) -> Result<Self, String> {
        let mut rng = crate::logistic::seeded_rng(rng_seed);
        let data_1 = core::index(data_1, 0);
        let offset = data_1.len() as i64;
        let data_2 = core::index(data_2, offset);

        let v1: Vec<Record> = data_1.values().cloned().collect();
        let v2: Vec<Record> = data_2.values().cloned().collect();
        let random_pair = (
            v1[rng.random_range(0..v1.len())].clone(),
            v2[rng.random_range(0..v2.len())].clone(),
        );
        let exact_match = (random_pair.0.clone(), random_pair.0.clone());

        let mut index_include = index_include;
        index_include.push(exact_match.clone());

        const N_SAMPLED_RECORDS: usize = 4000;
        let sampled_records_1 = sample_records(&data_1, N_SAMPLED_RECORDS, &mut rng);
        let index_data = sample_records(&data_2, 50_000, &mut rng);
        let sampled_records_2 = sample_records(&index_data, N_SAMPLED_RECORDS, &mut rng);

        let preds = filter_canopy_predicates(&candidate_predicates, false);
        let block_learner = training::RecordLinkBlockLearner::new(
            preds,
            &sampled_records_1,
            &sampled_records_2,
            &index_data,
        );

        let candidates = {
            let mut weights: Vec<(RecordId, RecordId, f64)> = Vec::new();
            for (_, covered) in block_learner.comparison_cover.iter() {
                if covered.len() < sampled_records_1.len() * sampled_records_2.len() {
                    let w = covered.len() as f64;
                    for (a, b) in covered {
                        weights.push((a.clone(), b.clone(), w));
                    }
                }
            }
            let indices = sample_indices(&weights, 10_000, usize::MAX, &mut rng);
            indices
                .into_iter()
                .map(|(a, b)| {
                    (
                        sampled_records_1.get(&a).or_else(|| sampled_records_2.get(&a)).cloned().unwrap_or_default(),
                        sampled_records_2.get(&b).or_else(|| sampled_records_1.get(&b)).cloned().unwrap_or_default(),
                    )
                })
                .collect::<Vec<_>>()
        };

        let mut examples_to_index = candidates.clone();
        examples_to_index.extend(index_include.iter().cloned());

        {
            let blocker = block_learner.blocker();
            let mut a_records: Vec<Record> = Vec::new();
            let mut b_records: Vec<Record> = Vec::new();
            for (a, b) in &examples_to_index {
                a_records.push(a.clone());
                b_records.push(b.clone());
            }
            let a_records = core::unique(&a_records);
            let b_records = core::unique(&b_records);
            for field in blocker.index_fields.keys() {
                let mut unique_fields: Vec<crate::value::Value> = Vec::new();
                let mut seen = HashSet::new();
                for record in &b_records {
                    let v = record.get(field).cloned().unwrap_or(crate::value::Value::Null);
                    if seen.insert(v.clone()) {
                        unique_fields.push(v);
                    }
                }
                blocker.index(&unique_fields, field);
            }
            for pred in &blocker.index_predicates {
                pred.0.freeze(&a_records, Some(&b_records));
            }
        }

        let matcher = DistanceMatchLearner::new(calculator.clone(), candidates.clone());
        let blocker = BlockingLearner::new(
            TrainingBlockLearner::RecordLink(block_learner),
            candidates,
            rng_seed,
        );
        let mut inner = DisagreementLearner::new(matcher, blocker, rng_seed);

        let mut examples = vec![exact_match.clone(); 4];
        examples.push(random_pair);
        let labels = vec![1usize, 1, 1, 1, 0];
        inner.mark(&examples, &labels);

        Ok(Self { inner, calculator })
    }
}
