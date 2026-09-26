//! The user-facing matching API.  Ported from `dedupe/api.py`.

use std::collections::BTreeSet;

use rustc_hash::{FxHashMap, FxHashSet};

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::blocking::Fingerprinter;
use crate::clustering::{self, Cluster, ScoredPair};
use crate::core;
use crate::datamodel::DataModel;
use crate::labeler::{DedupeDisagreementLearner, RecordLinkDisagreementLearner};
use crate::logistic::{Classifier, GridSearchCv, LogisticRegression};
use crate::predicates::{PredRef, PredicateConfig};
use crate::serializer::TrainingData;
use crate::value::{Data, Record, RecordId};
use crate::variables::{VarConfig, VariableDef};

/// A pair of `(record_id, record)` values.
pub type RecordPair = ((RecordId, Record), (RecordId, Record));

/// Join constraints for record linkage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinConstraint {
    OneToOne,
    ManyToOne,
    ManyToMany,
}

/// A matcher that can score pairs of records.
pub struct Matching {
    pub num_cores: usize,
    pub in_memory: bool,
    pub data_model: DataModel,
    pub classifier: LogisticRegression,
    pub predicates: Vec<PredRef>,
    pub fingerprinter: Option<Fingerprinter>,
    pub variable_configs: Vec<VarConfig>,
}

impl Matching {
    pub fn new(variable_defs: Vec<VariableDef>, num_cores: usize, in_memory: bool) -> Result<Self, String> {
        let variable_configs = variable_defs.iter().map(VarConfig::from_def).collect();
        let data_model = DataModel::new(variable_defs)?;
        Ok(Self {
            num_cores,
            in_memory,
            data_model,
            classifier: LogisticRegression::default(),
            predicates: Vec::new(),
            fingerprinter: None,
            variable_configs,
        })
    }

    pub fn fingerprinter(&self) -> Result<&Fingerprinter, String> {
        self.fingerprinter
            .as_ref()
            .ok_or_else(|| "the record fingerprinter is not initialized, please run the train method".to_string())
    }

    pub fn score(&self, pairs: &[RecordPair]) -> Vec<ScoredPair> {
        core::score_duplicates(pairs, self.data_model.calculator(), &self.classifier, self.num_cores)
    }

    /// Score id pairs against a dataset without copying records.
    pub fn score_ids(&self, ids: &[(RecordId, RecordId)], data: &Data) -> Vec<ScoredPair> {
        core::score_duplicate_ids(
            data,
            ids,
            self.data_model.calculator(),
            &self.classifier,
            self.num_cores,
        )
    }

    /// Score linkage id pairs against two datasets without copying records.
    pub fn score_link_ids(
        &self,
        ids: &[(RecordId, RecordId)],
        data_1: &Data,
        data_2: &Data,
    ) -> Vec<ScoredPair> {
        core::score_link_ids(
            data_1,
            data_2,
            ids,
            self.data_model.calculator(),
            &self.classifier,
            self.num_cores,
        )
    }
}

/// Maximum number of bits (256 MB) we are willing to allocate for the dense
/// pair matrix before falling back to a hash set.
const MAX_PAIR_BITS: u128 = 2_000_000_000;

/// Generate the distinct record pairs implied by a block map.
///
/// When the upper-triangle bit matrix fits in memory this uses a bitset, which
/// avoids the ~10x redundant hash insertions that overlapping blocks cause.
fn distinct_dedupe_pairs(
    blocks: &FxHashMap<String, Vec<RecordId>>,
) -> Vec<(RecordId, RecordId)> {
    // Assign a dense index to every record that appears in any block.
    let mut index: FxHashMap<RecordId, u32> = FxHashMap::default();
    let mut ids: Vec<RecordId> = Vec::new();
    for members in blocks.values() {
        for m in members {
            if !index.contains_key(m) {
                index.insert(m.clone(), ids.len() as u32);
                ids.push(m.clone());
            }
        }
    }
    let n = ids.len();
    if n < 2 {
        return Vec::new();
    }

    let bits = (n as u128) * ((n - 1) as u128) / 2;
    if bits <= MAX_PAIR_BITS {
        // Row `a` holds a bit for every `b > a`, at offset `b - a - 1`.
        let mut rows: Vec<Vec<u64>> = (0..n)
            .map(|a| vec![0u64; (n - 1 - a).div_ceil(64)])
            .collect();
        for members in blocks.values() {
            let idxs: Vec<usize> = members.iter().map(|m| index[m] as usize).collect();
            for i in 0..idxs.len() {
                for j in (i + 1)..idxs.len() {
                    let (a, b) = if idxs[i] < idxs[j] {
                        (idxs[i], idxs[j])
                    } else {
                        (idxs[j], idxs[i])
                    };
                    let k = b - a - 1;
                    rows[a][k >> 6] |= 1u64 << (k & 63);
                }
            }
        }
        let mut out = Vec::new();
        for a in 0..n {
            for (w, &word) in rows[a].iter().enumerate() {
                let mut word = word;
                while word != 0 {
                    let t = word.trailing_zeros() as usize;
                    let b = a + 1 + (w << 6) + t;
                    out.push((ids[a].clone(), ids[b].clone()));
                    word &= word - 1;
                }
            }
        }
        out
    } else {
        let mut seen: FxHashSet<(RecordId, RecordId)> = FxHashSet::default();
        for members in blocks.values() {
            for i in 0..members.len() {
                for j in (i + 1)..members.len() {
                    let (a, b) = if members[i] < members[j] {
                        (members[i].clone(), members[j].clone())
                    } else {
                        (members[j].clone(), members[i].clone())
                    };
                    seen.insert((a, b));
                }
            }
        }
        let mut out: Vec<(RecordId, RecordId)> = seen.into_iter().collect();
        out.sort();
        out
    }
}

/// Build blocking pair ids for deduplication (no record copies).
pub fn pairs_dedupe_ids(fp: &Fingerprinter, data: &Data) -> Vec<(RecordId, RecordId)> {
    fp.index_all(data);
    let mut blocks: FxHashMap<String, Vec<RecordId>> = FxHashMap::default();
    for (block_key, record_id) in fp.call(data.iter(), false) {
        blocks.entry(block_key).or_default().push(record_id);
    }
    fp.reset_indices();
    for ids in blocks.values_mut() {
        ids.sort();
        ids.dedup();
    }
    distinct_dedupe_pairs(&blocks)
}

/// Build blocking pair ids for record linkage (no record copies).
pub fn pairs_link_ids(fp: &Fingerprinter, data_1: &Data, data_2: &Data) -> Vec<(RecordId, RecordId)> {
    fp.index_all(data_2);
    let mut blocks_a: FxHashMap<String, Vec<RecordId>> = FxHashMap::default();
    for (block_key, record_id) in fp.call(data_1.iter(), false) {
        blocks_a.entry(block_key).or_default().push(record_id);
    }
    let mut blocks_b: FxHashMap<String, Vec<RecordId>> = FxHashMap::default();
    for (block_key, record_id) in fp.call(data_2.iter(), true) {
        blocks_b.entry(block_key).or_default().push(record_id);
    }
    fp.reset_indices();
    let mut seen: FxHashSet<(RecordId, RecordId)> = FxHashSet::default();
    for (block_key, ids_a) in &blocks_a {
        if let Some(ids_b) = blocks_b.get(block_key) {
            for a in ids_a {
                for b in ids_b {
                    seen.insert((a.clone(), b.clone()));
                }
            }
        }
    }
    let mut out: Vec<(RecordId, RecordId)> = seen.into_iter().collect();
    out.sort();
    out
}

/// Build blocking pairs for deduplication (with record copies).
pub fn pairs_dedupe(fp: &Fingerprinter, data: &Data) -> Vec<RecordPair> {
    pairs_dedupe_ids(fp, data)
        .into_iter()
        .filter_map(|(a, b)| {
            let ra = data.get(&a)?;
            let rb = data.get(&b)?;
            Some(((a, ra.clone()), (b, rb.clone())))
        })
        .collect()
}

/// Build blocking pairs for record linkage (with record copies).
pub fn pairs_link(fp: &Fingerprinter, data_1: &Data, data_2: &Data) -> Vec<RecordPair> {
    pairs_link_ids(fp, data_1, data_2)
        .into_iter()
        .filter_map(|(a, b)| {
            let ra = data_1.get(&a)?;
            let rb = data_2.get(&b)?;
            Some(((a, ra.clone()), (b, rb.clone())))
        })
        .collect()
}

fn add_singletons(all_ids: &[RecordId], clusters: Vec<Cluster>) -> Vec<Cluster> {
    let mut remaining: BTreeSet<RecordId> = all_ids.iter().cloned().collect();
    let mut out = Vec::new();
    for (ids, scores) in clusters {
        for id in &ids {
            remaining.remove(id);
        }
        out.push((ids, scores));
    }
    for singleton in remaining {
        out.push((vec![singleton], vec![1.0]));
    }
    out
}

/// A deduplication matcher (active learning).
pub struct Dedupe {
    pub matching: Matching,
    pub training_pairs: TrainingData,
    pub active_learner: Option<DedupeDisagreementLearner>,
    pub rng_seed: u64,
}

impl Dedupe {
    pub fn new(variable_defs: Vec<VariableDef>, num_cores: usize, in_memory: bool) -> Result<Self, String> {
        Ok(Self {
            matching: Matching::new(variable_defs, num_cores, in_memory)?,
            training_pairs: TrainingData::default(),
            active_learner: None,
            rng_seed: 0,
        })
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng_seed = seed;
        self
    }

    pub fn check_data(&self, data: &Data) -> Result<(), String> {
        if data.is_empty() {
            return Err("Dictionary of records is empty.".to_string());
        }
        let first = data.values().next().unwrap();
        self.matching.data_model.check(first)
    }

    /// Initialize the active learner with data and optional training data.
    pub fn prepare_training(&mut self, data: &Data) -> Result<(), String> {
        self.check_data(data)?;
        self.active_learner = None;

        let (examples, _y) = flatten_training(&self.training_pairs);
        let learner = DedupeDisagreementLearner::new(
            self.matching.data_model.predicates(),
            self.matching.data_model.calculator().clone(),
            data,
            examples,
            self.rng_seed,
        )?;
        self.active_learner = Some(learner);
        let (examples, y) = flatten_training(&self.training_pairs);
        self.active_learner.as_mut().unwrap().inner.mark(&examples, &y);
        Ok(())
    }

    /// The pairs the model is most curious to have labeled.
    pub fn uncertain_pairs(&mut self) -> Result<(Record, Record), String> {
        let learner = self
            .active_learner
            .as_mut()
            .ok_or("Please initialize with the prepare_training method")?;
        learner.inner.pop()
    }

    /// Add labeled pairs and update the model.
    pub fn mark_pairs(&mut self, labeled: &TrainingData) -> Result<(), String> {
        if !labeled.match_.is_empty() {
            self.check_record_pair(&labeled.match_[0])?;
        }
        if !labeled.distinct.is_empty() {
            self.check_record_pair(&labeled.distinct[0])?;
        }

        self.training_pairs
            .match_
            .extend(labeled.match_.iter().cloned());
        self.training_pairs
            .distinct
            .extend(labeled.distinct.iter().cloned());

        if let Some(learner) = self.active_learner.as_mut() {
            let (examples, y) = flatten_training(labeled);
            learner.inner.mark(&examples, &y);
        }
        Ok(())
    }

    fn check_record_pair(&self, pair: &(Record, Record)) -> Result<(), String> {
        self.matching.data_model.check(&pair.0)?;
        self.matching.data_model.check(&pair.1)
    }

    /// Learn the pairwise classifier and fingerprinting rules.
    pub fn train(&mut self, recall: f64, index_predicates: bool) -> Result<(), String> {
        let learner = self
            .active_learner
            .as_ref()
            .ok_or("Please initialize with the prepare_training method")?;

        let (examples, y) = flatten_training(&self.training_pairs);
        if examples.is_empty() {
            return Err("No training data".to_string());
        }
        let features = featurize(&self.matching.data_model, &examples);
        let mut gs = GridSearchCv::default();
        gs.fit(&features, &y);
        self.matching.classifier = gs.model;

        self.matching.predicates = learner.inner.learn_predicates(recall, index_predicates);
        self.matching.fingerprinter = Some(Fingerprinter::new(self.matching.predicates.clone()));
        Ok(())
    }

    /// Partition a dataset into clusters of duplicate records.
    pub fn partition(&mut self, data: &Data, threshold: f64) -> Result<Vec<Cluster>, String> {
        let fp = self.matching.fingerprinter()?.clone_ref();
        let ids = pairs_dedupe_ids(&fp, data);
        let scores = self.matching.score_ids(&ids, data);
        let clusters = clustering::cluster(&scores, threshold, 30000);
        Ok(add_singletons(&data.keys().cloned().collect::<Vec<_>>(), clusters))
    }

    /// Score arbitrary blocking pairs.
    pub fn score(&self, pairs: &[RecordPair]) -> Vec<ScoredPair> {
        self.matching.score(pairs)
    }

    pub fn cluster(&self, scores: &[ScoredPair], threshold: f64) -> Vec<Cluster> {
        clustering::cluster(scores, threshold, 30000)
    }
}

/// A record-linkage matcher (active learning).
pub struct RecordLink {
    pub matching: Matching,
    pub training_pairs: TrainingData,
    pub active_learner: Option<RecordLinkDisagreementLearner>,
    pub rng_seed: u64,
}

impl RecordLink {
    pub fn new(variable_defs: Vec<VariableDef>, num_cores: usize, in_memory: bool) -> Result<Self, String> {
        Ok(Self {
            matching: Matching::new(variable_defs, num_cores, in_memory)?,
            training_pairs: TrainingData::default(),
            active_learner: None,
            rng_seed: 0,
        })
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng_seed = seed;
        self
    }

    pub fn prepare_training(&mut self, data_1: &Data, data_2: &Data) -> Result<(), String> {
        if data_1.is_empty() {
            return Err("Dictionary of records from first dataset is empty.".to_string());
        }
        if data_2.is_empty() {
            return Err("Dictionary of records from second dataset is empty.".to_string());
        }
        self.matching
            .data_model
            .check(data_1.values().next().unwrap())?;
        self.matching
            .data_model
            .check(data_2.values().next().unwrap())?;

        self.active_learner = None;
        let (examples, _y) = flatten_training(&self.training_pairs);
        let learner = RecordLinkDisagreementLearner::new(
            self.matching.data_model.predicates(),
            self.matching.data_model.calculator().clone(),
            data_1,
            data_2,
            examples,
            self.rng_seed,
        )?;
        self.active_learner = Some(learner);
        let (examples, y) = flatten_training(&self.training_pairs);
        self.active_learner.as_mut().unwrap().inner.mark(&examples, &y);
        Ok(())
    }

    pub fn uncertain_pairs(&mut self) -> Result<(Record, Record), String> {
        let learner = self
            .active_learner
            .as_mut()
            .ok_or("Please initialize with the prepare_training method")?;
        learner.inner.pop()
    }

    pub fn mark_pairs(&mut self, labeled: &TrainingData) -> Result<(), String> {
        self.training_pairs
            .match_
            .extend(labeled.match_.iter().cloned());
        self.training_pairs
            .distinct
            .extend(labeled.distinct.iter().cloned());
        if let Some(learner) = self.active_learner.as_mut() {
            let (examples, y) = flatten_training(labeled);
            learner.inner.mark(&examples, &y);
        }
        Ok(())
    }

    pub fn train(&mut self, recall: f64, index_predicates: bool) -> Result<(), String> {
        let learner = self
            .active_learner
            .as_ref()
            .ok_or("Please initialize with the prepare_training method")?;
        let (examples, y) = flatten_training(&self.training_pairs);
        if examples.is_empty() {
            return Err("No training data".to_string());
        }
        let features = featurize(&self.matching.data_model, &examples);
        let mut gs = GridSearchCv::default();
        gs.fit(&features, &y);
        self.matching.classifier = gs.model;
        self.matching.predicates = learner.inner.learn_predicates(recall, index_predicates);
        self.matching.fingerprinter = Some(Fingerprinter::new(self.matching.predicates.clone()));
        Ok(())
    }

    pub fn join(
        &self,
        data_1: &Data,
        data_2: &Data,
        threshold: f64,
        constraint: JoinConstraint,
    ) -> Result<Vec<ScoredPair>, String> {
        let fp = self.matching.fingerprinter()?.clone_ref();
        let ids = pairs_link_ids(&fp, data_1, data_2);
        let scores = self.matching.score_link_ids(&ids, data_1, data_2);
        Ok(match constraint {
            JoinConstraint::OneToOne => {
                let filtered: Vec<ScoredPair> = scores
                    .into_iter()
                    .filter(|(_, s)| (*s as f64) > threshold)
                    .collect();
                clustering::greedy_matching(filtered)
            }
            JoinConstraint::ManyToOne => {
                clustering::pair_gazette_matching(scores, threshold, 1)
            }
            JoinConstraint::ManyToMany => scores
                .into_iter()
                .filter(|(_, s)| (*s as f64) > threshold)
                .collect(),
        })
    }
}

/// A gazetteer matcher.
pub struct Gazetteer {
    pub matching: Matching,
    pub training_pairs: TrainingData,
    pub active_learner: Option<RecordLinkDisagreementLearner>,
    pub indexed_data: Data,
    pub indexed_blocks: FxHashMap<String, Vec<RecordId>>,
    pub rng_seed: u64,
}

impl Gazetteer {
    pub fn new(variable_defs: Vec<VariableDef>, num_cores: usize, in_memory: bool) -> Result<Self, String> {
        Ok(Self {
            matching: Matching::new(variable_defs, num_cores, in_memory)?,
            training_pairs: TrainingData::default(),
            active_learner: None,
            indexed_data: Data::new(),
            indexed_blocks: FxHashMap::default(),
            rng_seed: 0,
        })
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng_seed = seed;
        self
    }

    /// Build a gazetteer around an already-configured matcher (settings files).
    pub fn from_matching(matching: Matching) -> Self {
        Self {
            matching,
            training_pairs: TrainingData::default(),
            active_learner: None,
            indexed_data: Data::new(),
            indexed_blocks: FxHashMap::default(),
            rng_seed: 0,
        }
    }

    pub fn prepare_training(&mut self, data_1: &Data, data_2: &Data) -> Result<(), String> {
        if data_1.is_empty() || data_2.is_empty() {
            return Err("Dictionary of records is empty.".to_string());
        }
        self.matching
            .data_model
            .check(data_1.values().next().unwrap())?;
        self.matching
            .data_model
            .check(data_2.values().next().unwrap())?;

        self.active_learner = None;
        let (examples, _) = flatten_training(&self.training_pairs);
        let learner = RecordLinkDisagreementLearner::new(
            self.matching.data_model.predicates(),
            self.matching.data_model.calculator().clone(),
            data_1,
            data_2,
            examples,
            self.rng_seed,
        )?;
        self.active_learner = Some(learner);
        let (examples, y) = flatten_training(&self.training_pairs);
        self.active_learner.as_mut().unwrap().inner.mark(&examples, &y);
        Ok(())
    }

    pub fn uncertain_pairs(&mut self) -> Result<(Record, Record), String> {
        self.active_learner
            .as_mut()
            .ok_or("Please initialize with the prepare_training method")?
            .inner
            .pop()
    }

    pub fn mark_pairs(&mut self, labeled: &TrainingData) -> Result<(), String> {
        self.training_pairs
            .match_
            .extend(labeled.match_.iter().cloned());
        self.training_pairs
            .distinct
            .extend(labeled.distinct.iter().cloned());
        if let Some(learner) = self.active_learner.as_mut() {
            let (examples, y) = flatten_training(labeled);
            learner.inner.mark(&examples, &y);
        }
        Ok(())
    }

    pub fn train(&mut self, recall: f64, index_predicates: bool) -> Result<(), String> {
        let learner = self
            .active_learner
            .as_ref()
            .ok_or("Please initialize with the prepare_training method")?;
        let (examples, y) = flatten_training(&self.training_pairs);
        if examples.is_empty() {
            return Err("No training data".to_string());
        }
        let features = featurize(&self.matching.data_model, &examples);
        let mut gs = GridSearchCv::default();
        gs.fit(&features, &y);
        self.matching.classifier = gs.model;
        self.matching.predicates = learner.inner.learn_predicates(recall, index_predicates);
        self.matching.fingerprinter = Some(Fingerprinter::new(self.matching.predicates.clone()));
        Ok(())
    }

    /// Add records to the index of records to match against.
    pub fn index(&mut self, data: &Data) -> Result<(), String> {
        let fp = self.matching.fingerprinter()?.clone_ref();
        fp.index_all(data);
        for (block_key, record_id) in fp.call(data.iter(), true) {
            self.indexed_blocks
                .entry(block_key)
                .or_default()
                .push(record_id);
        }
        for (k, v) in data {
            self.indexed_data.insert(k.clone(), v.clone());
        }
        Ok(())
    }

    /// Remove records from the index.
    pub fn unindex(&mut self, data: &Data) {
        let ids: BTreeSet<RecordId> = data.keys().cloned().collect();
        for values in self.indexed_blocks.values_mut() {
            values.retain(|id| !ids.contains(id));
        }
        for k in data.keys() {
            self.indexed_data.shift_remove(k);
        }
    }

    /// Yield groups of pairs of records that share fingerprints.
    pub fn blocks(&self, data: &Data) -> Result<Vec<Vec<RecordPair>>, String> {
        let fp = self.matching.fingerprinter()?;
        let mut out: Vec<Vec<RecordPair>> = Vec::new();
        let mut ordered: Vec<(&RecordId, &Record)> = data.iter().collect();
        ordered.sort_by(|a, b| a.0.cmp(b.0));
        for (record_id, record) in ordered {
            let keys = fp.call(std::iter::once((record_id, record)), false);
            let mut block: Vec<RecordPair> = Vec::new();
            let mut seen: BTreeSet<RecordId> = BTreeSet::new();
            for (block_key, _) in keys {
                if let Some(ids) = self.indexed_blocks.get(&block_key) {
                    for other in ids {
                        if seen.insert(other.clone()) {
                            if let Some(other_record) = self.indexed_data.get(other) {
                                block.push((
                                    (record_id.clone(), record.clone()),
                                    (other.clone(), other_record.clone()),
                                ));
                            }
                        }
                    }
                }
            }
            if !block.is_empty() {
                out.push(block);
            }
        }
        Ok(out)
    }

    pub fn many_to_n(
        &self,
        score_blocks: &[Vec<ScoredPair>],
        threshold: f64,
        n_matches: usize,
    ) -> Vec<Vec<ScoredPair>> {
        clustering::gazette_matching(score_blocks, threshold, n_matches)
    }

    /// Search the gazetteer for matches to `data`.
    pub fn search(
        &self,
        data: &Data,
        threshold: f64,
        n_matches: usize,
    ) -> Result<IndexMap<RecordId, Vec<(RecordId, f32)>>, String> {
        let blocks = self.blocks(data)?;
        let scored = core::score_gazette(
            &blocks,
            self.matching.data_model.calculator(),
            &self.matching.classifier,
        );
        let results = self.many_to_n(&scored, threshold, n_matches);

        let mut out: IndexMap<RecordId, Vec<(RecordId, f32)>> = IndexMap::new();
        for result in results {
            let mut a: Option<RecordId> = None;
            let mut prepared: Vec<(RecordId, f32)> = Vec::new();
            for ((ra, rb), score) in &result {
                a = Some(ra.clone());
                prepared.push((rb.clone(), *score));
            }
            if let Some(a) = a {
                out.insert(a, prepared);
            }
        }
        for k in data.keys() {
            out.entry(k.clone()).or_default();
        }
        Ok(out)
    }
}

/// Compute the featurized distance matrix for a set of record pairs.
pub fn featurize(data_model: &DataModel, pairs: &[(Record, Record)]) -> Vec<Vec<f64>> {
    data_model
        .distances(pairs)
        .into_iter()
        .map(|row| row.into_iter().map(|x| x as f64).collect())
        .collect()
}

/// Split training data into examples and labels.
pub fn flatten_training(training: &TrainingData) -> (Vec<(Record, Record)>, Vec<usize>) {
    let mut examples = Vec::new();
    let mut y = Vec::new();
    for pair in &training.match_ {
        examples.push(pair.clone());
        y.push(1);
    }
    for pair in &training.distinct {
        examples.push(pair.clone());
        y.push(0);
    }
    (examples, y)
}

// ---------------------------------------------------------------------------
// Settings files
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifierSettings {
    pub coef: Vec<f64>,
    pub intercept: f64,
    pub c: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    pub version: u32,
    pub variables: Vec<VarConfig>,
    pub predicates: Vec<PredicateConfig>,
    pub classifier: ClassifierSettings,
}

impl Settings {
    pub fn from_matching(matching: &Matching) -> Self {
        Settings {
            version: 1,
            variables: matching.variable_configs.clone(),
            predicates: matching.predicates.iter().map(|p| p.to_config()).collect(),
            classifier: ClassifierSettings {
                coef: matching.classifier.coef.clone(),
                intercept: matching.classifier.intercept,
                c: matching.classifier.c,
            },
        }
    }

    pub fn to_matching(&self, num_cores: usize, in_memory: bool) -> Result<Matching, String> {
        let mut defs = Vec::new();
        for v in &self.variables {
            defs.push(v.to_def()?);
        }
        let mut matching = Matching::new(defs, num_cores, in_memory)?;
        matching.predicates = self
            .predicates
            .iter()
            .map(|c| PredRef::from_config(c).ok_or("invalid predicate config"))
            .collect::<Result<Vec<_>, _>>()?;
        matching.fingerprinter = Some(Fingerprinter::new(matching.predicates.clone()));
        matching.classifier = LogisticRegression::from_coef(
            self.classifier.coef.clone(),
            self.classifier.intercept,
            self.classifier.c,
        );
        Ok(matching)
    }
}

/// A dedupe matcher loaded from settings.
pub struct StaticDedupe {
    pub matching: Matching,
}

impl StaticDedupe {
    pub fn from_settings(settings: &Settings, num_cores: usize, in_memory: bool) -> Result<Self, String> {
        Ok(Self {
            matching: settings.to_matching(num_cores, in_memory)?,
        })
    }

    pub fn partition(&self, data: &Data, threshold: f64) -> Result<Vec<Cluster>, String> {
        let fp = self.matching.fingerprinter()?.clone_ref();
        let ids = pairs_dedupe_ids(&fp, data);
        let scores = self.matching.score_ids(&ids, data);
        let clusters = clustering::cluster(&scores, threshold, 30000);
        Ok(add_singletons(&data.keys().cloned().collect::<Vec<_>>(), clusters))
    }
}

/// A record-linkage matcher loaded from settings.
pub struct StaticRecordLink {
    pub matching: Matching,
}

impl StaticRecordLink {
    pub fn from_settings(settings: &Settings, num_cores: usize, in_memory: bool) -> Result<Self, String> {
        Ok(Self {
            matching: settings.to_matching(num_cores, in_memory)?,
        })
    }

    pub fn join(
        &self,
        data_1: &Data,
        data_2: &Data,
        threshold: f64,
        constraint: JoinConstraint,
    ) -> Result<Vec<ScoredPair>, String> {
        let fp = self.matching.fingerprinter()?.clone_ref();
        let ids = pairs_link_ids(&fp, data_1, data_2);
        let scores = self.matching.score_link_ids(&ids, data_1, data_2);
        Ok(match constraint {
            JoinConstraint::OneToOne => clustering::greedy_matching(
                scores.into_iter().filter(|(_, s)| (*s as f64) > threshold).collect(),
            ),
            JoinConstraint::ManyToOne => clustering::pair_gazette_matching(scores, threshold, 1),
            JoinConstraint::ManyToMany => scores
                .into_iter()
                .filter(|(_, s)| (*s as f64) > threshold)
                .collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::variables;
    use crate::value::Value;

    fn data(records: &[(i64, &str, &str)]) -> Data {
        let mut d = Data::new();
        for (id, name, age) in records {
            let mut r = Record::new();
            r.insert("name".to_string(), Value::str(*name));
            r.insert("age".to_string(), Value::str(*age));
            d.insert(RecordId::Int(*id), r);
        }
        d
    }

    #[test]
    fn dedupe_pair_generation() {
        let defs = vec![VariableDef::Field(variables::string("name", false))];
        let matching = Matching::new(defs, 1, true).unwrap();
        let fp = Fingerprinter::new(matching.data_model.predicates());
        let d = data(&[(100, "Bob", "50"), (101, "Bob", "51"), (102, "Sue", "10")]);
        let pairs = pairs_dedupe(&fp, &d);
        assert!(pairs
            .iter()
            .any(|((a, _), (b, _))| a == &RecordId::Int(100) && b == &RecordId::Int(101)));
    }

    #[test]
    fn settings_round_trip() {
        let defs = vec![
            VariableDef::Field(variables::string("name", false)),
            VariableDef::Field(variables::categorical("age", &["10".into(), "50".into()])),
        ];
        let mut matching = Matching::new(defs, 1, true).unwrap();
        matching.predicates = matching.data_model.predicates();
        matching.fingerprinter = Some(Fingerprinter::new(matching.predicates.clone()));
        matching.classifier = LogisticRegression::from_coef(vec![0.5, -0.5], 0.1, 1.0);
        let settings = Settings::from_matching(&matching);
        let restored = settings.to_matching(1, true).unwrap();
        assert_eq!(restored.classifier.coef, vec![0.5, -0.5]);
        assert_eq!(restored.predicates.len(), matching.predicates.len());
    }
}
