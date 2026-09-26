//! The fingerprinter: turns records into blocking keys.  Ported from
//! `dedupe/blocking.py`.

use indexmap::IndexMap;

use crate::predicates::PredRef;
use crate::value::{Data, Record, RecordId, Value};

/// Maps a field name to an index type name to the predicates using that index.
pub type IndexFields = IndexMap<String, IndexMap<String, Vec<PredRef>>>;

pub struct Fingerprinter {
    pub predicates: Vec<PredRef>,
    pub index_fields: IndexFields,
    pub index_predicates: Vec<PredRef>,
}

impl Fingerprinter {
    /// Rebuild this fingerprinter from the same predicate references.
    pub fn clone_ref(&self) -> Fingerprinter {
        Fingerprinter::new(self.predicates.clone())
    }

    pub fn new(predicates: Vec<PredRef>) -> Self {
        let mut index_fields: IndexFields = IndexMap::new();
        let mut index_predicates = Vec::new();
        for full_predicate in &predicates {
            for predicate in full_predicate.flatten() {
                if predicate.0.is_index() {
                    let field = predicate.0.field().unwrap_or_default().to_string();
                    let type_name = predicate.0.type_name().to_string();
                    index_fields
                        .entry(field)
                        .or_default()
                        .entry(type_name)
                        .or_default()
                        .push(predicate.clone());
                    index_predicates.push(predicate);
                }
            }
        }
        Self {
            predicates,
            index_fields,
            index_predicates,
        }
    }

    /// Generate `(block_key, record_id)` pairs.
    pub fn call<'a, I>(&self, records: I, target: bool) -> Vec<(String, RecordId)>
    where
        I: IntoIterator<Item = (&'a RecordId, &'a Record)>,
    {
        let mut out = Vec::new();
        for (record_id, instance) in records {
            for (i, predicate) in self.predicates.iter().enumerate() {
                if let Ok(block_keys) = predicate.call(instance, target) {
                    let suffix = format!(":{i}");
                    for block_key in block_keys {
                        out.push((block_key + &suffix, record_id.clone()));
                    }
                }
            }
        }
        out
    }

    pub fn reset_indices(&self) {
        for predicate in &self.index_predicates {
            predicate.0.reset();
        }
    }

    /// Add docs to the indices used by fingerprinters.
    pub fn index(&self, docs: &[Value], field: &str) {
        let Some(by_type) = self.index_fields.get(field) else {
            return;
        };
        // (type_name, index, preprocess_fn)
        let mut indices: Vec<(String, crate::index::SharedIndex)> = Vec::new();
        for (index_type, predicates) in by_type {
            let predicate = &predicates[0];
            let index = match predicate.0.get_index() {
                Some(ix) => ix,
                None => predicate.0.init_index(predicate.0.threshold()),
            };
            indices.push((index_type.clone(), index));
        }

        for doc in docs {
            if !doc.is_truthy() {
                continue;
            }
            for (index_type, index) in &indices {
                let predicate = &by_type[index_type][0];
                if let Some(processed) = predicate.0.preprocess_doc(doc) {
                    index.borrow_mut().index_doc(&processed);
                }
            }
        }

        for (index_type, index) in &indices {
            index.borrow_mut().init_search();
            for predicate in &by_type[index_type] {
                predicate.0.set_index(index.clone());
            }
        }
    }

    /// Remove docs from the indices used by fingerprinters.
    pub fn unindex(&self, docs: &[Value], field: &str) {
        let Some(by_type) = self.index_fields.get(field) else {
            return;
        };
        let mut indices: Vec<(String, crate::index::SharedIndex)> = Vec::new();
        for (index_type, predicates) in by_type {
            let predicate = &predicates[0];
            let index = match predicate.0.get_index() {
                Some(ix) => ix,
                None => predicate.0.init_index(predicate.0.threshold()),
            };
            indices.push((index_type.clone(), index));
        }

        for doc in docs {
            if !doc.is_truthy() {
                continue;
            }
            for (index_type, index) in &indices {
                let predicate = &by_type[index_type][0];
                if let Some(processed) = predicate.0.preprocess_doc(doc) {
                    index.borrow_mut().unindex_doc(&processed);
                }
            }
        }

        for (index_type, index) in &indices {
            index.borrow_mut().init_search();
            for predicate in &by_type[index_type] {
                predicate.0.set_index(index.clone());
            }
        }
    }

    /// Index every field that has index predicates, over all records.
    pub fn index_all(&self, data: &Data) {
        let fields: Vec<String> = self.index_fields.keys().cloned().collect();
        for field in fields {
            let mut unique: Vec<Value> = Vec::new();
            let mut seen = std::collections::HashSet::new();
            for record in data.values() {
                let v = record.get(&field).cloned().unwrap_or(Value::Null);
                if v.is_truthy() && seen.insert(v.clone()) {
                    unique.push(v);
                }
            }
            self.index(&unique, &field);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datamodel::DataModel;
    use crate::variables::{self, VariableDef};

    #[test]
    fn blocks_group_identical_records() {
        let dm = DataModel::new(vec![VariableDef::Field(variables::exact("name"))]).unwrap();
        let fp = Fingerprinter::new(dm.predicates());
        let mut data = Data::new();
        for (i, name) in ["Bob", "Bob", "Sue"].iter().enumerate() {
            let mut r = Record::new();
            r.insert("name".to_string(), Value::str(*name));
            data.insert(RecordId::Int(i as i64), r);
        }
        fp.index_all(&data);
        let pairs = fp.call(data.iter(), false);
        // records 0 and 1 share a block key
        let mut groups: IndexMap<String, Vec<i64>> = IndexMap::new();
        for (k, id) in pairs {
            groups.entry(k).or_default().push(id.as_int().unwrap());
        }
        assert!(groups.values().any(|g| g == &vec![0, 1]));
    }
}
