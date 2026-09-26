//! The data model: derives the distance-matrix layout from variable
//! definitions.  Ported from `dedupe/datamodel.py`.

use std::sync::Arc;

use crate::comparators::Comparator;
use crate::predicates::PredRef;
use crate::value::{Record, Value};
use crate::variables::{FieldVariable, InteractionDef, InteractionVar, VariableDef};

/// Metadata for a single distance-matrix column.
#[derive(Debug, Clone)]
pub struct ColumnInfo {
    pub name: String,
    pub has_missing: bool,
    pub interaction_fields: Option<Vec<String>>,
}

/// A `Send + Sync` view of the data model used to compute distances in
/// parallel.
#[derive(Clone)]
pub struct DistanceCalculator {
    field_comparators: Vec<(String, Arc<dyn Comparator>, usize, usize)>,
    derived_start: usize,
    interaction_indices: Vec<Vec<usize>>,
    missing_field_indices: Vec<usize>,
    len: usize,
}

impl DistanceCalculator {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn distance_row(&self, r1: &Record, r2: &Record) -> Vec<f32> {
        let mut row = vec![f32::NAN; self.len];
        self.distance_row_into(r1, r2, &mut row);
        row
    }

    /// Compute the distance vector into a caller-provided buffer, avoiding
    /// per-pair allocations.
    pub fn distance_row_into(&self, r1: &Record, r2: &Record, row: &mut [f32]) {
        for (field, comparator, start, stop) in &self.field_comparators {
            let v1 = r1.get(field);
            let v2 = r2.get(field);
            let out = &mut row[*start..*stop];
            match (v1, v2) {
                (Some(a), Some(b)) if !a.is_null() && !b.is_null() => {
                    comparator.compare_into(a, b, out)
                }
                _ if comparator.handles_missing() => comparator.compare_into(
                    v1.unwrap_or(&Value::Null),
                    v2.unwrap_or(&Value::Null),
                    out,
                ),
                _ => {
                    for x in out.iter_mut() {
                        *x = f32::NAN;
                    }
                }
            }
        }

        // derived / interaction distances
        let mut current = self.derived_start;
        for indices in &self.interaction_indices {
            let mut product = 1.0f32;
            for &idx in indices {
                product *= row[idx];
            }
            row[current] = product;
            current += 1;
        }

        if self.missing_field_indices.is_empty() {
            // Fast path: no missing-data columns, so just zero out NaNs.
            for x in row[..current].iter_mut() {
                if x.is_nan() {
                    *x = 0.0;
                }
            }
        } else {
            let is_missing: Vec<bool> = row[..current].iter().map(|x| x.is_nan()).collect();
            for i in 0..current {
                if is_missing[i] {
                    row[i] = 0.0;
                }
            }
            for (k, &idx) in self.missing_field_indices.iter().enumerate() {
                row[current + k] = if is_missing[idx] { 0.0 } else { 1.0 };
            }
        }
    }

    pub fn distances(&self, pairs: &[(Record, Record)]) -> Vec<Vec<f32>> {
        pairs.iter().map(|(a, b)| self.distance_row(a, b)).collect()
    }
}

pub struct DataModel {
    pub field_variables: Vec<FieldVariable>,
    pub columns: Vec<ColumnInfo>,
    pub derived_start: usize,
    pub missing_field_indices: Vec<usize>,
    pub interaction_indices: Vec<Vec<usize>>,
    len: usize,
    calculator: DistanceCalculator,
}

impl DataModel {
    pub fn new(variable_definitions: Vec<VariableDef>) -> Result<Self, String> {
        if variable_definitions.is_empty() {
            return Err("The variable definitions cannot be empty".to_string());
        }

        let mut field_variables: Vec<FieldVariable> = Vec::new();
        let mut interactions: Vec<InteractionDef> = Vec::new();
        for def in variable_definitions {
            match def {
                VariableDef::Field(v) => field_variables.push(v),
                VariableDef::Interaction(i) => interactions.push(i),
            }
        }

        // Mirrors Python's short-circuiting `any(variable.predicates ...)`.
        if !field_variables.iter().any(|v| !v.predicates.is_empty()) {
            return Err("At least one of the variable types needs to be a type other than \
                 'Custom'. 'Custom' types have no associated blocking rules"
                .to_string());
        }

        let mut columns: Vec<ColumnInfo> = Vec::new();
        for variable in &field_variables {
            if variable.len == 1 {
                columns.push(ColumnInfo {
                    name: variable.name.clone(),
                    has_missing: variable.has_missing,
                    interaction_fields: None,
                });
            } else {
                for hv in &variable.higher_vars {
                    columns.push(ColumnInfo {
                        name: hv.name.clone(),
                        has_missing: hv.has_missing,
                        interaction_fields: None,
                    });
                }
            }
        }

        let derived_start = columns.len();

        // Expand interactions.
        let field_lookup: std::collections::HashMap<String, usize> = field_variables
            .iter()
            .enumerate()
            .map(|(i, v)| (v.name.clone(), i))
            .collect();

        for interaction in &interactions {
            let expanded = expand_interactions(interaction, &field_variables, &field_lookup)?;
            for iv in expanded {
                columns.push(ColumnInfo {
                    name: iv.name,
                    has_missing: iv.has_missing,
                    interaction_fields: Some(iv.interaction_fields),
                });
            }
        }

        let missing_field_indices: Vec<usize> = columns
            .iter()
            .enumerate()
            .filter(|(_, c)| c.has_missing)
            .map(|(i, _)| i)
            .collect();

        let var_names: Vec<String> = columns.iter().map(|c| c.name.clone()).collect();
        let mut interaction_indices: Vec<Vec<usize>> = Vec::new();
        for column in &columns {
            if let Some(fields) = &column.interaction_fields {
                let mut indices = Vec::with_capacity(fields.len());
                for f in fields {
                    let idx = var_names
                        .iter()
                        .position(|n| n == f)
                        .ok_or_else(|| format!("interaction field {f} not found in columns"))?;
                    indices.push(idx);
                }
                interaction_indices.push(indices);
            }
        }

        let len = columns.len() + missing_field_indices.len();

        let field_comparators: Vec<(String, Arc<dyn Comparator>, usize, usize)> = {
            let mut out = Vec::new();
            let mut start = 0usize;
            for var in &field_variables {
                let stop = start + var.len;
                out.push((var.field.clone(), var.comparator.clone(), start, stop));
                start = stop;
            }
            out
        };

        let calculator = DistanceCalculator {
            field_comparators,
            derived_start,
            interaction_indices: interaction_indices.clone(),
            missing_field_indices: missing_field_indices.clone(),
            len,
        };

        Ok(Self {
            field_variables,
            columns,
            derived_start,
            missing_field_indices,
            interaction_indices,
            len,
            calculator,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn calculator(&self) -> &DistanceCalculator {
        &self.calculator
    }

    /// The union of all variables' predicates (deduplicated by repr).
    pub fn predicates(&self) -> Vec<PredRef> {
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        for var in &self.field_variables {
            for p in &var.predicates {
                if seen.insert(p.0.repr()) {
                    out.push(p.clone());
                }
            }
        }
        out
    }

    pub fn distances(&self, pairs: &[(Record, Record)]) -> Vec<Vec<f32>> {
        self.calculator.distances(pairs)
    }

    pub fn check(&self, record: &Record) -> Result<(), String> {
        for var in &self.field_variables {
            if !record.contains_key(&var.field) {
                return Err(format!(
                    "Records do not line up with data model. The field '{}' is in data_model but not in a record",
                    var.field
                ));
            }
        }
        Ok(())
    }
}

fn atomic_interactions(
    interactions: &[String],
    field_variables: &[FieldVariable],
    field_lookup: &std::collections::HashMap<String, usize>,
) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    for field in interactions {
        let Some(&idx) = field_lookup.get(field) else {
            return Err(format!(
                "The interaction variable {field} is not a named variable in the variable definition"
            ));
        };
        let var = &field_variables[idx];
        // An interaction variable is never a field variable in our model, so
        // recursion only happens for nested named interactions, which we do not
        // currently support (Python supports named interactions that reference
        // other interactions).
        out.push(var.name.clone());
    }
    Ok(out)
}

/// Expand a user interaction into concrete interaction columns.
fn expand_interactions(
    interaction: &InteractionDef,
    field_variables: &[FieldVariable],
    field_lookup: &std::collections::HashMap<String, usize>,
) -> Result<Vec<InteractionVar>, String> {
    let atomic = atomic_interactions(&interaction.interactions, field_variables, field_lookup)?;

    let mut has_missing = false;
    for field in &atomic {
        if let Some(&idx) = field_lookup.get(field) {
            if field_variables[idx].has_missing {
                has_missing = true;
            }
        }
    }

    // Categoricals contribute their dummy higher vars; others stay atomic.
    let mut categorical_dummies: Vec<Vec<String>> = Vec::new();
    let mut noncategoricals: Vec<String> = Vec::new();
    for field in &atomic {
        let idx = field_lookup[field];
        let var = &field_variables[idx];
        if !var.higher_vars.is_empty() {
            categorical_dummies.push(var.higher_vars.iter().map(|h| h.name.clone()).collect());
        } else {
            noncategoricals.push(var.name.clone());
        }
    }

    let mut higher_vars = Vec::new();
    // Cartesian product over the categorical dummy groups.
    let mut indices = vec![0usize; categorical_dummies.len()];
    if categorical_dummies
        .iter()
        .any(|group| group.is_empty())
    {
        return Ok(higher_vars);
    }
    loop {
        let mut var_names: Vec<String> = categorical_dummies
            .iter()
            .zip(indices.iter())
            .map(|(group, i)| group[*i].clone())
            .collect();
        var_names.extend(noncategoricals.iter().cloned());
        let name = {
            let inner: Vec<String> = var_names.iter().map(|s| format!("'{s}'")).collect();
            format!("(Interaction: [{}])", inner.join(", "))
        };
        higher_vars.push(InteractionVar {
            name,
            interaction_fields: var_names,
            has_missing,
        });

        let mut k = 0;
        loop {
            if k >= indices.len() {
                return Ok(higher_vars);
            }
            indices[k] += 1;
            if indices[k] < categorical_dummies[k].len() {
                break;
            }
            indices[k] = 0;
            k += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::variables;
    use indexmap::IndexMap;

    fn rec(pairs: &[(&str, Value)]) -> Record {
        let mut r = IndexMap::new();
        for (k, v) in pairs {
            r.insert(k.to_string(), v.clone());
        }
        r
    }

    #[test]
    fn exact_distances() {
        let dm = DataModel::new(vec![VariableDef::Field(variables::exact("name"))]).unwrap();
        let pairs = vec![
            (rec(&[("name", Value::str("Shmoo"))]), rec(&[("name", Value::str("Shmee"))])),
            (rec(&[("name", Value::str("Shmoo"))]), rec(&[("name", Value::str("Shmoo"))])),
        ];
        let d = dm.distances(&pairs);
        assert_eq!(d, vec![vec![0.0], vec![1.0]]);
    }

    #[test]
    fn categorical_distances() {
        let dm = DataModel::new(vec![VariableDef::Field(variables::categorical(
            "type",
            &["a".into(), "b".into(), "c".into()],
        ))])
        .unwrap();
        let pairs = vec![
            (rec(&[("type", Value::str("a"))]), rec(&[("type", Value::str("b"))])),
            (rec(&[("type", Value::str("a"))]), rec(&[("type", Value::str("c"))])),
        ];
        let d = dm.distances(&pairs);
        assert_eq!(d[0], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
        assert_eq!(d[1], vec![0.0, 0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn interaction_distances() {
        // Mirrors tests/test_core.py::FieldDistances::test_comparator_interaction
        let dm = DataModel::new(vec![
            VariableDef::Field(variables::with_name(
                variables::categorical("type", &["a".into(), "b".into()]),
                "type",
            )),
            VariableDef::Interaction(InteractionDef::new(&["type", "name"])),
            VariableDef::Field(variables::with_name(variables::exact("name"), "name")),
        ])
        .unwrap();

        let p1 = (
            rec(&[("name", Value::str("steven")), ("type", Value::str("a"))]),
            rec(&[("name", Value::str("steven")), ("type", Value::str("b"))]),
        );
        let p2 = (
            rec(&[("name", Value::str("steven")), ("type", Value::str("b"))]),
            rec(&[("name", Value::str("steven")), ("type", Value::str("b"))]),
        );
        let d = dm.distances(&[p1, p2]);
        assert_eq!(d[0], vec![0.0, 1.0, 1.0, 0.0, 1.0]);
        assert_eq!(d[1], vec![1.0, 0.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn missing_field_indices() {
        let dm = DataModel::new(vec![
            VariableDef::Field(variables::with_missing(variables::string("a", false))),
            VariableDef::Field(variables::string("b", false)),
        ])
        .unwrap();
        assert_eq!(dm.missing_field_indices, vec![0]);
        assert_eq!(dm.len(), 2 + 1);
    }

    #[test]
    fn missing_values_become_zero_and_flag_column() {
        let dm = DataModel::new(vec![
            VariableDef::Field(variables::with_missing(variables::string("a", false))),
            VariableDef::Field(variables::string("b", false)),
        ])
        .unwrap();
        let r1 = rec(&[("a", Value::str("x")), ("b", Value::str("y"))]);
        let r2 = rec(&[("a", Value::Null), ("b", Value::str("y"))]);
        let d = dm.distances(&[(r1, r2)]);
        // a is missing -> distance 0, b present -> 0 for identical
        assert_eq!(d[0][0], 0.0);
        assert_eq!(d[0][2], 0.0); // missing flag for a
    }
}
