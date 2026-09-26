//! Variable definitions, ported from `dedupe.variables`.

use std::sync::Arc;

use crate::comparators::{
    CategoricalTypeComparator, Comparator, CosineComparator, CosineKind, CustomComparator,
    ExactComparator, ExistsComparator, LatLongComparator, PriceComparator, StringComparator,
    StringComparatorKind,
};
use crate::predicates::{
    threshold_repr, ExistsPredicate, IndexPredicate, PredRef, SimplePredicate,
};
use crate::value::Value;

/// A derived dummy variable produced by categorical / exists types.
#[derive(Debug, Clone)]
pub struct DerivedVar {
    pub name: String,
    pub var_type: &'static str,
    pub has_missing: bool,
}

impl DerivedVar {
    pub fn new(name: impl Into<String>, var_type: &'static str, has_missing: bool) -> Self {
        let name = name.into();
        Self {
            name: format!("({name}: {var_type})"),
            var_type,
            has_missing,
        }
    }
}

/// An expanded interaction variable (a column in the distance matrix).
#[derive(Debug, Clone)]
pub struct InteractionVar {
    pub name: String,
    pub interaction_fields: Vec<String>,
    pub has_missing: bool,
}

/// A user-defined interaction of named fields.
#[derive(Debug, Clone)]
pub struct InteractionDef {
    pub interactions: Vec<String>,
    pub has_missing: bool,
}

impl InteractionDef {
    pub fn new(interactions: &[&str]) -> Self {
        Self {
            interactions: interactions.iter().map(|s| s.to_string()).collect(),
            has_missing: false,
        }
    }

    pub fn name(&self) -> String {
        let inner: Vec<String> = self.interactions.iter().map(|s| format!("'{s}'")).collect();
        format!("(Interaction: [{}])", inner.join(", "))
    }
}

/// A field variable: field name, display name, generated predicates and
/// comparator.
pub struct FieldVariable {
    pub field: String,
    pub name: String,
    pub var_type: &'static str,
    pub has_missing: bool,
    pub predicates: Vec<PredRef>,
    pub comparator: Arc<dyn Comparator>,
    pub higher_vars: Vec<DerivedVar>,
    pub len: usize,
    /// Original category list (for settings round-trips).
    pub categories: Vec<String>,
    /// Original corpus (for settings round-trips).
    pub corpus: Vec<Value>,
}

impl std::fmt::Debug for FieldVariable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FieldVariable")
            .field("field", &self.field)
            .field("name", &self.name)
            .field("var_type", &self.var_type)
            .finish()
    }
}

/// The kind of a field variable, used for model introspection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarKind {
    Exact,
    ShortString,
    String,
    Text,
    Categorical,
    Exists,
    Price,
    LatLong,
    Set,
    Custom,
}

/// Variable definitions passed to [`crate::datamodel::DataModel`].
pub enum VariableDef {
    Field(FieldVariable),
    Interaction(InteractionDef),
}

// ---------------------------------------------------------------------------
// Predicate sets per variable type
// ---------------------------------------------------------------------------

const BASE_STRING_PREDICATES: &[&str] = &[
    "wholeFieldPredicate",
    "firstTokenPredicate",
    "firstTwoTokensPredicate",
    "commonIntegerPredicate",
    "nearIntegersPredicate",
    "firstIntegerPredicate",
    "hundredIntegerPredicate",
    "hundredIntegersOddPredicate",
    "alphaNumericPredicate",
    "sameThreeCharStartPredicate",
    "sameFiveCharStartPredicate",
    "sameSevenCharStartPredicate",
    "commonTwoTokens",
    "commonThreeTokens",
    "fingerprint",
    "oneGramFingerprint",
    "twoGramFingerprint",
    "sortedAcronym",
];

const SHORT_STRING_EXTRA: &[&str] = &[
    "commonFourGram",
    "commonSixGram",
    "tokenFieldPredicate",
    "suffixArray",
    "doubleMetaphone",
    "metaphoneToken",
];

const TFIDF_NGRAM_THRESHOLDS: &[(f64, &str)] =
    &[(0.2, "0.2"), (0.4, "0.4"), (0.6, "0.6"), (0.8, "0.8")];
const LEV_THRESHOLDS: &[(f64, &str)] = &[(1.0, "1"), (2.0, "2"), (3.0, "3"), (4.0, "4")];

fn string_predicates(
    field: &str,
    funcs: &[&str],
    index_types: &[&'static str],
    thresholds: &[(f64, &str)],
    has_missing: bool,
) -> Vec<PredRef> {
    let mut predicates: Vec<PredRef> = funcs
        .iter()
        .map(|f| PredRef::new(SimplePredicate::new(f, field, true)))
        .collect();
    predicates.extend(crate::predicates::index_predicate_types(
        index_types,
        thresholds,
        field,
    ));
    if has_missing {
        predicates.push(PredRef::new(ExistsPredicate::new(field)));
    }
    // BaseStringType always appends Levenshtein predicates.
    predicates.extend(crate::predicates::index_predicate_types(
        &[
            "LevenshteinCanopyPredicate",
            "LevenshteinSearchPredicate",
        ],
        LEV_THRESHOLDS,
        field,
    ));
    predicates
}

fn simple_predicates(field: &str, funcs: &[&str], has_missing: bool) -> Vec<PredRef> {
    let mut predicates: Vec<PredRef> = funcs
        .iter()
        .map(|f| PredRef::new(SimplePredicate::new(f, field, false)))
        .collect();
    if has_missing {
        predicates.push(PredRef::new(ExistsPredicate::new(field)));
    }
    predicates
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

/// Shared builder state.
struct Builder {
    field: String,
    name: String,
    var_type: &'static str,
    has_missing: bool,
    predicates: Vec<PredRef>,
    comparator: Arc<dyn Comparator>,
    higher_vars: Vec<DerivedVar>,
    len: usize,
    kind: VarKind,
    categories: Vec<String>,
    corpus: Vec<Value>,
}

impl Builder {
    fn finish(self) -> FieldVariable {
        let _ = self.kind;
        FieldVariable {
            field: self.field,
            name: self.name,
            var_type: self.var_type,
            has_missing: self.has_missing,
            predicates: self.predicates,
            comparator: self.comparator,
            higher_vars: self.higher_vars,
            len: self.len,
            categories: self.categories,
            corpus: self.corpus,
        }
    }
}

fn default_name(field: &str, var_type: &str) -> String {
    format!("({field}: {var_type})")
}

/// `dedupe.variables.Exact`.
pub fn exact(field: &str) -> FieldVariable {
    Builder {
        field: field.to_string(),
        name: default_name(field, "Exact"),
        var_type: "Exact",
        has_missing: false,
        predicates: simple_predicates(field, &["wholeFieldPredicate"], false),
        comparator: Arc::new(ExactComparator),
        higher_vars: Vec::new(),
        len: 1,
        kind: VarKind::Exact,
        categories: Vec::new(),
        corpus: Vec::new(),
    }
    .finish()
}

/// `dedupe.variables.ShortString`.
pub fn short_string(field: &str, crf: bool) -> FieldVariable {
    let mut funcs: Vec<&str> = BASE_STRING_PREDICATES.to_vec();
    funcs.extend_from_slice(SHORT_STRING_EXTRA);
    Builder {
        field: field.to_string(),
        name: default_name(field, "ShortString"),
        var_type: "ShortString",
        has_missing: false,
        predicates: string_predicates(
            field,
            &funcs,
            &[
                "TfidfNGramCanopyPredicate",
                "TfidfNGramSearchPredicate",
            ],
            TFIDF_NGRAM_THRESHOLDS,
            false,
        ),
        comparator: Arc::new(StringComparator {
            kind: if crf {
                StringComparatorKind::Crf
            } else {
                StringComparatorKind::AffineGap
            },
        }),
        higher_vars: Vec::new(),
        len: 1,
        kind: VarKind::ShortString,
        categories: Vec::new(),
        corpus: Vec::new(),
    }
    .finish()
}

/// `dedupe.variables.String`.
pub fn string(field: &str, crf: bool) -> FieldVariable {
    let mut funcs: Vec<&str> = BASE_STRING_PREDICATES.to_vec();
    funcs.extend_from_slice(SHORT_STRING_EXTRA);
    Builder {
        field: field.to_string(),
        name: default_name(field, "String"),
        var_type: "String",
        has_missing: false,
        predicates: string_predicates(
            field,
            &funcs,
            &[
                "TfidfNGramCanopyPredicate",
                "TfidfNGramSearchPredicate",
                "TfidfTextCanopyPredicate",
                "TfidfTextSearchPredicate",
            ],
            TFIDF_NGRAM_THRESHOLDS,
            false,
        ),
        comparator: Arc::new(StringComparator {
            kind: if crf {
                StringComparatorKind::Crf
            } else {
                StringComparatorKind::AffineGap
            },
        }),
        higher_vars: Vec::new(),
        len: 1,
        kind: VarKind::String,
        categories: Vec::new(),
        corpus: Vec::new(),
    }
    .finish()
}

/// `dedupe.variables.Text`.
pub fn text(field: &str, corpus: &[Value]) -> FieldVariable {
    Builder {
        field: field.to_string(),
        name: default_name(field, "Text"),
        var_type: "Text",
        has_missing: false,
        predicates: string_predicates(
            field,
            BASE_STRING_PREDICATES,
            &["TfidfTextCanopyPredicate", "TfidfTextSearchPredicate"],
            TFIDF_NGRAM_THRESHOLDS,
            false,
        ),
        comparator: Arc::new(CosineComparator::new(CosineKind::Text, corpus)),
        higher_vars: Vec::new(),
        len: 1,
        kind: VarKind::Text,
        categories: Vec::new(),
        corpus: corpus.to_vec(),
    }
    .finish()
}

/// `dedupe.variables.Set`.
pub fn set(field: &str, corpus: &[Value]) -> FieldVariable {
    let funcs = [
        "wholeSetPredicate",
        "commonSetElementPredicate",
        "lastSetElementPredicate",
        "commonTwoElementsPredicate",
        "commonThreeElementsPredicate",
        "magnitudeOfCardinality",
        "firstSetElementPredicate",
    ];
    let mut predicates = simple_predicates(field, &funcs, false);
    predicates.extend(crate::predicates::index_predicate_types(
        &["TfidfSetSearchPredicate", "TfidfSetCanopyPredicate"],
        TFIDF_NGRAM_THRESHOLDS,
        field,
    ));
    Builder {
        field: field.to_string(),
        name: default_name(field, "Set"),
        var_type: "Set",
        has_missing: false,
        predicates,
        comparator: Arc::new(CosineComparator::new(CosineKind::Set, corpus)),
        higher_vars: Vec::new(),
        len: 1,
        kind: VarKind::Set,
        categories: Vec::new(),
        corpus: corpus.to_vec(),
    }
    .finish()
}

/// `dedupe.variables.Categorical`.
pub fn categorical(field: &str, categories: &[String]) -> FieldVariable {
    let comparator = CategoricalTypeComparator {
        inner: crate::comparators::CategoricalComparator::new(categories),
    };
    let len = comparator.len();
    let dummy_names = comparator.inner.dummy_names.clone();
    let higher_vars = dummy_names
        .iter()
        .map(|d| DerivedVar::new(d.clone(), "Dummy", false))
        .collect();
    Builder {
        field: field.to_string(),
        name: default_name(field, "Categorical"),
        var_type: "Categorical",
        has_missing: false,
        predicates: simple_predicates(field, &["wholeFieldPredicate"], false),
        comparator: Arc::new(comparator),
        higher_vars,
        len,
        kind: VarKind::Categorical,
        categories: categories.to_vec(),
        corpus: Vec::new(),
    }
    .finish()
}

/// `dedupe.variables.Exists`.
pub fn exists(field: &str) -> FieldVariable {
    let comparator = ExistsComparator::new();
    let len = comparator.len();
    let dummy_names = comparator.dummy_names();
    let higher_vars = dummy_names
        .iter()
        .map(|d| DerivedVar::new(d.clone(), "Dummy", false))
        .collect();
    Builder {
        field: field.to_string(),
        name: default_name(field, "Exists"),
        var_type: "Exists",
        has_missing: false,
        predicates: simple_predicates(field, &[], false),
        comparator: Arc::new(comparator),
        higher_vars,
        len,
        kind: VarKind::Exists,
        categories: Vec::new(),
        corpus: Vec::new(),
    }
    .finish()
}

/// `dedupe.variables.Price`.
pub fn price(field: &str) -> FieldVariable {
    Builder {
        field: field.to_string(),
        name: default_name(field, "Price"),
        var_type: "Price",
        has_missing: false,
        predicates: simple_predicates(
            field,
            &["orderOfMagnitude", "wholeFieldPredicate", "roundTo1"],
            false,
        ),
        comparator: Arc::new(PriceComparator),
        higher_vars: Vec::new(),
        len: 1,
        kind: VarKind::Price,
        categories: Vec::new(),
        corpus: Vec::new(),
    }
    .finish()
}

/// `dedupe.variables.LatLong`.
pub fn latlong(field: &str) -> FieldVariable {
    Builder {
        field: field.to_string(),
        name: default_name(field, "LatLong"),
        var_type: "LatLong",
        has_missing: false,
        predicates: simple_predicates(field, &["latLongGridPredicate"], false),
        comparator: Arc::new(LatLongComparator),
        higher_vars: Vec::new(),
        len: 1,
        kind: VarKind::LatLong,
        categories: Vec::new(),
        corpus: Vec::new(),
    }
    .finish()
}

/// `dedupe.variables.Custom`.
pub fn custom(
    field: &str,
    label: &str,
    func: Arc<dyn Fn(&Value, &Value) -> f32 + Send + Sync>,
) -> FieldVariable {
    let name = format!("({field}: Custom, {label})");
    Builder {
        field: field.to_string(),
        name,
        var_type: "Custom",
        has_missing: false,
        predicates: simple_predicates(field, &[], false),
        comparator: Arc::new(CustomComparator {
            label: label.to_string(),
            func,
            len: 1,
        }),
        higher_vars: Vec::new(),
        len: 1,
        kind: VarKind::Custom,
        categories: Vec::new(),
        corpus: Vec::new(),
    }
    .finish()
}

/// Override the display name of a variable (Python's `name=` argument).
pub fn with_name(mut var: FieldVariable, name: &str) -> FieldVariable {
    var.name = name.to_string();
    var
}

/// Set `has_missing` on a variable, regenerating the `Exists` predicate.
pub fn with_missing(mut var: FieldVariable) -> FieldVariable {
    if var.has_missing {
        return var;
    }
    var.has_missing = true;
    var.predicates
        .push(PredRef::new(ExistsPredicate::new(&var.field)));
    // Overwrite the has_missing flag on the exists-derived dummy vars.
    for hv in var.higher_vars.iter_mut() {
        hv.has_missing = true;
    }
    var
}

impl FieldVariable {
    /// Regenerate index predicates (used when constructing from settings).
    #[allow(dead_code)]
    pub fn index_predicates(&self, type_name: &str, threshold: f64) -> Option<PredRef> {
        IndexPredicate::from_type_name(type_name, threshold, threshold_repr(threshold, threshold.fract() == 0.0), &self.field)
            .map(PredRef::new)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn string_predicate_count() {
        // 24 simple + 4 predicates x 4 thresholds + 2 levenshtein x 4 thresholds
        let v = string("name", false);
        assert_eq!(v.predicates.len(), 24 + 16 + 8);
    }

    #[test]
    fn set_predicate_order() {
        let v = set("colors", &[]);
        // 7 simple + 4 search + 4 canopy
        assert_eq!(v.predicates.len(), 7 + 8);
        let reprs: Vec<String> = v.predicates.iter().map(|p| p.0.repr()).collect();
        assert_eq!(reprs[7], "TfidfSetSearchPredicate: (0.2, colors)");
        assert_eq!(reprs[11], "TfidfSetCanopyPredicate: (0.2, colors)");
    }

    #[test]
    fn categorical_len() {
        let v = categorical("type", &["a".into(), "b".into(), "c".into()]);
        assert_eq!(v.len, 5);
        assert_eq!(v.higher_vars.len(), 5);
        assert_eq!(v.higher_vars[0].name, "((b, b): Dummy)");
    }
}

// ---------------------------------------------------------------------------
// Portable variable configuration (settings files)
// ---------------------------------------------------------------------------

/// A serializable description of a variable, used by settings files.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct VarConfig {
    pub kind: String,
    #[serde(default)]
    pub field: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub has_missing: bool,
    #[serde(default)]
    pub crf: bool,
    #[serde(default)]
    pub categories: Vec<String>,
    #[serde(default)]
    pub corpus: Vec<Value>,
    #[serde(default)]
    pub interactions: Vec<String>,
    #[serde(default)]
    pub comparator_label: Option<String>,
}

/// Registry for user-defined comparators referenced by settings files.
fn custom_registry(
) -> &'static std::sync::Mutex<std::collections::HashMap<String, Arc<dyn Fn(&Value, &Value) -> f32 + Send + Sync>>>
{
    static REG: std::sync::OnceLock<
        std::sync::Mutex<
            std::collections::HashMap<String, Arc<dyn Fn(&Value, &Value) -> f32 + Send + Sync>>,
        >,
    > = std::sync::OnceLock::new();
    REG.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

/// Register a custom comparator so it can be reconstructed from settings.
pub fn register_custom_comparator(
    label: &str,
    func: Arc<dyn Fn(&Value, &Value) -> f32 + Send + Sync>,
) {
    custom_registry()
        .lock()
        .unwrap()
        .insert(label.to_string(), func);
}

impl VarConfig {
    pub fn to_def(&self) -> Result<VariableDef, String> {
        let field = self.field.clone().unwrap_or_default();
        let mut var = match self.kind.as_str() {
            "Exact" => exact(&field),
            "ShortString" => short_string(&field, self.crf),
            "String" => string(&field, self.crf),
            "Text" => text(&field, &self.corpus),
            "Categorical" => categorical(&field, &self.categories),
            "Exists" => exists(&field),
            "Price" => price(&field),
            "LatLong" => latlong(&field),
            "Set" => set(&field, &self.corpus),
            "Interaction" => {
                let refs: Vec<&str> = self.interactions.iter().map(|s| s.as_str()).collect();
                return Ok(VariableDef::Interaction(InteractionDef::new(&refs)));
            }
            "Custom" => {
                let label = self
                    .comparator_label
                    .clone()
                    .ok_or("custom variable requires a comparator_label")?;
                let func = custom_registry()
                    .lock()
                    .unwrap()
                    .get(&label)
                    .cloned()
                    .ok_or_else(|| format!("custom comparator '{label}' is not registered"))?;
                custom(&field, &label, func)
            }
            other => return Err(format!("unknown variable kind {other}")),
        };
        if let Some(name) = &self.name {
            var = with_name(var, name);
        }
        if self.has_missing {
            var = with_missing(var);
        }
        Ok(VariableDef::Field(var))
    }

    pub fn from_def(def: &VariableDef) -> VarConfig {
        match def {
            VariableDef::Interaction(i) => VarConfig {
                kind: "Interaction".to_string(),
                interactions: i.interactions.clone(),
                ..Default::default()
            },
            VariableDef::Field(v) => VarConfig {
                kind: v.var_type.to_string(),
                field: Some(v.field.clone()),
                name: if v.name == default_name(&v.field, v.var_type) {
                    None
                } else {
                    Some(v.name.clone())
                },
                has_missing: v.has_missing,
                crf: false,
                categories: v.categories.clone(),
                corpus: v.corpus.clone(),
                interactions: Vec::new(),
                comparator_label: if v.var_type == "Custom" {
                    Some(v.comparator.name())
                } else {
                    None
                },
            },
        }
    }
}

