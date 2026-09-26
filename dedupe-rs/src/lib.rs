//! A Rust port of the [`dedupe`](https://github.com/dedupeio/dedupe) Python
//! library for fuzzy matching, de-duplication and entity resolution.
//!
//! The port follows the original module layout closely so that behaviour can be
//! compared side by side with the reference implementation.

#![allow(clippy::type_complexity)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::if_same_then_else)]

pub mod api;
pub mod convenience;
pub mod blocking;
pub mod branch_and_bound;
pub mod canonical;
pub mod clustering;
pub mod comparators;
pub mod core;
pub mod cpredicates;
pub mod datamodel;
pub mod double_metaphone;
pub mod index;
pub mod labeler;
pub mod logistic;
pub mod predicate_functions;
pub mod predicates;
pub mod serializer;
pub mod training;
pub mod value;
pub mod variables;

pub use value::{Data, Record, RecordId, Value};
