//! Branch-and-bound predicate selection.  Ported from
//! `dedupe/branch_and_bound.py`.

use std::collections::BTreeSet;

use indexmap::IndexMap;

use crate::predicates::PredRef;

/// A mapping from predicate to the set of covered example indices.
pub type Cover = IndexMap<PredRef, BTreeSet<usize>>;
/// An ordered set of predicates.
pub type Partial = Vec<PredRef>;

fn reachable(dupe_cover: &Cover) -> usize {
    if dupe_cover.is_empty() {
        return 0;
    }
    let mut union: BTreeSet<usize> = BTreeSet::new();
    for cover in dupe_cover.values() {
        union.extend(cover.iter().copied());
    }
    union.len()
}

fn remove_dominated(coverage: &Cover, dominator: &PredRef) -> Cover {
    let dominant_cover = &coverage[dominator];
    coverage
        .iter()
        .filter(|(pred, cover)| {
            !(dominator.0.cover_count() <= pred.0.cover_count() && dominant_cover.is_superset(cover))
        })
        .map(|(p, c)| (p.clone(), c.clone()))
        .collect()
}

fn uncovered_by(coverage: &Cover, covered: &BTreeSet<usize>) -> Cover {
    coverage
        .iter()
        .filter_map(|(pred, uncovered)| {
            let still: BTreeSet<usize> = uncovered.difference(covered).copied().collect();
            if still.is_empty() {
                None
            } else {
                Some((pred.clone(), still))
            }
        })
        .collect()
}

fn order_by(candidates: &Cover, p: &PredRef) -> (usize, isize) {
    (candidates[p].len(), -(p.0.cover_count() as isize))
}

fn score(partial: &[PredRef]) -> usize {
    partial.iter().map(|p| p.0.cover_count()).sum()
}

/// Greedy branch-and-bound search for a low-cost covering predicate set.
pub fn search(original_cover: &Cover, target: usize, calls_budget: usize) -> Partial {
    let covered = |partial: &[PredRef]| -> usize {
        if partial.is_empty() {
            return 0;
        }
        let mut union: BTreeSet<usize> = BTreeSet::new();
        for p in partial {
            union.extend(original_cover[p].iter().copied());
        }
        union.len()
    };

    let mut cheapest_score = f64::INFINITY;
    let mut cheapest: Partial = Vec::new();

    let mut to_explore: Vec<(Cover, Partial)> = vec![(original_cover.clone(), Vec::new())];
    let mut calls = calls_budget;

    while !to_explore.is_empty() && calls > 0 {
        let (candidates, partial) = to_explore.pop().unwrap();

        let covered_n = covered(&partial);
        let s = score(&partial);

        if covered_n < target {
            let window = cheapest_score - s as f64;
            let candidates: Cover = candidates
                .into_iter()
                .filter(|(p, _)| (p.0.cover_count() as f64) < window)
                .collect();

            let reach = reachable(&candidates) + covered_n;

            if !candidates.is_empty() && reach >= target {
                let mut best: Option<PredRef> = None;
                let mut best_key: Option<(usize, isize)> = None;
                for p in candidates.keys() {
                    let key = order_by(&candidates, p);
                    if best_key.is_none() || key > best_key.unwrap() {
                        best_key = Some(key);
                        best = Some(p.clone());
                    }
                }
                let best = best.unwrap();

                let reduced = remove_dominated(&candidates, &best);
                to_explore.push((reduced, partial.clone()));

                let remaining = uncovered_by(&candidates, &candidates[&best]);
                let mut new_partial = partial;
                new_partial.push(best);
                to_explore.push((remaining, new_partial));
            }
        } else if (s as f64) < cheapest_score {
            cheapest = partial;
            cheapest_score = s as f64;
        }

        calls -= 1;
    }

    cheapest
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::predicates::{PredRef, SimplePredicate};

    #[test]
    fn uncovered_by_matches_python() {
        let p1 = PredRef::new(SimplePredicate::new("wholeFieldPredicate", "a", false));
        let p2 = PredRef::new(SimplePredicate::new("wholeFieldPredicate", "b", false));
        let p3 = PredRef::new(SimplePredicate::new("wholeFieldPredicate", "c", false));
        let mut before: Cover = Cover::new();
        before.insert(p1.clone(), BTreeSet::from([1, 2, 3]));
        before.insert(p2.clone(), BTreeSet::from([1, 2]));
        before.insert(p3.clone(), BTreeSet::from([3]));

        let after = uncovered_by(&before, &BTreeSet::new());
        assert_eq!(after[&p1], BTreeSet::from([1, 2, 3]));
        let after3 = uncovered_by(&before, &BTreeSet::from([3]));
        assert_eq!(after3[&p1], BTreeSet::from([1, 2]));
        assert_eq!(after3[&p2], BTreeSet::from([1, 2]));
        assert_eq!(after3.len(), 2);
        // before unchanged
        assert_eq!(before[&p1], BTreeSet::from([1, 2, 3]));
    }
}
