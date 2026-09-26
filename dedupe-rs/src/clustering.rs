//! Clustering, ported from `dedupe/clustering.py`.
//!
//! The Python implementation delegates the hierarchical clustering to SciPy.
//! This module reimplements the centroid (UPGMC) linkage with Lance-Williams
//! updates and the `fcluster(..., criterion="distance")` cut.

use rustc_hash::FxHashMap;

use crate::value::RecordId;

/// A scored pair of record ids (the smaller id first).
pub type ScoredPair = ((RecordId, RecordId), f32);

/// A cluster: member ids and per-member confidence scores.
pub type Cluster = (Vec<RecordId>, Vec<f64>);

/// Compact connected-component structure: component `k` owns the edge indices
/// `order[starts[k]..starts[k + 1]]`.
pub struct ComponentRanges {
    pub order: Vec<u32>,
    pub starts: Vec<u32>,
}

impl ComponentRanges {
    pub fn len(&self) -> usize {
        self.starts.len().saturating_sub(1)
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn get(&self, k: usize) -> &[u32] {
        &self.order[self.starts[k] as usize..self.starts[k + 1] as usize]
    }
}

/// Compute connected components without allocating a `Vec` per component
/// (which is expensive when most components are single edges).
pub fn component_ranges(edges: &[ScoredPair]) -> ComponentRanges {
    if edges.is_empty() {
        return ComponentRanges {
            order: Vec::new(),
            starts: vec![0],
        };
    }

    let mut parent: FxHashMap<RecordId, RecordId> = FxHashMap::default();
    fn find(parent: &mut FxHashMap<RecordId, RecordId>, x: &RecordId) -> RecordId {
        let p = parent.get(x).cloned().unwrap_or_else(|| x.clone());
        if &p == x {
            return p;
        }
        let r = find(parent, &p);
        parent.insert(x.clone(), r.clone());
        r
    }
    for ((a, b), _) in edges {
        parent.entry(a.clone()).or_insert_with(|| a.clone());
        parent.entry(b.clone()).or_insert_with(|| b.clone());
        let ra = find(&mut parent, a);
        let rb = find(&mut parent, b);
        if ra != rb {
            parent.insert(ra, rb);
        }
    }

    // Assign a dense label to each component, in first-seen order.
    let mut label_of_root: FxHashMap<RecordId, u32> = FxHashMap::default();
    let mut labels: Vec<u32> = Vec::with_capacity(edges.len());
    let mut next_label: u32 = 0;
    for edge in edges {
        let root = find(&mut parent, &edge.0 .0);
        let label = *label_of_root.entry(root).or_insert_with(|| {
            let l = next_label;
            next_label += 1;
            l
        });
        labels.push(label);
    }

    // Stable sort of edge indices by component label groups each component's
    // edges contiguously.
    let mut order: Vec<u32> = (0..edges.len() as u32).collect();
    order.sort_by_key(|&i| labels[i as usize]);

    let mut starts: Vec<u32> = Vec::new();
    let mut prev = u32::MAX;
    for (pos, &i) in order.iter().enumerate() {
        let l = labels[i as usize];
        if l != prev {
            starts.push(pos as u32);
            prev = l;
        }
    }
    starts.push(order.len() as u32);

    ComponentRanges { order, starts }
}

/// Indices of the edges belonging to each connected component.
pub fn component_indices(edges: &[ScoredPair]) -> Vec<Vec<usize>> {
    let ranges = component_ranges(edges);
    (0..ranges.len())
        .map(|k| ranges.get(k).iter().map(|&i| i as usize).collect())
        .collect()
}

/// Connected components of the edge list, each returned as a list of edges.
pub fn connected_components(edges: &[ScoredPair]) -> Vec<Vec<ScoredPair>> {
    component_indices(edges)
        .into_iter()
        .map(|idx| idx.into_iter().map(|i| edges[i].clone()).collect())
        .collect()
}

/// Condensed distance matrix for a component, plus the id ordering.
pub fn condensed_distance(edges: &[ScoredPair]) -> (Vec<RecordId>, Vec<f64>, usize) {
    let indices: Vec<usize> = (0..edges.len()).collect();
    condensed_distance_idx(edges, &indices)
}

/// Condensed distance matrix for a subset of edges, plus the id ordering.
pub fn condensed_distance_idx(
    edges: &[ScoredPair],
    indices: &[usize],
) -> (Vec<RecordId>, Vec<f64>, usize) {
    let mut candidate_set: Vec<RecordId> = Vec::new();
    for &i in indices {
        let ((a, b), _) = &edges[i];
        candidate_set.push(a.clone());
        candidate_set.push(b.clone());
    }
    candidate_set.sort();
    candidate_set.dedup();
    let n = candidate_set.len();

    let pos: FxHashMap<&RecordId, usize> = candidate_set
        .iter()
        .enumerate()
        .map(|(i, id)| (id, i))
        .collect();

    let mut condensed = vec![1.0f64; n * n.saturating_sub(1) / 2];
    if n < 2 {
        return (candidate_set, condensed, n);
    }
    for &i in indices {
        let ((a, b), score) = &edges[i];
        let row = pos[a];
        let col = pos[b];
        let (row, col) = if row < col { (row, col) } else { (col, row) };
        let index = row * (2 * n - row - 3) / 2 + col - 1;
        condensed[index] = (1.0f32 - *score) as f64;
    }
    (candidate_set, condensed, n)
}

/// `condensed_index` from SciPy's `_hierarchy.pyx`.
#[inline]
fn condensed_index(n: usize, i: usize, j: usize) -> usize {
    if i < j {
        n * i - (i * (i + 1) / 2) + (j - i - 1)
    } else {
        n * j - (j * (j + 1) / 2) + (i - j - 1)
    }
}

#[derive(Clone, Copy)]
struct Pair {
    key: usize,
    value: f64,
}

/// SciPy `_structures.pxi` binary heap, ported verbatim so that tie-breaking
/// matches.
struct Heap {
    index_by_key: Vec<usize>,
    key_by_index: Vec<usize>,
    values: Vec<f64>,
    size: usize,
}

impl Heap {
    fn new(values: &[f64]) -> Self {
        let n = values.len();
        let mut h = Heap {
            index_by_key: (0..n).collect(),
            key_by_index: (0..n).collect(),
            values: values.to_vec(),
            size: n,
        };
        if n > 0 {
            for i in (0..(n / 2)).rev() {
                h.sift_down(i);
            }
        }
        h
    }

    fn get_min(&self) -> Pair {
        Pair {
            key: self.key_by_index[0],
            value: self.values[0],
        }
    }

    fn remove_min(&mut self) {
        self.swap(0, self.size - 1);
        self.size -= 1;
        self.sift_down(0);
    }

    fn change_value(&mut self, key: usize, value: f64) {
        let index = self.index_by_key[key];
        let old_value = self.values[index];
        self.values[index] = value;
        if value < old_value {
            self.sift_up(index);
        } else {
            self.sift_down(index);
        }
    }

    fn sift_up(&mut self, mut index: usize) {
        while index > 0 {
            let parent = (index - 1) >> 1;
            if self.values[parent] > self.values[index] {
                self.swap(index, parent);
                index = parent;
            } else {
                break;
            }
        }
    }

    fn sift_down(&mut self, mut index: usize) {
        let mut child = (index << 1) + 1;
        while child < self.size {
            if child + 1 < self.size && self.values[child + 1] < self.values[child] {
                child += 1;
            }
            if self.values[index] > self.values[child] {
                self.swap(index, child);
                index = child;
                child = (index << 1) + 1;
            } else {
                break;
            }
        }
    }

    fn swap(&mut self, i: usize, j: usize) {
        self.values.swap(i, j);
        let key_i = self.key_by_index[i];
        let key_j = self.key_by_index[j];
        self.key_by_index[i] = key_j;
        self.key_by_index[j] = key_i;
        self.index_by_key[key_i] = j;
        self.index_by_key[key_j] = i;
    }
}

/// SciPy `find_min_dist`.
fn find_min_dist(n: usize, d: &[f64], size: &[i64], x: usize) -> (usize, f64) {
    let mut current_min = f64::INFINITY;
    let mut y = usize::MAX;
    for i in (x + 1)..n {
        if size[i] == 0 {
            continue;
        }
        let dist = d[condensed_index(n, x, i)];
        if dist < current_min {
            current_min = dist;
            y = i;
        }
    }
    (y, current_min)
}

#[inline]
fn centroid_update(d_xi: f64, d_yi: f64, d_xy: f64, size_x: f64, size_y: f64) -> f64 {
    (((size_x * d_xi * d_xi) + (size_y * d_yi * d_yi)
        - (size_x * size_y * d_xy * d_xy) / (size_x + size_y))
        / (size_x + size_y))
        .sqrt()
}

/// Port of SciPy's `fast_linkage` ("Generic Clustering Algorithm", Mullner
/// 2011) for the centroid method.  Returns labeled
/// `(label_a, label_b, distance, size)` rows in merge order.
pub fn linkage_centroid(condensed: &[f64], n: usize) -> Vec<(usize, usize, f64, usize)> {
    let mut d = condensed.to_vec();
    linkage_centroid_in_place(&mut d, n)
}

/// As [`linkage_centroid`], but reuses `d` as the working distance matrix
/// (avoiding a copy of the O(n^2) condensed matrix).
pub fn linkage_centroid_in_place(d: &mut [f64], n: usize) -> Vec<(usize, usize, f64, usize)> {
    if n < 2 {
        return Vec::new();
    }
    let mut z = vec![(0usize, 0usize, 0.0f64, 0usize); n - 1];
    let mut size = vec![1i64; n];
    let mut cluster_id: Vec<usize> = (0..n).collect();

    let m = n - 1;
    let mut neighbor = vec![usize::MAX; m];
    let mut min_dist = vec![0.0f64; m];
    for x in 0..m {
        let (y, dist) = find_min_dist(n, d, &size, x);
        neighbor[x] = y;
        min_dist[x] = dist;
    }
    let mut heap = Heap::new(&min_dist);

    for k in 0..(n - 1) {
        let mut x = 0usize;
        let mut y = 0usize;

        let mut dist = 0.0f64;
        for _ in 0..(n - k) {
            let pair = heap.get_min();
            x = pair.key;
            dist = pair.value;
            y = neighbor[x];
            if dist == d[condensed_index(n, x, y)] {
                break;
            }
            let (yy, dd) = find_min_dist(n, d, &size, x);
            y = yy;
            dist = dd;
            neighbor[x] = y;
            min_dist[x] = dist;
            heap.change_value(x, dist);
        }
        heap.remove_min();

        let mut id_x = cluster_id[x];
        let mut id_y = cluster_id[y];
        let nx = size[x];
        let ny = size[y];
        if id_x > id_y {
            std::mem::swap(&mut id_x, &mut id_y);
        }
        z[k] = (id_x, id_y, dist, (nx + ny) as usize);

        size[x] = 0;
        size[y] = nx + ny;
        cluster_id[y] = n + k;

        for zz in 0..n {
            let nz = size[zz];
            if nz == 0 || zz == y {
                continue;
            }
            let dxi = d[condensed_index(n, zz, x)];
            let dyi = d[condensed_index(n, zz, y)];
            let val = centroid_update(dxi, dyi, dist, nx as f64, ny as f64);
            let ii = condensed_index(n, zz, y);
            d[ii] = val;
        }

        for zz in 0..x {
            if size[zz] > 0 && neighbor[zz] == x {
                neighbor[zz] = y;
            }
        }

        for zz in 0..y {
            if size[zz] == 0 {
                continue;
            }
            let dd = d[condensed_index(n, zz, y)];
            if dd < min_dist[zz] {
                neighbor[zz] = y;
                min_dist[zz] = dd;
                heap.change_value(zz, dd);
            }
        }

        if y < n - 1 {
            let (zz, dd) = find_min_dist(n, d, &size, y);
            if zz != usize::MAX {
                neighbor[y] = zz;
                min_dist[y] = dd;
                heap.change_value(y, dd);
            }
        }
    }

    z
}

/// `scipy.cluster.hierarchy.get_max_dist_for_each_cluster`.
fn get_max_dist_for_each_cluster(
    z: &[(usize, usize, f64, usize)],
    n: usize,
) -> Vec<f64> {
    let mut md = vec![0.0f64; n];
    let mut visited = vec![false; n];
    let mut curr = vec![0usize; n + 1];
    let mut k: isize = 0;
    curr[0] = 2 * n - 2;
    while k >= 0 {
        let root = curr[k as usize] - n;
        let (i_lc, i_rc, _, _) = z[root];
        if i_lc >= n && !visited[i_lc - n] {
            visited[i_lc - n] = true;
            k += 1;
            curr[k as usize] = i_lc;
            continue;
        }
        if i_rc >= n && !visited[i_rc - n] {
            visited[i_rc - n] = true;
            k += 1;
            curr[k as usize] = i_rc;
            continue;
        }
        let mut max_dist = z[root].2;
        if i_lc >= n && md[i_lc - n] > max_dist {
            max_dist = md[i_lc - n];
        }
        if i_rc >= n && md[i_rc - n] > max_dist {
            max_dist = md[i_rc - n];
        }
        md[root] = max_dist;
        k -= 1;
    }
    md
}

/// `scipy.cluster.hierarchy.cluster_monocrit`.
fn cluster_monocrit(
    z: &[(usize, usize, f64, usize)],
    mc: &[f64],
    n: usize,
    cutoff: f64,
) -> Vec<usize> {
    let mut t = vec![0usize; n];
    let mut visited = vec![false; n];
    let mut curr = vec![0usize; n + 1];
    let mut n_cluster: isize = 0;
    let mut cluster_leader: isize = -1;
    let mut k: isize = 0;
    curr[0] = 2 * n - 2;
    while k >= 0 {
        let root = curr[k as usize] - n;
        let (i_lc, i_rc, _, _) = z[root];
        if cluster_leader == -1 && mc[root] <= cutoff {
            cluster_leader = root as isize;
            n_cluster += 1;
        }
        if i_lc >= n && !visited[i_lc - n] {
            visited[i_lc - n] = true;
            k += 1;
            curr[k as usize] = i_lc;
            continue;
        }
        if i_rc >= n && !visited[i_rc - n] {
            visited[i_rc - n] = true;
            k += 1;
            curr[k as usize] = i_rc;
            continue;
        }
        if i_lc < n {
            if cluster_leader == -1 {
                n_cluster += 1;
            }
            t[i_lc] = n_cluster as usize;
        }
        if i_rc < n {
            if cluster_leader == -1 {
                n_cluster += 1;
            }
            t[i_rc] = n_cluster as usize;
        }
        if cluster_leader == root as isize {
            cluster_leader = -1;
        }
        k -= 1;
    }
    t
}

/// `scipy.cluster.hierarchy.fcluster(Z, t, criterion="distance")`.
pub fn fcluster_distance(merges: &[(usize, usize, f64, usize)], n: usize, t: f64) -> Vec<usize> {
    let md = get_max_dist_for_each_cluster(merges, n);
    cluster_monocrit(merges, &md, n, t)
}

/// Per-record confidence scores for a cluster (see `dedupe.clustering.confidences`).
pub fn confidences(cluster: &[usize], squared_distances: &[f64], d: usize) -> Vec<f64> {
    let mut scores_d: FxHashMap<usize, f64> = cluster.iter().map(|i| (*i, 0.0)).collect();
    let c = 2 * d - 3;
    for a in 0..cluster.len() {
        for b in (a + 1)..cluster.len() {
            let i = cluster[a];
            let j = cluster[b];
            let index = i * (c - i) / 2 + j - 1;
            let sq = squared_distances[index];
            *scores_d.get_mut(&i).unwrap() += sq;
            *scores_d.get_mut(&j).unwrap() += sq;
        }
    }
    let mut items: Vec<(usize, f64)> = scores_d.into_iter().collect();
    items.sort_by_key(|(i, _)| *i);
    let len = cluster.len() as f64 - 1.0;
    items
        .into_iter()
        .map(|(_, s)| {
            let v = (s / len).sqrt();
            1.0 - v
        })
        .collect()
}

/// Cluster scored pairs (see `dedupe.clustering.cluster`).
pub fn cluster(scores: &[ScoredPair], threshold: f64, max_components: usize) -> Vec<Cluster> {
    let distance_threshold = 1.0 - threshold;
    let mut out = Vec::new();

    let ranges = component_ranges(scores);
    for k in 0..ranges.len() {
        let mut indices: Vec<u32> = ranges.get(k).to_vec();
        sort_by_score(scores, &mut indices);
        process_component(scores, &indices, threshold, distance_threshold, max_components, &mut out);
    }

    out
}

/// Sort a component's edge indices by ascending score, matching the stable
/// `(label, score)` ordering Python's `union_find` produces.
fn sort_by_score(edges: &[ScoredPair], indices: &mut [u32]) {
    indices.sort_by(|&a, &b| {
        edges[a as usize]
            .1
            .partial_cmp(&edges[b as usize].1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Recursively split over-large components exactly like
/// `dedupe.clustering._connected_components`, then cluster each small one.
fn process_component(
    edges: &[ScoredPair],
    indices: &[u32],
    threshold: f64,
    distance_threshold: f64,
    max_components: usize,
    out: &mut Vec<Cluster>,
) {
    if indices.is_empty() {
        return;
    }

    // Python re-filters components whose edge/node count exceeds the limit to
    // avoid O(n^2) hierarchical clustering.
    let mut needs_filtering = false;
    if indices.len() + 1 > max_components {
        let mut nodes: rustc_hash::FxHashSet<&RecordId> =
            rustc_hash::FxHashSet::default();
        for &i in indices {
            let ((a, b), _) = &edges[i as usize];
            nodes.insert(a);
            nodes.insert(b);
        }
        if nodes.len() > max_components {
            needs_filtering = true;
        }
    }

    if needs_filtering {
        // Sorting by score is only needed for the filtering path.
        let mut sorted: Vec<u32> = indices.to_vec();
        sort_by_score(edges, &mut sorted);
        let min_score = edges[sorted[0] as usize].1 as f64;
        let min_score_logit = min_score.ln() - (1.0 - min_score).ln();
        let cut_threshold = 1.0 / (1.0 + (-min_score_logit - 1.0).exp());
        // `searchsorted(scores, threshold)` on the ascending score array.
        let cut_point = sorted
            .partition_point(|&i| (edges[i as usize].1 as f64) < cut_threshold);
        let keep = &sorted[cut_point.max(2)..];
        if keep.is_empty() {
            return;
        }
        let sub_edges: Vec<ScoredPair> = keep.iter().map(|&i| edges[i as usize].clone()).collect();
        let sub_ranges = component_ranges(&sub_edges);
        for j in 0..sub_ranges.len() {
            let sub_indices: Vec<u32> = sub_ranges.get(j).to_vec();
            process_component(
                &sub_edges,
                &sub_indices,
                threshold,
                distance_threshold,
                max_components,
                out,
            );
        }
        return;
    }

    if indices.len() > 1 {
        let idx: Vec<usize> = indices.iter().map(|&i| i as usize).collect();
        let (i_to_id, mut condensed, n) = condensed_distance_idx(edges, &idx);

        // SciPy/Python squares the f32 condensed distances, then widens.
        let squared: Vec<f64> = condensed
            .iter()
            .map(|x| {
                let v = *x as f32;
                (v * v) as f64
            })
            .collect();

        let merges = linkage_centroid_in_place(&mut condensed, n);
        let partition = fcluster_distance(&merges, n, distance_threshold);

        let mut clusters: indexmap::IndexMap<usize, Vec<usize>> = indexmap::IndexMap::new();
        let mut cluster_order: Vec<usize> = Vec::new();
        for (i, &cluster_id) in partition.iter().enumerate() {
            if !clusters.contains_key(&cluster_id) {
                cluster_order.push(cluster_id);
            }
            clusters.entry(cluster_id).or_default().push(i);
        }

        for cluster_id in cluster_order {
            let cluster = &clusters[&cluster_id];
            if cluster.len() > 1 {
                let scores_v = confidences(cluster, &squared, n);
                let ids: Vec<RecordId> = cluster.iter().map(|i| i_to_id[*i].clone()).collect();
                out.push((ids, scores_v));
            }
        }
    } else {
        let (ids, score) = &edges[indices[0] as usize];
        if (*score as f64) > threshold {
            out.push((
                vec![ids.0.clone(), ids.1.clone()],
                vec![*score as f64, *score as f64],
            ));
        }
    }
}

/// Greedy one-to-one matching over scored pairs.
pub fn greedy_matching(mut dupes: Vec<ScoredPair>) -> Vec<ScoredPair> {
    let mut a_set = std::collections::HashSet::new();
    let mut b_set = std::collections::HashSet::new();
    // Sort by score ascending then reverse (stable), mirroring `sort` + `[::-1]`.
    dupes.sort_by(|x, y| x.1.partial_cmp(&y.1).unwrap_or(std::cmp::Ordering::Equal));
    dupes.reverse();
    let mut out = Vec::new();
    for ((a, b), score) in dupes {
        if !a_set.contains(&a) && !b_set.contains(&b) {
            a_set.insert(a.clone());
            b_set.insert(b.clone());
            out.push(((a, b), score));
        }
    }
    out
}

/// Yield the highest scoring N pairs from each scored block.
pub fn gazette_matching(
    scored_blocks: &[Vec<ScoredPair>],
    threshold: f64,
    n_matches: usize,
) -> Vec<Vec<ScoredPair>> {
    let mut out = Vec::new();
    for block in scored_blocks {
        let mut filtered: Vec<ScoredPair> = block
            .iter()
            .filter(|(_, s)| (*s as f64) > threshold)
            .cloned()
            .collect();
        filtered.sort_by(|x, y| x.1.partial_cmp(&y.1).unwrap_or(std::cmp::Ordering::Equal));
        filtered.reverse();
        if !filtered.is_empty() {
            if n_matches > 0 {
                filtered.truncate(n_matches);
            }
            out.push(filtered);
        }
    }
    out
}

/// Group scored pairs by the first id and yield top matches per group.
pub fn pair_gazette_matching(
    mut scored_pairs: Vec<ScoredPair>,
    threshold: f64,
    n_matches: usize,
) -> Vec<ScoredPair> {
    scored_pairs.sort_by(|x, y| x.0.cmp(&y.0));
    let mut blocks: Vec<Vec<ScoredPair>> = Vec::new();
    let mut current: Vec<ScoredPair> = Vec::new();
    let mut current_key: Option<RecordId> = None;
    for pair in scored_pairs {
        let key = pair.0 .0.clone();
        if Some(&key) != current_key.as_ref() {
            if !current.is_empty() {
                blocks.push(std::mem::take(&mut current));
            }
            current_key = Some(key);
        }
        current.push(pair);
    }
    if !current.is_empty() {
        blocks.push(current);
    }

    let mut out = Vec::new();
    for block in gazette_matching(&blocks, threshold, n_matches) {
        out.extend(block);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rid(i: i64) -> RecordId {
        RecordId::Int(i)
    }

    #[test]
    fn components() {
        let edges: Vec<ScoredPair> = vec![
            ((rid(1), rid(2)), 0.1),
            ((rid(2), rid(3)), 0.2),
            ((rid(4), rid(5)), 0.2),
            ((rid(4), rid(6)), 0.2),
            ((rid(7), rid(9)), 0.2),
            ((rid(8), rid(9)), 0.2),
            ((rid(10), rid(11)), 0.2),
            ((rid(12), rid(13)), 0.2),
            ((rid(12), rid(14)), 0.5),
            ((rid(11), rid(12)), 0.2),
        ];
        let comps = connected_components(&edges);
        let sets: std::collections::HashSet<Vec<(i64, i64)>> = comps
            .iter()
            .map(|c| {
                let mut v: Vec<(i64, i64)> = c
                    .iter()
                    .map(|((a, b), _)| (a.as_int().unwrap(), b.as_int().unwrap()))
                    .collect();
                v.sort_unstable();
                v
            })
            .collect();
        assert_eq!(sets.len(), 4);
        assert!(sets.contains(&vec![(1, 2), (2, 3)]));
        assert!(sets.contains(&vec![(4, 5), (4, 6)]));
        assert!(sets.contains(&vec![(7, 9), (8, 9)]));
        assert!(sets.contains(&vec![(10, 11), (11, 12), (12, 13), (12, 14)]));
    }

    #[test]
    fn greedy() {
        let dupes = vec![
            ((rid(1), rid(5)), 0.1),
            ((rid(1), rid(6)), 0.72),
            ((rid(1), rid(7)), 0.2),
            ((rid(1), rid(8)), 0.6),
            ((rid(2), rid(5)), 0.2),
            ((rid(2), rid(6)), 0.2),
            ((rid(2), rid(7)), 0.72),
            ((rid(2), rid(8)), 0.3),
            ((rid(3), rid(5)), 0.24),
            ((rid(3), rid(6)), 0.72),
            ((rid(3), rid(7)), 0.24),
            ((rid(3), rid(8)), 0.65),
            ((rid(4), rid(5)), 0.63),
            ((rid(4), rid(6)), 0.96),
            ((rid(4), rid(7)), 0.23),
            ((rid(5), rid(8)), 0.24),
        ];
        let got = greedy_matching(dupes);
        let summary: Vec<((i64, i64), f32)> = got
            .iter()
            .map(|((a, b), s)| ((a.as_int().unwrap(), b.as_int().unwrap()), *s))
            .collect();
        assert_eq!(
            summary,
            vec![
                ((4, 6), 0.96),
                ((2, 7), 0.72),
                ((3, 8), 0.65),
                ((1, 5), 0.1)
            ]
        );
    }
}
