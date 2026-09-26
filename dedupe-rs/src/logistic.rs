//! Logistic regression and grid-search cross validation, replacing the
//! scikit-learn estimators used by dedupe (`LogisticRegression`,
//! `GridSearchCV`).

use rand::SeedableRng;

/// A pairwise classifier.
pub trait Classifier: Send + Sync {
    fn fit(&mut self, x: &[Vec<f64>], y: &[usize]);
    /// Probability that each pair is a match.
    fn predict_proba(&self, x: &[Vec<f64>]) -> Vec<f64>;
    /// Probability for a single pair.
    fn predict_proba_one(&self, x: &[f64]) -> f64 {
        self.predict_proba(&[x.to_vec()])[0]
    }
    /// Probability for a single f32 feature row (no allocation by default).
    fn predict_proba_f32(&self, x: &[f32]) -> f64 {
        let mut z = 0.0;
        let _ = &mut z;
        let f: Vec<f64> = x.iter().map(|v| *v as f64).collect();
        self.predict_proba_one(&f)
    }
    fn clone_box(&self) -> Box<dyn Classifier>;
}

#[inline]
fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L2-regularized logistic regression (matching sklearn's objective).
#[derive(Debug, Clone)]
pub struct LogisticRegression {
    pub c: f64,
    pub coef: Vec<f64>,
    pub intercept: f64,
    pub max_iter: usize,
    pub tol: f64,
}

impl Default for LogisticRegression {
    fn default() -> Self {
        Self {
            c: 1.0,
            coef: Vec::new(),
            intercept: 0.0,
            max_iter: 200,
            tol: 1e-8,
        }
    }
}

impl LogisticRegression {
    pub fn new(c: f64) -> Self {
        Self {
            c,
            ..Default::default()
        }
    }

    fn objective(&self, w: &[f64], b: f64, x: &[Vec<f64>], y: &[f64], c: f64) -> f64 {
        let mut loss = 0.5 * dot(w, w);
        for (xi, yi) in x.iter().zip(y.iter()) {
            let z = dot(w, xi) + b;
            // log(1 + exp(-y z))
            loss += c * (-yi * z).exp().ln_1p();
        }
        loss
    }

    /// Fit the model.  Returns the number of Newton iterations used.
    pub fn fit_with(&mut self, x: &[Vec<f64>], y: &[usize], c: f64) -> usize {
        let n = x.len();
        let d = if n > 0 { x[0].len() } else { 0 };
        let yv: Vec<f64> = y.iter().map(|&v| v as f64).collect();
        let mut w = vec![0.0f64; d];
        let mut b = 0.0f64;

        let mut iters = 0;
        for _ in 0..self.max_iter {
            iters += 1;
            let mut g = vec![0.0f64; d + 1];
            let mut h = vec![vec![0.0f64; d + 1]; d + 1];

            for i in 0..n {
                let z = dot(&w, &x[i]) + b;
                let p = sigmoid(z);
                let r = p - yv[i];
                for j in 0..d {
                    g[j] += r * x[i][j];
                }
                g[d] += r;
                let wgt = p * (1.0 - p);
                for j in 0..d {
                    let xj = x[i][j] * wgt;
                    for k in 0..d {
                        h[j][k] += xj * x[i][k];
                    }
                    h[j][d] += xj;
                    h[d][j] += xj;
                }
                h[d][d] += wgt;
            }

            // Regularization (L2, no penalty on the intercept).
            for j in 0..d {
                g[j] = w[j] + c * g[j];
                for k in 0..d {
                    h[j][k] *= c;
                }
                h[j][j] += 1.0;
            }
            g[d] *= c;
            for j in 0..d {
                h[j][d] *= c;
                h[d][j] *= c;
            }
            h[d][d] *= c;

            let gmax = g.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            if gmax < self.tol {
                break;
            }

            let mut rhs: Vec<f64> = g.iter().map(|v| -v).collect();
            let Some(delta) = solve(&h, &mut rhs) else {
                break;
            };

            // Backtracking line search on the objective.
            let j0 = self.objective(&w, b, x, &yv, c);
            let gd = dot(&g, &delta);
            let mut step = 1.0f64;
            loop {
                let w2: Vec<f64> = w
                    .iter()
                    .zip(delta.iter())
                    .map(|(a, d)| a + step * d)
                    .collect();
                let b2 = b + step * delta[d];
                let j1 = self.objective(&w2, b2, x, &yv, c);
                if j1 <= j0 + 1e-4 * step * gd || step < 1e-10 {
                    w = w2;
                    b = b2;
                    break;
                }
                step *= 0.5;
            }
        }

        self.c = c;
        self.coef = w;
        self.intercept = b;
        iters
    }

    pub fn from_coef(coef: Vec<f64>, intercept: f64, c: f64) -> Self {
        Self {
            c,
            coef,
            intercept,
            max_iter: 200,
            tol: 1e-8,
        }
    }

    pub fn predict_proba_one(&self, x: &[f64]) -> f64 {
        sigmoid(dot(&self.coef, x) + self.intercept)
    }
}

impl Classifier for LogisticRegression {
    fn fit(&mut self, x: &[Vec<f64>], y: &[usize]) {
        let c = self.c;
        self.fit_with(x, y, c);
    }

    fn predict_proba(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter().map(|xi| self.predict_proba_one(xi)).collect()
    }

    fn predict_proba_f32(&self, x: &[f32]) -> f64 {
        let mut z = self.intercept;
        for (c, v) in self.coef.iter().zip(x.iter()) {
            z += c * (*v as f64);
        }
        crate::logistic::sigmoid(z)
    }

    fn clone_box(&self) -> Box<dyn Classifier> {
        Box::new(self.clone())
    }
}

/// Solve `A x = b` in place with partial pivoting.  Returns `None` if singular.
fn solve(a: &[Vec<f64>], b: &mut [f64]) -> Option<Vec<f64>> {
    let n = b.len();
    let mut m: Vec<Vec<f64>> = a.to_vec();
    let mut x = b.to_vec();

    for col in 0..n {
        // pivot
        let mut pivot = col;
        let mut best = m[col][col].abs();
        for row in (col + 1)..n {
            if m[row][col].abs() > best {
                best = m[row][col].abs();
                pivot = row;
            }
        }
        if best < 1e-14 {
            return None;
        }
        if pivot != col {
            m.swap(col, pivot);
            x.swap(col, pivot);
        }
        let diag = m[col][col];
        for row in (col + 1)..n {
            let factor = m[row][col] / diag;
            if factor == 0.0 {
                continue;
            }
            for k in col..n {
                m[row][k] -= factor * m[col][k];
            }
            x[row] -= factor * x[col];
        }
    }

    let mut out = vec![0.0f64; n];
    for col in (0..n).rev() {
        let mut sum = x[col];
        for k in (col + 1)..n {
            sum -= m[col][k] * out[k];
        }
        out[col] = sum / m[col][col];
    }
    Some(out)
}

/// Stratified k-fold split (mirrors `sklearn.model_selection.StratifiedKFold`
/// with `shuffle=False`).
pub fn stratified_folds(y: &[usize], n_splits: usize) -> Vec<Vec<usize>> {
    let n = y.len();
    let mut folds: Vec<Vec<usize>> = vec![Vec::new(); n_splits];
    let classes: Vec<usize> = {
        let mut c: Vec<usize> = y.to_vec();
        c.sort_unstable();
        c.dedup();
        c
    };
    for class in classes {
        let idx: Vec<usize> = (0..n).filter(|&i| y[i] == class).collect();
        for (k, &i) in idx.iter().enumerate() {
            folds[k % n_splits].push(i);
        }
    }
    folds
}

/// Grid search over the regularization strength, scoring by F1.
#[derive(Debug, Clone)]
pub struct GridSearchCv {
    pub c_values: Vec<f64>,
    pub n_splits: usize,
    pub best_c: f64,
    pub model: LogisticRegression,
}

impl Default for GridSearchCv {
    fn default() -> Self {
        Self {
            c_values: vec![0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
            n_splits: 5,
            best_c: 1.0,
            model: LogisticRegression::default(),
        }
    }
}

fn f1_score(y_true: &[usize], y_pred: &[usize]) -> f64 {
    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut fnn = 0.0;
    for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
        if p == 1 && t == 1 {
            tp += 1.0;
        } else if p == 1 && t == 0 {
            fp += 1.0;
        } else if p == 0 && t == 1 {
            fnn += 1.0;
        }
    }
    if tp == 0.0 {
        return 0.0;
    }
    2.0 * tp / (2.0 * tp + fp + fnn)
}

impl Classifier for GridSearchCv {
    fn fit(&mut self, x: &[Vec<f64>], y: &[usize]) {
        let n = y.len();
        let classes: std::collections::HashSet<usize> = y.iter().copied().collect();
        let min_class_count = classes
            .iter()
            .map(|c| y.iter().filter(|v| *v == c).count())
            .min()
            .unwrap_or(0);
        let mut n_splits = self.n_splits.min(min_class_count);
        if n_splits < 2 {
            n_splits = 0;
        }

        let mut best_c = self.c_values[0];
        let mut best_score = f64::NEG_INFINITY;

        if n_splits >= 2 && classes.len() >= 2 {
            let folds = stratified_folds(y, n_splits);
            for &c in &self.c_values {
                let mut scores = Vec::new();
                for fold in 0..n_splits {
                    let test_idx = &folds[fold];
                    let train_idx: Vec<usize> = (0..n)
                        .filter(|i| !test_idx.contains(i))
                        .collect();
                    if train_idx.is_empty() {
                        continue;
                    }
                    let xtr: Vec<Vec<f64>> = train_idx.iter().map(|&i| x[i].clone()).collect();
                    let ytr: Vec<usize> = train_idx.iter().map(|&i| y[i]).collect();
                    let xte: Vec<Vec<f64>> = test_idx.iter().map(|&i| x[i].clone()).collect();
                    let yte: Vec<usize> = test_idx.iter().map(|&i| y[i]).collect();

                    if ytr.iter().all(|v| *v == ytr[0]) {
                        continue;
                    }
                    let mut m = LogisticRegression::new(c);
                    m.fit(&xtr, &ytr);
                    let probs = m.predict_proba(&xte);
                    let preds: Vec<usize> = probs.iter().map(|p| (*p > 0.5) as usize).collect();
                    scores.push(f1_score(&yte, &preds));
                }
                let mean = if scores.is_empty() {
                    0.0
                } else {
                    scores.iter().sum::<f64>() / scores.len() as f64
                };
                if mean > best_score {
                    best_score = mean;
                    best_c = c;
                }
            }
        }

        self.best_c = best_c;
        self.model = LogisticRegression::new(best_c);
        self.model.fit(x, y);
    }

    fn predict_proba(&self, x: &[Vec<f64>]) -> Vec<f64> {
        self.model.predict_proba(x)
    }

    fn clone_box(&self) -> Box<dyn Classifier> {
        Box::new(self.clone())
    }
}

/// A fixed random classifier, useful for tests.
#[derive(Debug, Clone)]
pub struct MockClassifier {
    pub weights: Vec<f64>,
    pub rng_seed: u64,
}

impl Classifier for MockClassifier {
    fn fit(&mut self, _x: &[Vec<f64>], _y: &[usize]) {}
    fn predict_proba(&self, x: &[Vec<f64>]) -> Vec<f64> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.rng_seed);
        x.iter()
            .map(|xi| {
                let _ = &mut rng;
                sigmoid(dot(&self.weights, xi))
            })
            .collect()
    }
    fn clone_box(&self) -> Box<dyn Classifier> {
        Box::new(self.clone())
    }
}

/// A deterministic random state helper.
pub fn seeded_rng(seed: u64) -> impl rand::Rng {
    rand::rngs::StdRng::seed_from_u64(seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn separates_two_clusters() {
        let x = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![1.0, 1.0],
            vec![1.1, 1.0],
            vec![1.0, 1.1],
        ];
        let y = vec![0, 0, 0, 1, 1, 1];
        let mut lr = LogisticRegression::new(1.0);
        lr.fit(&x, &y);
        let p = lr.predict_proba(&x);
        assert!(p[0] < 0.5 && p[3] > 0.5);
    }

    #[test]
    fn f1_perfect() {
        assert_eq!(f1_score(&[1, 0, 1], &[1, 0, 1]), 1.0);
    }
}
