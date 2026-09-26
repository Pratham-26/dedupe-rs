//! CRF edit distance, ported from `highered.CRFEditDistance` (which wraps
//! `pyhacrf`).  The CRF parameters are fixed by the library.
//!
//! `CRFEditDistance.__call__` builds a lattice of two features (bias and
//! character match), projects them through the learned parameter matrix, runs
//! the forward algorithm over a two-state machine (match / non-match) and
//! returns the marginal probability of the non-match class.

const LOG_2: f64 = std::f64::consts::LN_2;
const LOG_3: f64 = 1.098_612_288_668_109_8;

/// `highered`'s fixed parameter rows (`model.parameters`), 8 rows x 2 columns.
/// Column 0 is the match-state weight, column 1 the non-match-state weight.
const PARAMS: [[f64; 2]; 8] = [
    [-0.22937526, 0.51326066],
    [0.01038001, -0.13348901],
    [-0.03062821, 0.13769178],
    [0.02024813, -0.01835538],
    [0.09208272, 0.15466022],
    [-0.08170265, -0.02484392],
    [-0.01762858, 0.17504624],
    [0.02800866, -0.04442708],
];

#[inline]
fn logaddexp(x: f64, y: f64) -> f64 {
    if x == y {
        x + LOG_2
    } else {
        let tmp = x - y;
        if tmp > 0.0 {
            x + (-tmp).exp().ln_1p()
        } else if tmp <= 0.0 {
            y + tmp.exp().ln_1p()
        } else {
            tmp
        }
    }
}

#[inline]
fn logsumexp(x: f64, y: f64, z: f64) -> f64 {
    if x == y && y == z {
        x + LOG_3
    } else if (x > y && y > z) || (x > z && z > y) {
        x + ((y - x).exp() + (z - x).exp()).ln_1p()
    } else if x > y && y == z {
        x + ((y - x).exp() * 2.0).ln_1p()
    } else if x == y && y > z {
        x + LOG_2 + ((z - x - LOG_2).exp()).ln_1p()
    } else if x == z && z > y {
        x + LOG_2 + ((y - x - LOG_2).exp()).ln_1p()
    } else if (y > x && x > z) || (y > z && z > x) {
        y + ((x - y).exp() + (z - y).exp()).ln_1p()
    } else if y > x && x == z {
        y + ((x - y).exp() * 2.0).ln_1p()
    } else if y == z && z > x {
        y + LOG_2 + ((x - y - LOG_2).exp()).ln_1p()
    } else if (z > x && x > y) || (z > y && y > x) {
        z + ((x - z).exp() + (y - z).exp()).ln_1p()
    } else if z > x && x == y {
        z + ((x - z).exp() * 2.0).ln_1p()
    } else {
        f64::NAN
    }
}

/// Reproduce `forward_predict(x_dot_parameters, S)` and return the marginal
/// distribution over the `S` classes at the final lattice position.
fn forward_predict(x_dot: &[f64], i_len: usize, j_len: usize, s_count: usize) -> Vec<f64> {
    // x_dot is laid out [i][j][k] with k in 0..(4*S).
    let idx = |i: usize, j: usize, k: usize| (i * j_len + j) * (4 * s_count) + k;
    let mut alpha = x_dot.to_vec();

    let matching = s_count;
    let deletion = 2 * s_count;
    let insertion = 3 * s_count;

    for s in 0..s_count {
        alpha[idx(0, 0, s)] = x_dot[idx(0, 0, s)];
        for i in 1..i_len {
            let insert = alpha[idx(i - 1, 0, s)] + x_dot[idx(i, 0, insertion + s)];
            alpha[idx(i, 0, s)] = x_dot[idx(i, 0, s)] + insert;
        }
        for j in 1..j_len {
            let delete = alpha[idx(0, j - 1, s)] + x_dot[idx(0, j, deletion + s)];
            alpha[idx(0, j, s)] = x_dot[idx(0, j, s)] + delete;
        }
        for j in 1..j_len {
            for i in 1..i_len {
                let insert = alpha[idx(i - 1, j, s)] + x_dot[idx(i, j, insertion + s)];
                let delete = alpha[idx(i, j - 1, s)] + x_dot[idx(i, j, deletion + s)];
                let mtch = alpha[idx(i - 1, j - 1, s)] + x_dot[idx(i, j, matching + s)];
                alpha[idx(i, j, s)] =
                    x_dot[idx(i, j, s)] + logsumexp(insert, delete, mtch);
            }
        }
    }

    let mut z = f64::NEG_INFINITY;
    for s in 0..s_count {
        z = logaddexp(z, alpha[idx(i_len - 1, j_len - 1, s)]);
    }
    let mut out = Vec::with_capacity(s_count);
    for s in 0..s_count {
        out.push((alpha[idx(i_len - 1, j_len - 1, s)] - z).exp());
    }
    out
}

/// `highered.CRFEditDistance.__call__`.
pub fn crf_edit_distance(string_1: &str, string_2: &str) -> f32 {
    let (s1, s2) = if string_1.chars().count() > string_2.chars().count() {
        (string_2, string_1)
    } else {
        (string_1, string_2)
    };
    let a: Vec<char> = s1.chars().collect();
    let b: Vec<char> = s2.chars().collect();
    let i_len = a.len();
    let j_len = b.len();
    if i_len == 0 || j_len == 0 {
        return f32::NAN;
    }
    let s_count = 2;
    let k = 4 * s_count;

    // x[i][j][0] = 1.0 (bias), x[i][j][1] = match indicator.
    // Project through parameters: x_dot[i,j,p] = sum_f x[i,j,f] * params[p,f].
    let mut x_dot = vec![0.0f64; i_len * j_len * k];
    for i in 0..i_len {
        for j in 0..j_len {
            let m = if a[i] == b[j] { 1.0 } else { 0.0 };
            for p in 0..k {
                x_dot[(i * j_len + j) * k + p] = PARAMS[p][0] + m * PARAMS[p][1];
            }
        }
    }

    let probs = forward_predict(&x_dot, i_len, j_len, s_count);
    probs[1] as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crf_bounds() {
        let d = crf_edit_distance("kitten", "sitting");
        assert!((0.0..=1.0).contains(&d), "got {d}");
        let same = crf_edit_distance("abc", "abc");
        assert!(same <= d + 1e-9);
    }
}
