//! Dense f32 GEMM: `C[m, n] = A[m, k] @ W[n, k]^T`.
//!
//! The kernel itself is delegated to the `gemm` crate — the same choice the
//! sibling `qwen3-asr-wgpu` port makes. It ships tuned AVX2/AVX-512 micro-kernels
//! and handles the threading, so this module only owns the layout mapping and the
//! weight storage. A hand-written AVX2 kernel here peaked at roughly an eighth of
//! one core's FMA throughput; there is no reason to keep maintaining it.
//!
//! Layout mapping (the crate takes `(cs, rs)` in that order):
//!
//! ```text
//! C  row-major [m][n]  -> dst_cs = 1,  dst_rs = n
//! A  row-major [m][k]  -> lhs_cs = 1,  lhs_rs = k
//! W  row-major [n][k]  -> rhs_cs = k,  rhs_rs = 1     (as B = [k][n])
//! ```
//!
//! Weights are stored exactly as the checkpoint delivers them: `W` is already
//! `[n][k]` row-major, which is what `rhs_cs = k, rhs_rs = 1` addresses, so no
//! repacking pass exists at all.

/// A `(n, k)` weight matrix, row-major, exactly as the checkpoint stores it.
pub struct PackW {
    n: usize,
    k: usize,
    row_major: Vec<f32>,
}

impl PackW {
    pub fn from_rows(w: &[f32], n: usize, k: usize) -> Self {
        debug_assert_eq!(w.len(), n * k, "weight is {n}x{k}");
        Self {
            n,
            k,
            row_major: w.to_vec(),
        }
    }

    /// Same as [`Self::from_rows`] but takes ownership, so a caller that already
    /// owns a freshly converted `Vec<f32>` pays no second copy.
    pub fn from_vec(row_major: Vec<f32>, n: usize, k: usize) -> Self {
        debug_assert_eq!(row_major.len(), n * k, "weight is {n}x{k}");
        Self { n, k, row_major }
    }

    pub fn rows(&self) -> usize {
        self.n
    }

    pub fn cols(&self) -> usize {
        self.k
    }

    pub fn row_major(&self) -> &[f32] {
        &self.row_major
    }
}

/// `C[m, n] = A[m, k] @ W[n, k]^T`, all row-major, `C` fully overwritten.
pub fn gemm(a: &[f32], w: &PackW, c: &mut [f32]) {
    let m = a.len().checked_div(w.k).unwrap_or(0);
    gemm_into(a, w, c, m);
}

/// [`gemm`] with an explicit row count, so callers can pass a padded buffer.
pub fn gemm_into(a: &[f32], w: &PackW, c: &mut [f32], m: usize) {
    gemm_rows_parallel(a, &w.row_major, c, m, w.n, w.k);
}

/// `C[m, n] = A[m, k] @ B[n, k]^T` for a bare row-major `B`, single threaded.
///
/// Per-head attention launches one of these per head from inside its own Rayon
/// task, so it must not spawn workers of its own.
pub fn gemm_rows_serial(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    call(a, b, c, m, n, k, gemm::Parallelism::None);
}

/// The whole-machine variant, used for the model's own Linear layers.
pub fn gemm_rows_parallel(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    call(a, b, c, m, n, k, gemm::Parallelism::Rayon(0));
}

/// `C[m, n] = A[m, k] @ B[k, n]` with a **row-major `[k][n]`** right operand,
/// single threaded. [`gemm_rows_serial`] addresses `B` as `[n][k]` instead, which
/// is what a Linear weight looks like; attention's `scores @ V` needs this form.
pub fn gemm_right_row_major_serial(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    assert!(a.len() >= m * k, "A is too short for {m} rows of {k}");
    assert!(b.len() >= k * n, "B is too short for {k} rows of {n}");
    assert!(c.len() >= m * n, "C is too short for {m} rows of {n}");
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    // SAFETY: buffers are sized by the asserts; strides stay inside `b`; the call
    // is synchronous. `dst = alpha*dst + beta*lhs*rhs` with alpha scaling the
    // destination, so overwriting passes alpha = 0, beta = 1.
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            c.as_mut_ptr(),
            1,
            n as isize,
            false,
            a.as_ptr(),
            1,
            k as isize,
            b.as_ptr(),
            1,
            n as isize,
            0.0f32,
            1.0f32,
            false,
            false,
            false,
            gemm::Parallelism::None,
        );
    }
}

fn call(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    parallelism: gemm::Parallelism,
) {
    assert!(a.len() >= m * k, "A is too short for {m} rows of {k}");
    assert!(b.len() >= n * k, "B is too short for {n} rows of {k}");
    assert!(c.len() >= m * n, "C is too short for {m} rows of {n}");
    if m == 0 || n == 0 || k == 0 {
        return;
    }

    // SAFETY: the three buffers are sized by the asserts above, every stride stays
    // inside its buffer, and the call is synchronous — it returns only after its
    // workers have joined, so `c` is not aliased while borrowed here.
    //
    // The crate computes `dst = alpha*dst + beta*lhs*rhs`, i.e. `alpha` scales the
    // *destination* and `beta` the product — the opposite of the BLAS convention.
    // Overwriting `C` therefore passes alpha = 0 and beta = 1.
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            c.as_mut_ptr(),
            1,
            n as isize,
            false,
            a.as_ptr(),
            1,
            k as isize,
            b.as_ptr(),
            k as isize,
            1,
            0.0f32,
            1.0f32,
            false,
            false,
            false,
            parallelism,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(a: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for row in 0..m {
            for column in 0..n {
                let mut sum = 0.0f32;
                for index in 0..k {
                    sum += a[row * k + index] * w[column * k + index];
                }
                c[row * n + column] = sum;
            }
        }
        c
    }

    fn run(m: usize, n: usize, k: usize) {
        let a: Vec<f32> = (0..m * k).map(|i| (i % 13) as f32 * 0.25 - 1.5).collect();
        let w: Vec<f32> = (0..n * k).map(|i| (i % 7) as f32 * 0.5 - 1.0).collect();
        let packed = PackW::from_rows(&w, n, k);
        let mut c = vec![0.0f32; m * n];
        gemm_into(&a, &packed, &mut c, m);
        let expected = reference(&a, &w, m, n, k);
        let worst = c
            .iter()
            .zip(&expected)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-3, "{m}x{n}x{k}: max diff {worst}");
    }

    #[test]
    fn matches_the_reference_for_odd_shapes() {
        // Shapes off PANEL/micro-tile boundaries plus the real encoder shapes.
        for (m, n, k) in [
            (1, 1, 1),
            (1, 17, 5),
            (7, 16, 3),
            (6, 1024, 33),
            (13, 200, 64),
            (189, 4096, 1024),
            (189, 1024, 1024),
        ] {
            run(m, n, k);
        }
    }

    #[test]
    fn serial_entries_match_naive() {
        let (m, n, k) = (5usize, 3usize, 4usize);
        let a: Vec<f32> = (0..m * k).map(|i| (i % 5) as f32 - 2.0).collect();
        let b_nk: Vec<f32> = (0..n * k).map(|i| (i % 7) as f32 * 0.5 - 1.0).collect();

        // gemm_rows_serial: B is [n][k], contract over k.
        let mut c = vec![0.0f32; m * n];
        gemm_rows_serial(&a, &b_nk, &mut c, m, n, k);
        for i in 0..m {
            for j in 0..n {
                let mut want = 0.0f32;
                for kk in 0..k {
                    want += a[i * k + kk] * b_nk[j * k + kk];
                }
                assert!((c[i * n + j] - want).abs() < 1e-5, "rows_serial {i},{j}");
            }
        }

        // gemm_right_row_major_serial: B is [k][n], contract over k.
        let b_kn: Vec<f32> = (0..k * n).map(|i| (i % 6) as f32 * 0.25 - 0.5).collect();
        let mut d = vec![0.0f32; m * n];
        gemm_right_row_major_serial(&a, &b_kn, &mut d, m, n, k);
        for i in 0..m {
            for j in 0..n {
                let mut want = 0.0f32;
                for kk in 0..k {
                    want += a[i * k + kk] * b_kn[kk * n + j];
                }
                assert!(
                    (d[i * n + j] - want).abs() < 1e-5,
                    "right_row_major {i},{j}"
                );
            }
        }
    }

    #[test]
    fn weights_are_kept_verbatim() {
        let (n, k) = (19, 5);
        let w: Vec<f32> = (0..n * k).map(|i| i as f32).collect();
        let packed = PackW::from_rows(&w, n, k);
        assert_eq!(packed.rows(), n);
        assert_eq!(packed.cols(), k);
        assert_eq!(packed.row_major(), &w[..]);
    }
}
