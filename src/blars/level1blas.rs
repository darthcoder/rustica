use ndarray::Array1;
use num_complex::{Complex32, Complex64};

// SAXPY function (y = a * x + y) for single-precision floats (f32)
pub fn saxpy(a: f32, x: &Array1<f32>, y: &mut Array1<f32>) {
    assert_eq!(x.len(), y.len());
    for i in 0..x.len() {
        y[i] = a * x[i] + y[i];
    }
}

// DAXPY function (y = a * x + y) for double-precision floats (f64)
pub fn daxpy(a: f64, x: &Array1<f64>, y: &mut Array1<f64>) {
    assert_eq!(x.len(), y.len());
    for i in 0..x.len() {
        y[i] = a * x[i] + y[i];
    }
}

// CAXPY function (y = a * x + y) for single-precision complex numbers (Complex32)
pub fn caxpy(a: Complex32, x: &Array1<Complex32>, y: &mut Array1<Complex32>) {
    assert_eq!(x.len(), y.len());
    for i in 0..x.len() {
        y[i] = a * x[i] + y[i];
    }
}

// ZAXPY function (y = a * x + y) for double-precision complex numbers (Complex64)
pub fn zaxpy(a: Complex64, x: &Array1<Complex64>, y: &mut Array1<Complex64>) {
    assert_eq!(x.len(), y.len());
    for i in 0..x.len() {
        y[i] = a * x[i] + y[i];
    }
}

// Other Level 1 BLAS functions...
