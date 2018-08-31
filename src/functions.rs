use ndarray::prelude::*;
use ndarray::{Array, Data, DataMut, Dimension, OwnedRepr};

pub fn sigmoid_grad<S, D>(x: &ArrayBase<S, D>) -> ArrayBase<OwnedRepr<f64>, D>
where
    S: Data<Elem = f64>,
    D: Dimension,
{
    (1.0 - sigmoid(x)) * sigmoid(x)
}

pub fn cross_entropy_error<S: Data<Elem = f64>, T: Data<Elem = f64>>(
    y: &ArrayBase<S, Ix2>,
    t: &ArrayBase<T, Ix2>,
) -> f64 {
    let batch_size = y.shape()[0];
    -(log(&(y + 1e-7)) * t).scalar_sum() / batch_size as f64
}

pub fn log<S: Data<Elem = f64>>(a: &ArrayBase<S, Ix2>) -> ArrayBase<OwnedRepr<f64>, Ix2> {
    a.mapv(f64::ln)
}

pub fn softmax<S>(x: &ArrayBase<S, Ix2>) -> ArrayBase<OwnedRepr<f64>, Ix2>
where
    S: Data<Elem = f64> + DataMut,
{
    let n = x.rows();
    let d = x.raw_dim();
    let mut y = Array::zeros(d);
    for i in 0..n {
        let max_scalar = max(&x.row(i)).unwrap();
        let exp = x.row(i).mapv(|f| (f - max_scalar).exp());
        let sum = exp.scalar_sum();
        y.row_mut(i).assign(&(exp / sum));
    }
    y
}

pub fn sigmoid<S, D>(x: &ArrayBase<S, D>) -> ArrayBase<OwnedRepr<f64>, D>
where
    S: Data<Elem = f64>,
    D: Dimension,
{
    1.0 / x.mapv(|f| (-f).exp() + 1.0)
}

pub fn max<S, D>(a: &ArrayBase<S, D>) -> Option<f64>
where
    S: Data<Elem = f64>,
    D: Dimension,
{
    let mut result: Option<f64> = None;
    for &a in a.iter() {
        let p = result.unwrap_or(a);
        result = Some(if p > a { p } else { a });
    }
    result
}
