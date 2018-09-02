use functions::{cross_entropy_error, softmax};
use ndarray::{
    Array, Array1, Array2, ArrayBase, Axis, Data, DataMut, Dimension, Ix1, Ix2, OwnedRepr,
};
use num_traits;

use std::rc::Rc;

pub struct MulLayer<T> {
    x: T,
    y: T,
}

impl<T: num_traits::Float> MulLayer<T> {
    pub fn new() -> Self {
        MulLayer {
            x: num_traits::zero(),
            y: num_traits::zero(),
        }
    }

    pub fn forward(&mut self, x: T, y: T) -> T {
        self.x = x;
        self.y = y;
        x * y
    }

    pub fn backward(&self, out: T) -> (T, T) {
        let dx = out * self.y;
        let dy = out * self.x;
        (dx, dy)
    }
}

pub struct AddLayer<T> {
    t: ::std::marker::PhantomData<T>,
}

impl<T: num_traits::Float> AddLayer<T> {
    pub fn new() -> Self {
        AddLayer {
            t: ::std::marker::PhantomData,
        }
    }
    pub fn forward(&self, x: T, y: T) -> T {
        x + y
    }

    pub fn backward(&self, out: T) -> (T, T) {
        (out, out)
    }
}

pub struct Relu {
    mask: ArrayBase<OwnedRepr<f64>, Ix2>,
}

impl Relu {
    pub fn new() -> Self {
        Relu {
            mask: Array::zeros((0, 0)),
        }
    }
    pub fn forward<S: Data<Elem = f64>>(
        &mut self,
        x: &ArrayBase<S, Ix2>,
    ) -> ArrayBase<OwnedRepr<f64>, Ix2> {
        self.mask = x.mapv(|x| if x <= 0.0 { 0.0 } else { 1.0 });
        x.mapv(|x| if x <= 0.0 { 0.0 } else { x })
    }

    pub fn backward<S: Data<Elem = f64>>(
        &self,
        d_out: &ArrayBase<S, Ix2>,
    ) -> ArrayBase<OwnedRepr<f64>, Ix2> {
        d_out * &self.mask
    }
}

struct Sigmoid {
    out: ArrayBase<OwnedRepr<f64>, Ix2>,
}

impl Sigmoid {
    pub fn forward<S: Data<Elem = f64>>(&mut self, x: &ArrayBase<S, Ix2>) {
        self.out = 1.0 / x.mapv(|x| (-x).exp() + 1.0);
    }

    pub fn backward<S: Data<Elem = f64>>(
        &self,
        d_out: &ArrayBase<S, Ix2>,
    ) -> ArrayBase<OwnedRepr<f64>, Ix2> {
        d_out * &self.out * (1.0 - &self.out)
    }
}

pub struct Affine {
    x: Array2<f64>,
    pub dw: Array2<f64>,
    pub db: Array1<f64>,
}

impl Affine {
    pub fn new(w: &Array2<f64>, b: &Array1<f64>) -> Self {
        let dw = Array::zeros(w.raw_dim());
        let db = Array::zeros(b.raw_dim());
        Affine {
            x: Array::zeros((0, 0)),
            dw: dw,
            db: db,
        }
    }

    pub fn forward<S: Data<Elem = f64>, T: Data<Elem = f64>, U: Data<Elem = f64>>(
        &mut self,
        w: &ArrayBase<S, Ix2>,
        b: &ArrayBase<T, Ix1>,
        x: &ArrayBase<U, Ix2>,
    ) -> ArrayBase<OwnedRepr<f64>, Ix2> {
        self.x = x * 1.0;
        x.dot(w) + b
    }

    pub fn backward<S: Data<Elem = f64>, T: Data<Elem = f64>>(
        &mut self,
        w: &ArrayBase<S, Ix2>,
        d_out: &ArrayBase<T, Ix2>,
    ) -> ArrayBase<OwnedRepr<f64>, Ix2> {
        let dx = d_out.dot(&w.t());
        self.dw.assign(&self.x.t().dot(d_out));
        self.db.assign(&d_out.sum_axis(Axis(0)));
        dx
    }
}

pub struct SoftmaxWithLoss {
    y: ArrayBase<OwnedRepr<f64>, Ix2>,
    loss: f64,
}

impl SoftmaxWithLoss {
    pub fn new() -> Self {
        SoftmaxWithLoss {
            y: Array::zeros((0, 0)),
            loss: 0.0,
        }
    }
    pub fn forward<S: Data<Elem = f64> + DataMut, T: Data<Elem = f64>>(
        &mut self,
        x: &ArrayBase<S, Ix2>,
        label: &ArrayBase<T, Ix2>,
    ) -> f64 {
        self.y = softmax(x);
        self.loss = cross_entropy_error(&self.y, label);
        self.loss
    }

    pub fn backward<T: Data<Elem = f64>>(
        &mut self,
        label: &ArrayBase<T, Ix2>,
    ) -> ArrayBase<OwnedRepr<f64>, Ix2> {
        (&self.y - label) / label.shape()[0] as f64
    }
}
