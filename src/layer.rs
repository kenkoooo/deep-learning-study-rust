use functions::{cross_entropy_error, softmax};
use ndarray::{Array, ArrayBase, Axis, Data, DataMut, Dimension, Ix1, Ix2, OwnedRepr};
use num_traits;

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
    w: ArrayBase<OwnedRepr<f64>, Ix2>,
    b: ArrayBase<OwnedRepr<f64>, Ix1>,
    x: ArrayBase<OwnedRepr<f64>, Ix2>,
    pub dw: ArrayBase<OwnedRepr<f64>, Ix2>,
    pub db: ArrayBase<OwnedRepr<f64>, Ix1>,
}

impl Affine {
    pub fn new<S: Data<Elem = f64>>(w: &ArrayBase<S, Ix2>, b: &ArrayBase<S, Ix1>) -> Self {
        let mut tw = Array::zeros(w.raw_dim());
        let mut tb = Array::zeros(b.raw_dim());
        tw.assign(w);
        tb.assign(b);
        Affine {
            w: tw,
            b: tb,
            x: Array::zeros(w.raw_dim()),
            dw: Array::zeros(w.raw_dim()),
            db: Array::zeros(b.raw_dim()),
        }
    }

    pub fn forward<S: Data<Elem = f64>>(
        &mut self,
        x: &ArrayBase<S, Ix2>,
    ) -> ArrayBase<OwnedRepr<f64>, Ix2> {
        self.x.assign(x);
        x.dot(&self.w) + &self.b
    }

    pub fn backward<S: Data<Elem = f64>>(
        &mut self,
        d_out: &ArrayBase<S, Ix2>,
    ) -> ArrayBase<OwnedRepr<f64>, Ix2> {
        let dx = d_out.dot(&self.w.t());
        self.dw.assign(&self.x.t().dot(d_out));
        self.db.assign(&d_out.sum_axis(Axis(0)));
        dx
    }
}

pub struct SoftmaxWithLoss {
    y: ArrayBase<OwnedRepr<f64>, Ix2>,
    t: ArrayBase<OwnedRepr<f64>, Ix2>,
    loss: f64,
}

impl SoftmaxWithLoss {
    pub fn new() -> Self {
        SoftmaxWithLoss {
            y: Array::zeros((0, 0)),
            t: Array::zeros((0, 0)),
            loss: 0.0,
        }
    }
    pub fn forward<S: Data<Elem = f64> + DataMut, T: Data<Elem = f64>>(
        &mut self,
        x: &ArrayBase<S, Ix2>,
        t: &ArrayBase<T, Ix2>,
    ) -> f64 {
        self.t = Array::zeros(t.raw_dim());
        self.t.assign(t);
        self.y = softmax(x);
        self.loss = cross_entropy_error(&self.y, &self.t);
        self.loss
    }

    pub fn backward(&mut self) -> ArrayBase<OwnedRepr<f64>, Ix2> {
        (&self.y - &self.t) / self.t.shape()[0] as f64
    }
}
