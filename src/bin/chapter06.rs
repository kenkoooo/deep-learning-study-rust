extern crate deep_learning_study_rust;
extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;

use deep_learning_study_rust::functions::accuracy;
use deep_learning_study_rust::layer;
use deep_learning_study_rust::mnist;
use deep_learning_study_rust::utils::*;

use ndarray::{Array, Array1, Array2, ArrayBase, Data, Dimension, Ix1, Ix2};
use ndarray_rand::RandomExt;
use rand::distributions::Normal;
use rand::thread_rng;

struct SGD {
    learning_rate: f64,
}

impl SGD {
    fn new(learning_rate: f64) -> Self {
        SGD {
            learning_rate: learning_rate,
        }
    }

    fn update(&self, grad: Grad, network: &mut TwoLayerNet) {
        network.b1 -= &(self.learning_rate * grad.b1);
        network.b2 -= &(self.learning_rate * grad.b2);
        network.w1 -= &(self.learning_rate * grad.w1);
        network.w2 -= &(self.learning_rate * grad.w2);
    }
}

struct Momentum {
    learning_rate: f64,
    momentum: f64,
    v: Option<Grad>,
}

impl Momentum {
    fn new(learning_rate: f64, momentum: f64) -> Self {
        Momentum {
            learning_rate,
            momentum,
            v: None,
        }
    }

    fn update(&mut self, grad: Grad, network: &mut TwoLayerNet) {
        if self.v.is_none() {
            self.v = Some(Grad {
                b1: Array::zeros(grad.b1.raw_dim()),
                b2: Array::zeros(grad.b2.raw_dim()),
                w1: Array::zeros(grad.w1.raw_dim()),
                w2: Array::zeros(grad.w2.raw_dim()),
            });
        }

        match self.v {
            Some(ref mut v) => {
                v.b1 = self.momentum * &v.b1 - self.learning_rate * grad.b1;
                v.b2 = self.momentum * &v.b2 - self.learning_rate * grad.b2;
                v.w1 = self.momentum * &v.w1 - self.learning_rate * grad.w1;
                v.w2 = self.momentum * &v.w2 - self.learning_rate * grad.w2;
                network.b1 += &v.b1;
                network.b2 += &v.b2;
                network.w1 += &v.w1;
                network.w2 += &v.w2;
            }
            _ => unreachable!(),
        }
    }
}

struct AdaGrad {
    learning_rate: f64,
    v: Option<Grad>,
}

impl AdaGrad {
    fn new(learning_rate: f64) -> Self {
        AdaGrad {
            learning_rate,
            v: None,
        }
    }
    fn update(&mut self, grad: Grad, network: &mut TwoLayerNet) {
        if self.v.is_none() {
            self.v = Some(Grad {
                b1: Array::zeros(grad.b1.raw_dim()),
                b2: Array::zeros(grad.b2.raw_dim()),
                w1: Array::zeros(grad.w1.raw_dim()),
                w2: Array::zeros(grad.w2.raw_dim()),
            });
        }

        match self.v {
            Some(ref mut v) => {
                v.b1 += &grad.b1.powi(2);
                v.b2 += &grad.b2.powi(2);
                v.w1 += &grad.w1.powi(2);
                v.w2 += &grad.w2.powi(2);
                network.b1 -= &(self.learning_rate * grad.b1 / (v.b1.sqrt() + 1e-7));
                network.b2 -= &(self.learning_rate * grad.b2 / (v.b2.sqrt() + 1e-7));
                network.w1 -= &(self.learning_rate * grad.w1 / (v.w1.sqrt() + 1e-7));
                network.w2 -= &(self.learning_rate * grad.w2 / (v.w2.sqrt() + 1e-7));
            }
            _ => unreachable!(),
        }
    }
}

struct Adam {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    mv: Option<(Grad, Grad)>,
    iter: i32,
}

impl Adam {
    fn new(learning_rate: f64, beta1: f64, beta2: f64) -> Self {
        Adam {
            learning_rate,
            beta1,
            beta2,
            iter: 0,
            mv: None,
        }
    }

    fn update(&mut self, grad: Grad, network: &mut TwoLayerNet) {
        if self.mv.is_none() {
            let m = Grad {
                b1: Array::zeros(grad.b1.raw_dim()),
                b2: Array::zeros(grad.b2.raw_dim()),
                w1: Array::zeros(grad.w1.raw_dim()),
                w2: Array::zeros(grad.w2.raw_dim()),
            };
            let v = Grad {
                b1: Array::zeros(grad.b1.raw_dim()),
                b2: Array::zeros(grad.b2.raw_dim()),
                w1: Array::zeros(grad.w1.raw_dim()),
                w2: Array::zeros(grad.w2.raw_dim()),
            };
            self.mv = Some((m, v));
        }

        self.iter += 1;
        let t = self.learning_rate * (1.0 - self.beta2.powi(self.iter)).sqrt()
            / (1.0 - self.beta1.powi(self.iter));

        fn update_inner<D: Dimension>(
            g: Array<f64, D>,
            v: &mut Array<f64, D>,
            m: &mut Array<f64, D>,
            n: &mut Array<f64, D>,
            t: f64,
            beta1: f64,
            beta2: f64,
        ) {
            *v += &((1.0 - beta2) * (g.powi(2) - &(*v)));
            *m += &((1.0 - beta1) * (g - &(*m)));
            *n -= &(t * &(*m) / (v.sqrt() + 1e-7));
        }

        match self.mv {
            Some((ref mut v, ref mut m)) => {
                update_inner(
                    grad.b1,
                    &mut v.b1,
                    &mut m.b1,
                    &mut network.b1,
                    t,
                    self.beta1,
                    self.beta2,
                );
                update_inner(
                    grad.b2,
                    &mut v.b2,
                    &mut m.b2,
                    &mut network.b2,
                    t,
                    self.beta1,
                    self.beta2,
                );
                update_inner(
                    grad.w1,
                    &mut v.w1,
                    &mut m.w1,
                    &mut network.w1,
                    t,
                    self.beta1,
                    self.beta2,
                );
                update_inner(
                    grad.w2,
                    &mut v.w2,
                    &mut m.w2,
                    &mut network.w2,
                    t,
                    self.beta1,
                    self.beta2,
                );
            }
            _ => unreachable!(),
        }
    }
}

fn main() {
    let (x_train, t_train, x_test, t_test) = mnist::load_data().unwrap();

    let mut network = TwoLayerNet::new(784, 50, 10, 0.01);

    let iter_num = 10000;
    let train_size = x_train.shape()[0];
    let batch_size = 100;
    let iter_per_epoch = train_size / batch_size;

    // let optimizer = SGD::new(0.1);
    // let mut optimizer = Momentum::new(0.01, 0.9);
    // let mut optimizer = AdaGrad::new(0.01);
    let mut optimizer = Adam::new(0.001, 0.9, 0.999);

    let mut rng = thread_rng();

    for i in 0..iter_num {
        let batch_mask = choice(&mut rng, train_size, batch_size);
        let x_batch = x_train.masked_copy(&batch_mask);
        let t_batch = t_train.masked_copy(&batch_mask);

        let grad = network.gradient(&x_batch, &t_batch);
        optimizer.update(grad, &mut network);

        if i % iter_per_epoch == 0 {
            let train_acc = network.accuracy(&x_train, &t_train);
            let test_acc = network.accuracy(&x_test, &t_test);
            println!("{} {}", train_acc, test_acc);
        }
    }
}

struct TwoLayerNet {
    w1: Array2<f64>,
    b1: Array1<f64>,
    w2: Array2<f64>,
    b2: Array1<f64>,

    affine1: layer::Affine,
    relu1: layer::Relu,
    affine2: layer::Affine,
    last_layer: layer::SoftmaxWithLoss,
}

struct Grad {
    w1: Array2<f64>,
    b1: Array1<f64>,
    w2: Array2<f64>,
    b2: Array1<f64>,
}

impl TwoLayerNet {
    fn new(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        weight_init_std: f64,
    ) -> Self {
        let w1: Array2<f64> =
            Array::random((input_size, hidden_size), Normal::new(0.0, 1.0)) * weight_init_std;
        let b1: Array1<f64> = Array::zeros(hidden_size);
        let w2: Array2<f64> =
            Array::random((hidden_size, output_size), Normal::new(0.0, 1.0)) * weight_init_std;
        let b2: Array1<f64> = Array::zeros(output_size);

        let affine1 = layer::Affine::new(&w1, &b1);
        let relu1 = layer::Relu::new();
        let affine2 = layer::Affine::new(&w2, &b2);
        let last_layer = layer::SoftmaxWithLoss::new();
        TwoLayerNet {
            w1,
            b1,
            w2,
            b2,
            affine1,
            relu1,
            affine2,
            last_layer,
        }
    }

    fn accuracy<S: ndarray::Data<Elem = f64>, T: ndarray::Data<Elem = f64>>(
        &mut self,
        x: &ArrayBase<S, Ix2>,
        t: &ArrayBase<T, Ix2>,
    ) -> f64 {
        let y = self.predict(x);
        accuracy(&y, t)
    }

    fn loss<S: Data<Elem = f64>>(&mut self, x: &ArrayBase<S, Ix2>, t: &ArrayBase<S, Ix2>) -> f64 {
        let y = self.predict(x);
        self.last_layer.forward(&y, t)
    }

    fn predict<S: Data<Elem = f64>>(
        &mut self,
        x: &ArrayBase<S, Ix2>,
    ) -> ArrayBase<ndarray::OwnedRepr<f64>, Ix2> {
        let x = self.affine1.forward(&self.w1, &self.b1, x);
        let x = self.relu1.forward(&x);
        let x = self.affine2.forward(&self.w2, &self.b2, &x);
        x
    }

    fn gradient<S: ndarray::Data<Elem = f64>>(
        &mut self,
        x: &ArrayBase<S, Ix2>,
        t: &ArrayBase<S, Ix2>,
    ) -> Grad {
        // forward
        self.loss(x, t);

        // backward
        let d_out = self.last_layer.backward(t);
        let d_out = self.affine2.backward(&self.w2, &d_out);
        let d_out = self.relu1.backward(&d_out);
        self.affine1.backward(&self.w1, &d_out);

        Grad {
            w1: &self.affine1.dw * 1.0,
            b1: &self.affine1.db * 1.0,
            w2: &self.affine2.dw * 1.0,
            b2: &self.affine2.db * 1.0,
        }
    }
}
