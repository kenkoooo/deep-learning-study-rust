extern crate deep_learning_study_rust;
extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;

use deep_learning_study_rust::functions::*;
use deep_learning_study_rust::mnist;
use deep_learning_study_rust::utils::{choice, ArrayChoiceCopy};

use ndarray::prelude::*;
use ndarray::Array;
use ndarray_rand::RandomExt;
use rand::distributions::Normal;
use rand::thread_rng;

use std::cmp;

fn main() {
    let (x_train, t_train, x_test, t_test) = mnist::load_data().unwrap();

    // hyper parameters
    let iters_num = 20000;
    let train_size = x_train.shape()[0];
    let batch_size = 100;
    let learning_rate = 0.1;

    let iter_per_epoch = cmp::max(train_size / batch_size, 1);

    let mut rng = thread_rng();

    let mut network = TwoLayerNet::new(784, 50, 10, 0.01);

    for i in 0..iters_num {
        let mask = choice(&mut rng, train_size, batch_size);
        assert_eq!(mask.len(), batch_size);
        let x_batch = x_train.masked_copy(&mask);
        let t_batch = t_train.masked_copy(&mask);

        let grad = network.gradient(&x_batch, &t_batch);
        network.w1 = network.w1 - grad.w1 * learning_rate;
        network.b1 = network.b1 - grad.b1 * learning_rate;
        network.w2 = network.w2 - grad.w2 * learning_rate;
        network.b2 = network.b2 - grad.b2 * learning_rate;

        if i % iter_per_epoch == 0 {
            let train_acc = network.accuracy(&x_train, &t_train);
            let test_acc = network.accuracy(&x_test, &t_test);
            println!("train acc, test acc | {} {}", train_acc, test_acc);
        }
    }
}

pub struct TwoLayerNet {
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
        TwoLayerNet {
            w1: Array::random((input_size, hidden_size), Normal::new(0.0, 1.0)) * weight_init_std,
            b1: Array::zeros(hidden_size),
            w2: Array::random((hidden_size, output_size), Normal::new(0.0, 1.0)) * weight_init_std,
            b2: Array::zeros(output_size),
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

    fn loss<S: ndarray::Data<Elem = f64>>(
        &mut self,
        x: &ArrayBase<S, Ix2>,
        t: &ArrayBase<S, Ix2>,
    ) -> f64 {
        let y = self.predict(x);
        cross_entropy_error(&y, t)
    }

    fn predict<S: ndarray::Data<Elem = f64>>(
        &mut self,
        x: &ArrayBase<S, Ix2>,
    ) -> ArrayBase<ndarray::OwnedRepr<f64>, Ix2> {
        let a1 = x.dot(&self.w1) + &self.b1;
        let z1 = sigmoid(&a1);
        let a2 = z1.dot(&self.w2) + &self.b2;
        let y = softmax(&a2);
        y
    }

    fn gradient<S: ndarray::Data<Elem = f64>>(
        &mut self,
        x: &ArrayBase<S, Ix2>,
        t: &ArrayBase<S, Ix2>,
    ) -> TwoLayerNet {
        let batch_num = x.shape()[0];

        // forward
        let a1 = x.dot(&self.w1) + &self.b1;
        let z1 = sigmoid(&a1);
        let a2 = z1.dot(&self.w2) + &self.b2;
        let y = softmax(&a2);

        // backward
        let dy = (y - t) / (batch_num as f64);
        let g_w2 = z1.t().dot(&dy);
        let g_b2 = dy.sum_axis(Axis(0));

        let dz1 = dy.dot(&self.w2.t());
        let da1 = sigmoid_grad(&a1) * dz1;
        let g_w1 = x.t().dot(&da1);
        let g_b1 = da1.sum_axis(Axis(0));

        TwoLayerNet {
            b1: g_b1,
            b2: g_b2,
            w1: g_w1,
            w2: g_w2,
        }
    }
}
