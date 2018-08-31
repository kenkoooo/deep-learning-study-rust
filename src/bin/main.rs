extern crate deep_learning_study_rust;
extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;

use deep_learning_study_rust::functions::*;
use deep_learning_study_rust::mnist;
use ndarray::prelude::*;
use ndarray::Array;
use ndarray_rand::RandomExt;
use rand::distributions::Normal;
use rand::{thread_rng, Rng};

use std::cmp;
use std::collections::BTreeSet;

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

trait ArrayChoiceCopy<S> {
    fn masked_copy(&self, mask: &Vec<usize>) -> ArrayBase<ndarray::OwnedRepr<f64>, Ix2>;
}

impl<S> ArrayChoiceCopy<S> for ArrayBase<S, Ix2>
where
    S: ndarray::Data<Elem = f64>,
{
    fn masked_copy(&self, mask: &Vec<usize>) -> ArrayBase<ndarray::OwnedRepr<f64>, Ix2> {
        let m = mask.len();
        let cols = self.shape()[1];
        let mut result = Array::zeros((m, cols));
        for row in 0..m {
            result.row_mut(row).assign(&self.row(mask[row]));
        }
        result
    }
}

fn choice<R: Rng>(rng: &mut R, from_size: usize, choice_size: usize) -> Vec<usize> {
    let mut selected = BTreeSet::new();
    let select_fewer = from_size > choice_size * 2;
    let choice_size = cmp::min(from_size, choice_size);
    let choice_size = if select_fewer {
        choice_size
    } else {
        from_size - choice_size
    };
    while selected.len() < choice_size {
        selected.insert(rng.gen_range(0, from_size));
    }
    if select_fewer {
        selected.iter().map(|&c| c).collect()
    } else {
        let mut result = vec![];
        for i in 0..from_size {
            if !selected.contains(&i) {
                result.push(i);
            }
        }
        result
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

    fn predict<S: ndarray::Data<Elem = f64>>(
        &self,
        x: &ArrayBase<S, Ix2>,
    ) -> ArrayBase<ndarray::OwnedRepr<f64>, Ix2> {
        let a1 = x.dot(&self.w1) + &self.b1;
        let z1 = sigmoid(&a1);
        let a2 = z1.dot(&self.w2) + &self.b2;
        let y = softmax(&a2);
        y
    }

    fn loss<S: ndarray::Data<Elem = f64>>(
        &self,
        x: &ArrayBase<S, Ix2>,
        t: &ArrayBase<S, Ix2>,
    ) -> f64 {
        let y = self.predict(x);
        cross_entropy_error(&y, t)
    }

    fn accuracy(&self, x: &Array2<f64>, t: &Array2<f64>) -> f64 {
        let argmax = |x: &ArrayView1<f64>| {
            let mut i = 0;
            assert!(x.len() > 0);
            for j in 1..x.len() {
                if x[j] > x[i] {
                    i = j;
                }
            }
            i
        };
        let y = self.predict(x);
        assert_eq!(t.rows(), y.rows());
        let n = y.rows();

        let mut sum = 0;
        for i in 0..n {
            if argmax(&y.row(i)) == argmax(&t.row(i)) {
                sum += 1;
            }
        }
        sum as f64 / x.shape()[0] as f64
    }

    fn gradient<S: ndarray::Data<Elem = f64>>(
        &self,
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
