extern crate deep_learning_study_rust;
extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;

use deep_learning_study_rust::mnist;
use ndarray::prelude::*;
use ndarray::{Array, Slice};
use ndarray_rand::RandomExt;
use rand::distributions::Range;
use rand::distributions::*;
use rand::{thread_rng, Rng};

use std::collections::BTreeSet;

fn get_train_data() -> (Array2<f64>, Vec<u8>) {
    let x_train = mnist::load_images_array("dataset/train-images-idx3-ubyte").unwrap();
    let t_train = mnist::load_labels("dataset/train-labels-idx1-ubyte").unwrap();
    (x_train, t_train)
}

fn get_test_data() -> (Array2<f64>, Vec<u8>) {
    let x_train = mnist::load_images_array("dataset/t10k-images-idx3-ubyte").unwrap();
    let t_train = mnist::load_labels("dataset/t10k-labels-idx1-ubyte").unwrap();
    (x_train, t_train)
}

fn label_to_one_hot(t: &Vec<u8>) -> Array2<f64> {
    let n = t.len();
    let mut result = Array::zeros((n, 10));
    for i in 0..n {
        result.row_mut(i)[t[i] as usize] = 1.0;
    }
    result
}

fn main() {
    // chapter 04
    let (x_train, t_train) = get_train_data();
    let t_train = label_to_one_hot(&t_train);
    println!("train data loaded");

    // hyper parameters
    let iters_num = 10000;
    let train_size = x_train.shape()[0];
    let batch_size = 100;
    let learning_rate = 0.1;

    assert!(train_size > batch_size);
    let iter_per_epoch = train_size / batch_size;

    let mut rng = thread_rng();
    rng.gen_range(0, 1);

    let mut network = TwoLayerNet::new(784, 50, 10, 0.01);

    for i in 0..iters_num {
        let choice_slice = mock_choice(&mut rng, batch_size, train_size);
        let x_batch = x_train.slice_axis(Axis(0), choice_slice);
        let t_batch = t_train.slice_axis(Axis(0), choice_slice);
        let grad = network.gradient(&x_batch, &t_batch);
        network.w1 = network.w1 - learning_rate * grad.w1;
        network.b1 = network.b1 - learning_rate * grad.b1;
        network.w2 = network.w2 - learning_rate * grad.w2;
        network.b2 = network.b2 - learning_rate * grad.b2;

        let loss = network.loss(&x_batch, &t_batch);
        if i % iter_per_epoch == 0 {
            let train_acc = network.accuracy(&x_train, &t_train);
            println!("train acc, test acc | {}", train_acc);
        }
    }
}

fn mock_choice<R: Rng>(rng: &mut R, batch_size: usize, total_size: usize) -> Slice {
    let max = total_size / batch_size;
    let step = rng.gen_range(1, max);
    let bulk_size = step * batch_size;
    let start = rng.gen_range(0, total_size - bulk_size + 1);
    let end = start + step * (batch_size - 1);
    Slice::new(start as isize, Some(end as isize), step as isize)
}

fn choice<R: Rng>(rng: &mut R, from_size: usize, choice_size: usize) -> Vec<usize> {
    let mut selected = BTreeSet::new();
    let select_fewer = from_size > choice_size * 2;
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
            w1: Array::random((input_size, hidden_size), Range::new(0., 1.)) * weight_init_std,
            b1: Array::<f64, _>::zeros(hidden_size),
            w2: Array::random((hidden_size, output_size), Range::new(0., 1.)) * weight_init_std,
            b2: Array::<f64, _>::zeros(output_size),
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
        return y;
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
        let dy = (y - t) / batch_num as f64;
        let w2 = z1.t().dot(&dy);
        let b2 = dy.sum_axis(Axis(0));

        let dz1 = dy.dot(&w2.t());
        let da1 = sigmoid_grad(&a1) * dz1;
        let w1 = x.t().dot(&da1);
        let b1 = da1.sum_axis(Axis(0));
        TwoLayerNet { b1, b2, w1, w2 }
    }
}

fn sigmoid_grad<S: ndarray::Data<Elem = f64>>(
    x: &ArrayBase<S, Ix2>,
) -> ArrayBase<ndarray::OwnedRepr<f64>, Ix2> {
    (1.0 - sigmoid(x)) * sigmoid(x)
}

fn cross_entropy_error<S: ndarray::Data<Elem = f64>, T: ndarray::Data<Elem = f64>>(
    y: &ArrayBase<S, Ix2>,
    t: &ArrayBase<T, Ix2>,
) -> f64 {
    let batch_size = y.shape()[0];
    (log(&(y + 1e-7)) * t).scalar_sum() / batch_size as f64
}

fn log<S: ndarray::Data<Elem = f64>>(
    a: &ArrayBase<S, Ix2>,
) -> ArrayBase<ndarray::OwnedRepr<f64>, Ix2> {
    a.mapv(f64::ln)
}

fn softmax<S: ndarray::Data<Elem = f64>>(
    a: &ArrayBase<S, Ix2>,
) -> ArrayBase<ndarray::OwnedRepr<f64>, Ix2> {
    let c = max(a).unwrap();
    let a = a - c;
    let exp_a = a.mapv(f64::exp);
    let sum_exp_a: f64 = exp_a.scalar_sum();
    exp_a / sum_exp_a
}

fn and(x1: f64, x2: f64) -> bool {
    let x = arr1(&[x1, x2]);
    let w = arr1(&[0.5, 0.5]);
    let b = -0.7;
    x.dot(&w) + b > 0.0
}

fn nand(x1: f64, x2: f64) -> bool {
    let x = arr1(&[x1, x2]);
    let w = arr1(&[-0.5, -0.5]);
    let b = 0.7;
    x.dot(&w) + b > 0.0
}

fn or(x1: f64, x2: f64) -> bool {
    let x = arr1(&[x1, x2]);
    let w = arr1(&[0.5, 0.5]);
    let b = -0.2;
    x.dot(&w) + b > 0.0
}

fn xor(x1: f64, x2: f64) -> bool {
    let s1 = nand(x1, x2);
    let s2 = nand(x1, x2);
    and(if s1 { 1.0 } else { 0.0 }, if s2 { 1.0 } else { 0.0 })
}

fn sigmoid<S: ndarray::Data<Elem = f64>>(
    x: &ArrayBase<S, Ix2>,
) -> ArrayBase<ndarray::OwnedRepr<f64>, Ix2> {
    let sigmoid_function = |x: f64| 1.0 / (1.0 + (-x).exp());
    x.mapv(sigmoid_function)
}

fn relu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

fn max<S: ndarray::Data<Elem = f64>>(a: &ArrayBase<S, Ix2>) -> Option<f64> {
    let mut result: Option<f64> = None;
    for &a in a.iter() {
        let p = result.unwrap_or(a);
        result = Some(if p > a { p } else { a });
    }
    result
}
