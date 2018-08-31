extern crate deep_learning_study_rust;
extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;

use deep_learning_study_rust::mnist;
use ndarray::prelude::*;
use ndarray::{Array, Slice};
use ndarray_rand::RandomExt;
use rand::distributions::Normal;
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

fn train() {
    // chapter 04
    let (x_train, t_train) = get_train_data();
    let t_train = label_to_one_hot(&t_train);
    println!("train data loaded");

    // hyper parameters
    let iters_num = 10000;
    let train_size = x_train.shape()[0];
    let batch_size = 100;
    let learning_rate = 0.1;

    assert!(train_size >= batch_size);
    let iter_per_epoch = train_size / batch_size;

    let mut rng = thread_rng();

    let mut network = TwoLayerNet::new(784, 50, 10, 0.01);

    for i in 0..iters_num {
        let mask = choice(&mut rng, train_size, batch_size);
        assert_eq!(mask.len(), batch_size);
        let x_batch = masked_copy(&x_train, &mask);
        let t_batch = masked_copy(&t_train, &mask);

        let grad = network.gradient(&x_batch, &t_batch);
        network.w1 = network.w1 - grad.w1 * learning_rate;
        network.b1 = network.b1 - grad.b1 * learning_rate;
        network.w2 = network.w2 - grad.w2 * learning_rate;
        network.b2 = network.b2 - grad.b2 * learning_rate;

        let loss = network.loss(&x_batch, &t_batch);
        if i % iter_per_epoch == 0 {
            let train_acc = network.accuracy(&x_train, &t_train);
            println!("{}", loss);
            println!("train acc, test acc | {}", train_acc);
        }
    }
}

fn main() {
    train();
}

fn masked_copy<S: ndarray::Data<Elem = f64>>(
    x: &ArrayBase<S, Ix2>,
    mask: &Vec<usize>,
) -> ArrayBase<ndarray::OwnedRepr<f64>, Ix2> {
    let m = mask.len();
    let cols = x.shape()[1];
    let mut result = Array::zeros((m, cols));
    for row in 0..m {
        result.row_mut(row).assign(&x.row(mask[row]));
    }
    result
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
            w1: Array::random((input_size, hidden_size), Normal::new(0.0, 1.0)) * weight_init_std,
            b1: Array::<f64, _>::zeros(hidden_size),
            w2: Array::random((hidden_size, output_size), Normal::new(0.0, 1.0)) * weight_init_std,
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

fn sigmoid_grad<S, D>(x: &ArrayBase<S, D>) -> ArrayBase<ndarray::OwnedRepr<f64>, D>
where
    S: ndarray::Data<Elem = f64>,
    D: ndarray::Dimension,
{
    (1.0 - sigmoid(x)) * sigmoid(x)
}

fn cross_entropy_error<S: ndarray::Data<Elem = f64>, T: ndarray::Data<Elem = f64>>(
    y: &ArrayBase<S, Ix2>,
    t: &ArrayBase<T, Ix2>,
) -> f64 {
    let batch_size = y.shape()[0];
    -(log(&(y + 1e-7)) * t).scalar_sum() / batch_size as f64
}

fn log<S: ndarray::Data<Elem = f64>>(
    a: &ArrayBase<S, Ix2>,
) -> ArrayBase<ndarray::OwnedRepr<f64>, Ix2> {
    a.mapv(f64::ln)
}

fn softmax<S>(x: &ArrayBase<S, Ix2>) -> ArrayBase<ndarray::OwnedRepr<f64>, Ix2>
where
    S: ndarray::Data<Elem = f64> + ndarray::DataMut,
{
    let n = x.rows();
    let d = x.raw_dim();
    let mut y = Array::zeros(d);
    for i in 0..n {
        let max_scalar = max(&x.row(i)).unwrap();
        let exp = x.row(i).mapv(|f| f - max_scalar).mapv(f64::exp);
        let sum = exp.scalar_sum();
        y.row_mut(i).assign(&(exp / sum));
    }
    y
}

fn sigmoid<S, D>(x: &ArrayBase<S, D>) -> ArrayBase<ndarray::OwnedRepr<f64>, D>
where
    S: ndarray::Data<Elem = f64>,
    D: ndarray::Dimension,
{
    1.0 / ((-x).mapv(f64::exp) + 1.0)
}

fn max<S, D>(a: &ArrayBase<S, D>) -> Option<f64>
where
    S: ndarray::Data<Elem = f64>,
    D: ndarray::Dimension,
{
    let mut result: Option<f64> = None;
    for &a in a.iter() {
        let p = result.unwrap_or(a);
        result = Some(if p > a { p } else { a });
    }
    result
}
