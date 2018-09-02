extern crate deep_learning_study_rust;
extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;

use deep_learning_study_rust::layer;
use deep_learning_study_rust::layer::{AddLayer, MulLayer};
use deep_learning_study_rust::mnist;
use deep_learning_study_rust::two_layer_net::{Gradient, TwoLayerNetInterface};
use deep_learning_study_rust::utils::*;

use ndarray::{Array, Array1, Array2, ArrayBase, Data, Ix2};
use ndarray_rand::RandomExt;
use rand::distributions::Normal;
use rand::thread_rng;

fn main() {
    let apple = 100.0;
    let apple_num = 2.0;
    let tax = 1.1;

    let mut mul_apple_layer = MulLayer::new();
    let mut mul_tax_layer = MulLayer::new();

    // forward
    let apple_price = mul_apple_layer.forward(apple, apple_num);
    let price = mul_tax_layer.forward(apple_price, tax);

    println!("{}", price);

    let d_price = 1.0;
    let (d_apple_price, d_tax) = mul_tax_layer.backward(d_price);
    let (d_apple, d_apple_num) = mul_apple_layer.backward(d_apple_price);

    println!("{} {} {}", d_apple, d_apple_num, d_tax);

    let apple = 100.0;
    let apple_num = 2.0;
    let orange = 150.0;
    let orange_num = 3.0;
    let tax = 1.1;

    let mut mul_apple_layer = MulLayer::new();
    let mut mul_orange_layer = MulLayer::new();
    let add_apple_orange_layer = AddLayer::new();
    let mut mul_tax_layer = MulLayer::new();

    let apple_price = mul_apple_layer.forward(apple, apple_num);
    let orange_price = mul_orange_layer.forward(orange, orange_num);
    let all_price = add_apple_orange_layer.forward(apple_price, orange_price);
    let price = mul_tax_layer.forward(all_price, tax);

    let d_price = 1.0;
    let (d_all_price, d_tax) = mul_tax_layer.backward(d_price);
    let (d_apple_price, d_orange_price) = add_apple_orange_layer.backward(d_all_price);
    let (d_orange, d_orange_num) = mul_orange_layer.backward(d_orange_price);
    let (d_apple, d_apple_num) = mul_apple_layer.backward(d_apple_price);

    println!("{}", price);
    println!(
        "{} {} {} {} {}",
        d_apple_num, d_apple, d_orange, d_orange_num, d_tax
    );

    let (x_train, t_train, x_test, t_test) = mnist::load_data().unwrap();

    let mut network = TwoLayerNet::new(784, 50, 10, 0.01);

    let iter_num = 10000;
    let train_size = x_train.shape()[0];
    let batch_size = 100;
    let learning_rate = 0.1;
    let iter_per_epoch = train_size / batch_size;

    let mut rng = thread_rng();

    for i in 0..iter_num {
        let batch_mask = choice(&mut rng, train_size, batch_size);
        let x_batch = x_train.masked_copy(&batch_mask);
        let t_batch = t_train.masked_copy(&batch_mask);

        let grad = network.gradient(&x_batch, &t_batch);
        network.b1 -= &(learning_rate * grad.b1);
        network.b2 -= &(learning_rate * grad.b2);
        network.w1 -= &(learning_rate * grad.w1);
        network.w2 -= &(learning_rate * grad.w2);

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
}

impl TwoLayerNetInterface for TwoLayerNet {
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
    ) -> Gradient {
        // forward
        self.loss(x, t);

        // backward
        let d_out = self.last_layer.backward();
        let d_out = self.affine2.backward(&self.w2, &d_out);
        let d_out = self.relu1.backward(&d_out);
        self.affine1.backward(&self.w1, &d_out);

        Gradient {
            w1: &self.affine1.dw * 1.0,
            b1: &self.affine1.db * 1.0,
            w2: &self.affine2.dw * 1.0,
            b2: &self.affine2.db * 1.0,
        }
    }
}
