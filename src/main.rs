extern crate ndarray;
use ndarray::prelude::*;

fn main() {
    let x = arr1(&[1.0, 0.5]);
    let w1 = arr2(&[[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]);
    let b1 = arr1(&[0.1, 0.2, 0.3]);
    let a1 = x.dot(&w1) + b1;
    let z1 = sigmoid(&a1);

    println!("{:?}", a1);
    println!("{:?}", z1);

    let w2 = arr2(&[[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]);
    let b2 = arr1(&[0.1, 0.2]);

    let a2 = z1.dot(&w2) + b2;
    let z2 = sigmoid(&a2);

    let w3 = arr2(&[[0.1, 0.3], [0.2, 0.4]]);
    let b3 = arr1(&[0.1, 0.2]);

    let a3 = z2.dot(&w3) + b3;

    println!("{:?}", a3);

    let a: Array1<f64> = arr1(&[0.3, 2.9, 4.0]);
    println!("{:?}", softmax(&a));
}

fn softmax(a: &Array1<f64>) -> Array1<f64> {
    let c = max(a).unwrap();
    let a = a - c;
    let exp_a: Array1<f64> = a.iter().map(|e| e.exp()).collect();
    let sum_exp_a: f64 = exp_a.iter().sum();
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

fn sigmoid(x: &Array1<f64>) -> Array1<f64> {
    let sigmoid_function = |x: f64| 1.0 / (1.0 + (-x).exp());
    x.iter().map(|&t| sigmoid_function(t)).collect()
}

fn relu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

fn max(a: &Array1<f64>) -> Option<f64> {
    if a.len() == 0 {
        None
    } else {
        let mut result = a[0];
        for &a in a.iter() {
            if a > result {
                result = a;
            }
        }
        Some(result)
    }
}
