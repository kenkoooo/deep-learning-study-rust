use std::cmp;
use std::collections::BTreeSet;

use ndarray::{Array, ArrayBase, Data, Dimension, Ix2, OwnedRepr};
use rand::Rng;

pub trait ArrayChoiceCopy<S> {
    fn masked_copy(&self, mask: &Vec<usize>) -> ArrayBase<OwnedRepr<f64>, Ix2>;
}

impl<S> ArrayChoiceCopy<S> for ArrayBase<S, Ix2>
where
    S: Data<Elem = f64>,
{
    fn masked_copy(&self, mask: &Vec<usize>) -> ArrayBase<OwnedRepr<f64>, Ix2> {
        let m = mask.len();
        let cols = self.shape()[1];
        let mut result = Array::zeros((m, cols));
        for row in 0..m {
            result.row_mut(row).assign(&self.row(mask[row]));
        }
        result
    }
}

pub fn choice<R: Rng>(rng: &mut R, from_size: usize, choice_size: usize) -> Vec<usize> {
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

pub trait ArrayFunctions<D> {
    fn sqrt(&self) -> Array<f64, D>;
    fn powi(&self, i: i32) -> Array<f64, D>;
}

impl<D> ArrayFunctions<D> for Array<f64, D>
where
    D: Dimension,
{
    fn sqrt(&self) -> Array<f64, D> {
        self.mapv(f64::sqrt)
    }

    fn powi(&self, i: i32) -> Array<f64, D> {
        self.mapv(|t| t.powi(i))
    }
}
