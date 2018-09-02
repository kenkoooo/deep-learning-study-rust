use functions::{accuracy, cross_entropy_error};

use ndarray::{Array1, Array2, ArrayBase, ArrayView1, Data, Ix1, Ix2, OwnedRepr};

pub struct Gradient {
    pub w1: Array2<f64>,
    pub b1: Array1<f64>,
    pub w2: Array2<f64>,
    pub b2: Array1<f64>,
}

pub trait TwoLayerNetInterface {
    fn predict<S: Data<Elem = f64>>(
        &mut self,
        x: &ArrayBase<S, Ix2>,
    ) -> ArrayBase<OwnedRepr<f64>, Ix2>;

    fn loss<S: Data<Elem = f64>>(&mut self, x: &ArrayBase<S, Ix2>, t: &ArrayBase<S, Ix2>) -> f64;

    fn accuracy(&mut self, x: &Array2<f64>, t: &Array2<f64>) -> f64 {
        let y = self.predict(x);
        accuracy(&y, t)
    }

    fn gradient<S: Data<Elem = f64>>(
        &mut self,
        x: &ArrayBase<S, Ix2>,
        t: &ArrayBase<S, Ix2>,
    ) -> Gradient;
}
