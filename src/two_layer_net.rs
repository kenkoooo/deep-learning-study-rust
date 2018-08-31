use functions::cross_entropy_error;

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

    fn gradient<S: Data<Elem = f64>>(
        &mut self,
        x: &ArrayBase<S, Ix2>,
        t: &ArrayBase<S, Ix2>,
    ) -> Gradient;
}
