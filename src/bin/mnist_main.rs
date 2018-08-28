extern crate deep_learning_study_rust;
use deep_learning_study_rust::mnist;

fn main() {
    match mnist::load_images("dataset/train-images-idx3-ubyte") {
        Ok(images) => {
            println!("{} images are loaded", images.len());
        }
        _ => {
            panic!();
        }
    }
    match mnist::load_labels("dataset/train-labels-idx1-ubyte") {
        Ok(labels) => println!("{}", labels.len()),
        _ => {
            panic!();
        }
    }
}
