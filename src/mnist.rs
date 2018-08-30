use std::fs;
use std::io::{BufReader, Bytes, Read};
use std::u8;

use ndarray::{arr1, Array, Array2};

pub fn load_labels(filename: &str) -> Result<Vec<u8>, ()> {
    get_reader(filename)
        .and_then(
            |mut reader| match load_number_from_byte_reader(&mut reader, 4) {
                Ok(2049) => Ok(reader),
                _ => Err(()),
            },
        ).and_then(
        |mut reader| match load_number_from_byte_reader(&mut reader, 4) {
            Ok(num) => load_bytes_from_byte_reader(&mut reader, num),
            _ => Err(()),
        },
    )
}

pub fn load_images_array(filename: &str) -> Result<Array2<f64>, ()> {
    get_reader(filename)
        .and_then(
            |mut reader| match load_number_from_byte_reader(&mut reader, 4) {
                Ok(2051) => Ok(reader),
                _ => Err(()),
            },
        ).and_then(|mut reader| {
        let num = load_number_from_byte_reader(&mut reader, 4);
        let height = load_number_from_byte_reader(&mut reader, 4);
        let width = load_number_from_byte_reader(&mut reader, 4);
        match (num, height, width) {
            (Ok(num), Ok(height), Ok(width)) => Ok((num, height, width, reader)),
            _ => Err(()),
        }
    }).and_then(|(num, height, width, mut reader)| {
        let iter = (0..num).map(|_| load_bytes_from_byte_reader(&mut reader, height * width));
        let mut result = Array::zeros((num, height * width));
        for (i, v) in iter.enumerate() {
            if v.is_ok() {
                let v = v.unwrap();
                result.row_mut(i).assign(&arr1(&v).mapv(|f| f as f64));
            } else {
                return Err(());
            }
        }
        return Ok(result);
    })
}

pub fn load_images(filename: &str) -> Result<Vec<Vec<u8>>, ()> {
    get_reader(filename)
        .and_then(
            |mut reader| match load_number_from_byte_reader(&mut reader, 4) {
                Ok(2051) => Ok(reader),
                _ => Err(()),
            },
        ).and_then(|mut reader| {
        let num = load_number_from_byte_reader(&mut reader, 4);
        let height = load_number_from_byte_reader(&mut reader, 4);
        let width = load_number_from_byte_reader(&mut reader, 4);
        match (num, height, width) {
            (Ok(num), Ok(height), Ok(width)) => Ok((num, height, width, reader)),
            _ => Err(()),
        }
    }).and_then(|(num, height, width, mut reader)| {
        (0..num)
            .map(|_| load_bytes_from_byte_reader(&mut reader, height * width))
            .collect::<Result<Vec<Vec<u8>>, ()>>()
    })
}

fn get_reader(filename: &str) -> Result<Bytes<BufReader<fs::File>>, ()> {
    match fs::File::open(filename) {
        Ok(f) => Ok(BufReader::new(f).bytes()),
        _ => Err(()),
    }
}

fn load_bytes_from_byte_reader(
    reader: &mut Bytes<BufReader<fs::File>>,
    length: usize,
) -> Result<Vec<u8>, ()> {
    let mut result = vec![0; length];
    for i in 0..length {
        let loaded = match reader.next() {
            Some(Ok(b)) => {
                result[i] = b;
                true
            }
            _ => false,
        };
        if !loaded {
            return Err(());
        }
    }
    Ok(result)
}

fn load_number_from_byte_reader(
    reader: &mut Bytes<BufReader<fs::File>>,
    length: usize,
) -> Result<usize, ()> {
    match load_bytes_from_byte_reader(reader, length) {
        Ok(bytes) => {
            let mut result = 0;
            for &b in bytes.iter() {
                result *= u8::MAX as usize + 1;
                result += b as usize;
            }
            Ok(result)
        }
        _ => Err(()),
    }
}
