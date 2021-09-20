use criterion::{black_box, criterion_group, criterion_main, Criterion};
use gstools_core::field::summator;
use ndarray::{arr2, Array1, Array2};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

fn read_1d_from_file(file_path: &Path) -> Array1<f64> {
    let file = File::open(file_path).expect("File wasn't found");
    let reader = BufReader::new(file);

    reader
        .lines()
        .map(|line| line.unwrap().parse::<f64>().unwrap())
        .collect()
}

fn read_2d_from_file(file_path: &Path) {
    //-> Array2::<f64> {
    let file = File::open(file_path).expect("File wasn't found");
    let reader = BufReader::new(file);

    let mut data = Vec::new();

    let r: Vec<Vec<f64>> = reader
        .lines()
        .map(|line| {
            line.unwrap()
                .split(char::is_whitespace)
                .map(|number| number.parse::<f64>().unwrap())
                .collect()
        })
        .collect();
    let r = arr2(&r);

    println!("r = {:?}", r);
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let path = Path::new("benches/input");

    let x = read_1d_from_file(&path.join("field_x.txt"));
    let y = read_1d_from_file(&path.join("field_y.txt"));
    let z = read_1d_from_file(&path.join("field_z.txt"));

    let cov_samples = read_2d_from_file(&path.join("field_cov_samples.txt"));

    let z_1 = read_1d_from_file(&path.join("field_z_1.txt"));
    let z_2 = read_1d_from_file(&path.join("field_z_2.txt"));

    //c.bench_function("1st test", |b| b.iter(|| black_box(20)));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
