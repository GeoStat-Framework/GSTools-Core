use criterion::{criterion_group, criterion_main, Criterion};
use gstools_core::field::{summator, summator_incompr};
use ndarray::{stack, Array1, Array2, Axis};
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

fn read_2d_from_file(file_path: &Path) -> Array2<f64> {
    let file = File::open(file_path).expect("File wasn't found");
    let reader = BufReader::new(file);

    let vec: Vec<Array1<f64>> = reader
        .lines()
        .map(|l| {
            l.unwrap()
                .split(char::is_whitespace)
                .map(|number| number.parse::<f64>().unwrap())
                .collect()
        })
        .collect();

    let shape = (vec.len(), vec[0].dim());

    let flat_vec: Vec<f64> = vec.iter().flatten().cloned().collect();

    Array2::from_shape_vec(shape, flat_vec).unwrap()
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let path = Path::new("benches/input");

    let x = read_1d_from_file(&path.join("field_bench_x.txt"));
    let y = read_1d_from_file(&path.join("field_bench_y.txt"));
    let z = read_1d_from_file(&path.join("field_bench_z.txt"));

    let pos = stack![Axis(0), x, y, z];

    let cov_samples = read_2d_from_file(&path.join("field_bench_cov_samples.txt"));

    let z_1 = read_1d_from_file(&path.join("field_bench_z_1.txt"));
    let z_2 = read_1d_from_file(&path.join("field_bench_z_2.txt"));

    c.bench_function("field summate", |b| {
        b.iter(|| summator(cov_samples.view(), z_1.view(), z_2.view(), pos.view()))
    });

    c.bench_function("field summate incompr", |b| {
        b.iter(|| summator_incompr(cov_samples.view(), z_1.view(), z_2.view(), pos.view()))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
