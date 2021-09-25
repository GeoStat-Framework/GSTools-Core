use criterion::{criterion_group, criterion_main, Criterion};
use gstools_core::field::{summator, summator_incompr};
use gstools_core::krige::{calculator_field_krige, calculator_field_krige_and_variance};
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

    let mut vec = Vec::new();
    let mut lines = 0;

    for line in reader.lines() {
        vec.extend(
            line.unwrap()
                .split_whitespace()
                .map(|number| number.parse::<f64>().unwrap()),
        );
        lines += 1;
    }

    let shape = (lines, vec.len() / lines);

    Array2::from_shape_vec(shape, vec).unwrap()
}

pub fn field_benchmark(c: &mut Criterion) {
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

pub fn krige_benchmark(c: &mut Criterion) {
    let path = Path::new("benches/input");

    let krige_mat = read_2d_from_file(&path.join("krige_bench_krige_mat.txt"));
    let k_vec = read_2d_from_file(&path.join("krige_bench_k_vec.txt"));
    let krige_cond = read_1d_from_file(&path.join("krige_bench_krige_cond.txt"));

    c.bench_function("krige error", |b| {
        b.iter(|| {
            calculator_field_krige_and_variance(krige_mat.view(), k_vec.view(), krige_cond.view())
        })
    });

    c.bench_function("krige", |b| {
        b.iter(|| calculator_field_krige(krige_mat.view(), k_vec.view(), krige_cond.view()))
    });
}

criterion_group!(benches, field_benchmark, krige_benchmark);
criterion_main!(benches);
