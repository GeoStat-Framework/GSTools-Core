use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::{arr2, stack, Array1, Array2, Axis};
use ndarray_rand::{
    rand::{rngs::SmallRng, SeedableRng},
    rand_distr::{Bernoulli, Uniform},
    RandomExt,
};

use gstools_core::field::{summator, summator_incompr};
use gstools_core::krige::{calculator_field_krige, calculator_field_krige_and_variance};
use gstools_core::variogram::{
    variogram_directional, variogram_ma_structured, variogram_structured, variogram_unstructured,
};

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

    let x = read_1d_from_file(&path.join("field_x.txt"));
    let y = read_1d_from_file(&path.join("field_y.txt"));
    let z = read_1d_from_file(&path.join("field_z.txt"));

    let pos = stack![Axis(0), x, y, z];

    let cov_samples = read_2d_from_file(&path.join("field_cov_samples.txt"));

    let z_1 = read_1d_from_file(&path.join("field_z_1.txt"));
    let z_2 = read_1d_from_file(&path.join("field_z_2.txt"));

    c.bench_function("field summate", |b| {
        b.iter(|| summator(cov_samples.view(), z_1.view(), z_2.view(), pos.view(), None))
    });

    c.bench_function("field summate incompr", |b| {
        b.iter(|| summator_incompr(cov_samples.view(), z_1.view(), z_2.view(), pos.view(), None))
    });
}

pub fn krige_benchmark(c: &mut Criterion) {
    let path = Path::new("benches/input");

    let krige_mat = read_2d_from_file(&path.join("krige_krige_mat.txt"));
    let k_vec = read_2d_from_file(&path.join("krige_k_vec.txt"));
    let krige_cond = read_1d_from_file(&path.join("krige_krige_cond.txt"));

    c.bench_function("krige error", |b| {
        b.iter(|| {
            calculator_field_krige_and_variance(
                krige_mat.view(),
                k_vec.view(),
                krige_cond.view(),
                None,
            )
        })
    });

    c.bench_function("krige", |b| {
        b.iter(|| calculator_field_krige(krige_mat.view(), k_vec.view(), krige_cond.view(), None))
    });
}

pub fn variogram_benchmark(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);

    let x = 600;
    let y = 500;
    let f = Array2::from_elem((x, y), 1.0);
    let mask = Array2::random_using((x, y), Bernoulli::new(0.5).unwrap(), &mut rng);

    c.bench_function("variogram structured", |b| {
        b.iter(|| {
            variogram_structured(black_box(f.view()), 'm', None);
        })
    });
    c.bench_function("variogram masked structured", |b| {
        b.iter(|| {
            variogram_ma_structured(black_box(f.view()), black_box(mask.view()), 'm', None);
        })
    });

    let pos_no = 2_000;
    let f = Array2::from_elem((1, pos_no), 1.0);
    let bin_edges = Array1::linspace(0., 20., 30);
    let pos = Array2::random_using((2, pos_no), Uniform::new(-10., 10.), &mut rng);

    c.bench_function("variogram unstructured", |b| {
        b.iter(|| {
            variogram_unstructured(
                black_box(f.view()),
                black_box(bin_edges.view()),
                black_box(pos.view()),
                'm',
                'e',
                None,
            );
        })
    });

    let direction = arr2(&[[0., std::f64::consts::PI], [0., 0.]]);

    c.bench_function("variogram directional", |b| {
        b.iter(|| {
            variogram_directional(
                black_box(f.view()),
                black_box(bin_edges.view()),
                black_box(pos.view()),
                black_box(direction.view()),
                std::f64::consts::PI / 8.,
                -1.0,
                false,
                'm',
                None,
            );
        })
    });
}

criterion_group!(
    benches,
    field_benchmark,
    krige_benchmark,
    variogram_benchmark
);
criterion_main!(benches);
