use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::Array1;

use gstools_core::mps::{
    dist_block_categorical, dist_block_categorical_masked, dist_block_categorical_masked_rayon,
    dist_block_categorical_rayon,
};

fn categorical_candidate_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("mps categorical candidate block");
    let rayon_threads = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .min(4);
    let rayon_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(rayon_threads)
        .build()
        .expect("MPS benchmark Rayon pool");
    let one_thread_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("MPS benchmark one-thread pool");

    for n_lags in [8_usize, 16] {
        let de_sim = Array1::from_iter((0..n_lags).map(|index| (index % 3) as f64));
        let lag_flat = Array1::from_iter((0..n_lags).map(|index| index as i64));
        let weights = Array1::from_elem(n_lags, 1.0 / n_lags as f64);

        for candidate_count in [64_usize, 1024, 4096] {
            let ti_flat = Array1::from_iter(
                (0..candidate_count * n_lags).map(|index| ((index / 5) % 3) as f64),
            );
            let mut ti_masked = ti_flat.clone();
            for index in (0..ti_masked.len()).step_by(10) {
                ti_masked[index] = f64::NAN;
            }
            let base_flat =
                Array1::from_iter((0..candidate_count).map(|index| (index * n_lags) as i64));
            group.throughput(Throughput::Elements(candidate_count as u64));

            let parameter = format!("B={candidate_count},n={n_lags}");
            group.bench_with_input(
                BenchmarkId::new("serial", &parameter),
                &candidate_count,
                |b, _| {
                    b.iter(|| {
                        dist_block_categorical(
                            black_box(de_sim.view()),
                            black_box(ti_flat.view()),
                            black_box(base_flat.view()),
                            black_box(lag_flat.view()),
                            black_box(weights.view()),
                        )
                    })
                },
            );
            group.bench_with_input(
                BenchmarkId::new("rayon-1", &parameter),
                &candidate_count,
                |b, _| {
                    b.iter(|| {
                        one_thread_pool.install(|| {
                            dist_block_categorical_rayon(
                                black_box(de_sim.view()),
                                black_box(ti_flat.view()),
                                black_box(base_flat.view()),
                                black_box(lag_flat.view()),
                                black_box(weights.view()),
                            )
                        })
                    })
                },
            );
            group.bench_with_input(
                BenchmarkId::new(format!("rayon-{rayon_threads}"), &parameter),
                &candidate_count,
                |b, _| {
                    b.iter(|| {
                        rayon_pool.install(|| {
                            dist_block_categorical_rayon(
                                black_box(de_sim.view()),
                                black_box(ti_flat.view()),
                                black_box(base_flat.view()),
                                black_box(lag_flat.view()),
                                black_box(weights.view()),
                            )
                        })
                    })
                },
            );
            group.bench_with_input(
                BenchmarkId::new("masked-serial", &parameter),
                &candidate_count,
                |b, _| {
                    b.iter(|| {
                        dist_block_categorical_masked(
                            black_box(de_sim.view()),
                            black_box(ti_masked.view()),
                            black_box(base_flat.view()),
                            black_box(lag_flat.view()),
                            black_box(weights.view()),
                        )
                    })
                },
            );
            group.bench_with_input(
                BenchmarkId::new(format!("masked-rayon-{rayon_threads}"), &parameter),
                &candidate_count,
                |b, _| {
                    b.iter(|| {
                        rayon_pool.install(|| {
                            dist_block_categorical_masked_rayon(
                                black_box(de_sim.view()),
                                black_box(ti_masked.view()),
                                black_box(base_flat.view()),
                                black_box(lag_flat.view()),
                                black_box(weights.view()),
                            )
                        })
                    })
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, categorical_candidate_benchmark);
criterion_main!(benches);
