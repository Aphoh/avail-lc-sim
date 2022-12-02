use crate::{grid1d::Grid1dErasure, grid2d::Grid2dErasure, traits::Reconstructable};
use indicatif::ParallelProgressIterator;
use rand::{rngs::SmallRng, thread_rng, SeedableRng};
use rayon::prelude::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::error::Error;
use traits::GridParams;

mod grid1d;
mod grid2d;
mod traits;

struct GridParams1d256 {}
impl GridParams for GridParams1d256 {
    const WIDTH: usize = 256;
    const HEIGHT: usize = 512;
}

struct GridParams2d16 {}
impl GridParams for GridParams2d16 {
    const WIDTH: usize = 32;
    const HEIGHT: usize = 32;
}
struct GridParams2d32 {}
impl GridParams for GridParams2d32 {
    const WIDTH: usize = 64;
    const HEIGHT: usize = 64;
}

struct GridParams2d256 {}
impl GridParams for GridParams2d256 {
    const WIDTH: usize = 512;
    const HEIGHT: usize = 512;
}

#[derive(Debug)]
struct ExperimentConfig {
    n_clients: usize,
    n_censored: usize,
    n_samples: usize,
}

impl ExperimentConfig {
    fn run<R: Reconstructable>(&self) -> f32 {
        let (mask, censor_target) = R::new_mask(&mut thread_rng());

        let mut recon_count = 0;
        const N_EXPERIMENTS: usize = 500;
        for _ in 0..N_EXPERIMENTS {
            let mut rng = SmallRng::from_entropy();
            let censor_iter = (0..self.n_censored).into_iter().map(|_| {
                let mut grid = R::new();
                grid.sample_exclusion(&mut rng, self.n_samples, &mask);
                grid
            });
            let mut rng = SmallRng::from_entropy();
            let honest_iter = (0..self.n_clients - self.n_censored).into_iter().map(|_| {
                let mut grid = R::new();
                grid.sample(&mut rng, self.n_samples);
                grid
            });

            let res = censor_iter.chain(honest_iter).reduce(R::merge).unwrap();

            let recon = res.can_reconstruct(censor_target.clone());
            recon_count += recon as i32;
        }
        (recon_count as f32) / (N_EXPERIMENTS as f32)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut exps = Vec::new();
    println!("Running Experiments");
    for n_samples in [10, 15, 20, 25, 30, 35, 40] {
        for n_clients in (500..10000).step_by(500) {
            for n_censored in (0..(n_clients - 1)).step_by(500) {
                let e = ExperimentConfig {
                    n_clients,
                    n_censored,
                    n_samples,
                };
                exps.push(e);
            }
        }
    }

    let results = exps
        .par_iter()
        .progress_count(exps.len() as u64)
        .flat_map(|e| {
            [
                (1u8, 256, e, e.run::<Grid1dErasure<GridParams1d256>>()),
                (2u8, 16, e, e.run::<Grid2dErasure<GridParams2d16>>()),
                (2u8, 32, e, e.run::<Grid2dErasure<GridParams2d32>>()),
                (2u8, 256, e, e.run::<Grid2dErasure<GridParams2d256>>()),
            ]
        })
        .collect::<Vec<_>>();

    println!("Writing");
    let mut writer = csv::Writer::from_path("samples.csv")?;
    writer.write_record(&[
        "dims",
        "matrix_size",
        "n_clients",
        "n_censored",
        "n_samples",
        "prob",
    ])?;
    for (dims, matrix_size, exp, prob) in results {
        writer.write_record(&[
            dims.to_string(),
            matrix_size.to_string(),
            exp.n_clients.to_string(),
            exp.n_censored.to_string(),
            exp.n_samples.to_string(),
            format!("{:.10}", prob),
        ])?;
    }
    writer.flush()?;
    Ok(())
}
