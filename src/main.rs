use crate::{grid1d::Grid1dErasure, grid2d::Grid2dErasure, traits::Reconstructable};
use indicatif::ParallelProgressIterator;
use rand::{rngs::SmallRng, thread_rng, SeedableRng};
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use std::error::Error;

mod grid1d;
mod grid2d;
mod traits;
mod base_grid;

#[derive(Debug)]
struct ExperimentConfig {
    n: usize,
    dims: usize,
    n_clients: usize,
    percent_censored: f64,
    n_samples: usize,
}

impl ExperimentConfig {
    fn run(&self) -> f32 {
        if self.dims == 1 {
            self.run_generic::<Grid1dErasure>()
        } else if self.dims == 2 {
            self.run_generic::<Grid2dErasure>()
        } else {
            panic!("Not allowed");
        }
    }

    fn run_generic<R: Reconstructable>(&self) -> f32 {
        let (mask, censor_target) = R::new_mask(&mut thread_rng(), self.n);

        let mut recon_count = 0;
        const N_EXPERIMENTS: usize = 500;
        let n_censored = (self.n_clients as f64 * self.percent_censored).floor() as usize;
        for _ in 0..N_EXPERIMENTS {
            let mut rng = SmallRng::from_entropy();
            let censor_iter = (0..n_censored).into_iter().map(|_| {
                let mut grid = R::new(self.n);
                grid.sample_exclusion(&mut rng, self.n_samples, &mask);
                grid
            });
            let mut rng = SmallRng::from_entropy();
            let honest_iter = (0..self.n_clients - n_censored).into_iter().map(|_| {
                let mut grid = R::new(self.n);
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
    let mut exps: Vec<ExperimentConfig> = Vec::new();
    println!("Running Experiments");
    for n_samples in [10, 15, 20, 25, 30, 35, 40] {
        for n_clients in (50..=1000).step_by(50) {
            for percent_censored in [0.00, 0.2, 0.4, 0.6, 0.8, 0.9] {
                for n in [16, 32, 64, 128] {
                    for dims in [1, 2] {
                        let e = ExperimentConfig {
                            n,
                            dims,
                            n_clients,
                            percent_censored,
                            n_samples,
                        };
                        exps.push(e);
                    }
                }
            }
        }
    }

    let results = exps
        .par_iter()
        .progress_count(exps.len() as u64)
        .map(|e| (e, e.run()))
        .collect::<Vec<_>>();

    println!("Writing");
    let mut writer = csv::Writer::from_path("samples.csv")?;
    writer.write_record(&[
        "dims",
        "matrix_size",
        "n_clients",
        "percent_censored",
        "n_samples",
        "prob",
    ])?;
    for (e, prob) in results {
        writer.write_record(&[
            e.dims.to_string(),
            e.n.to_string(),
            e.n_clients.to_string(),
            e.percent_censored.to_string(),
            e.n_samples.to_string(),
            format!("{:.10}", prob),
        ])?;
    }
    writer.flush()?;
    Ok(())
}
