use crate::{grid1d::Grid1dErasure, traits::Reconstructable};
use indicatif::ParallelProgressIterator;
use rand::{rngs::SmallRng, thread_rng, SeedableRng};
use rayon::prelude::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::error::Error;
use traits::GridParams;

mod grid1d;
mod traits;

struct GridParams256 {}
impl GridParams for GridParams256 {
    const WIDTH: usize = 256;
    const HEIGHT: usize = 512;
}

#[derive(Debug)]
struct ExperimentConfig {
    n_clients: usize,
    n_censored: usize,
    n_samples: usize,
}

impl ExperimentConfig {
    fn run<P: GridParams + Send>(&self) -> f32 {
        let (mask, censor_target) = <Grid1dErasure<P>>::new_mask(&mut thread_rng());

        let mut recon_count = 0;
        // Run 1000 experiments
        const N_EXPERIMENTS: usize = 100;
        for _ in 0..N_EXPERIMENTS {
            let censor_iter = (0..self.n_censored).into_par_iter().map_init(
                || SmallRng::from_entropy(),
                |rng, _| {
                    let mut grid = <Grid1dErasure<P>>::new();
                    grid.sample_exclusion(rng, self.n_samples, &mask);
                    grid
                },
            );
            let honest_iter = (0..self.n_clients - self.n_censored)
                .into_par_iter()
                .map_init(
                    || SmallRng::from_entropy(),
                    |rng, _| {
                        let mut grid = <Grid1dErasure<P>>::new();
                        grid.sample(rng, self.n_samples);
                        grid
                    },
                );

            let res = censor_iter
                .chain(honest_iter)
                .reduce(<Grid1dErasure<P>>::new, <Grid1dErasure<P>>::merge);

            let recon = res.can_reconstruct(censor_target);
            recon_count += recon as i32;
        }
        (recon_count as f32) / (N_EXPERIMENTS as f32)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut exps = Vec::new();
    println!("Running Experiments");
    for n_samples in [20, 30, 40, 50, 60] {
        for n_clients in (1000..20000).step_by(1000) {
            for n_censored in (0..(n_clients - 1)).step_by(1000) {
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
        .map(|e| (e, e.run::<GridParams256>()))
        .collect::<Vec<_>>();

    println!("Writing");
    let mut writer = csv::Writer::from_path("samples.csv")?;
    writer.write_record(&["n_clients", "n_censored", "n_samples", "prob"])?;
    for (exp, prob) in results {
        writer.write_record(&[
            exp.n_clients.to_string(),
            exp.n_censored.to_string(),
            exp.n_samples.to_string(),
            prob.to_string(),
        ])?;
    }
    Ok(())
}
