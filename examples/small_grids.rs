use avail_lc_sim::{ExperimentConfig, SampleStrategy};
use indicatif::ParallelProgressIterator;
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use std::error::Error;

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
                            sample_strategy: SampleStrategy::RandomPoints,
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
    let mut writer = csv::Writer::from_path("small_grids.csv")?;
    writer.write_record(ExperimentConfig::header())?;
    for (e, prob) in results {
        writer.write_record(e.to_row(prob))?;
    }
    writer.flush()?;
    Ok(())
}
