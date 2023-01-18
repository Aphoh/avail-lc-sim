use avail_lc_sim::{ExperimentConfig, SampleStrategy};
use indicatif::ParallelProgressIterator;
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let mut exps: Vec<ExperimentConfig> = Vec::new();
    println!("Running Experiments");
    // The approximate number of eleemnts we'll try to request from the grid
    let target_n_samples = 1024;
    let n = 256; // 256 grid
    for n_clients in (10..=300).step_by(20) {
        for percent_censored in [0.00, 0.2, 0.4, 0.6, 0.8, 0.9] {
            for width in [1, 2, 4, 8, 16, 32, 64, 128] {
                for height in [1, 2, 4, 8, 16, 32, 64, 128] {
                    let wh = width * height;
                    // if our block is too big, or our choice doesn't evenly divide
                    // target_n_samples, skip it
                    if width * height > target_n_samples || target_n_samples % wh != 0 {
                        continue;
                    }
                    let n_samples = target_n_samples / wh;
                    for dims in [1, 2] {
                        let e = ExperimentConfig {
                            n,
                            dims,
                            n_clients,
                            percent_censored,
                            n_samples,
                            sample_strategy: SampleStrategy::Box { width, height },
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
    let mut writer = csv::Writer::from_path("block_sampling.csv")?;
    writer.write_record(ExperimentConfig::header())?;
    for (e, prob) in results {
        writer.write_record(e.to_row(prob))?;
    }
    writer.flush()?;
    Ok(())
}
