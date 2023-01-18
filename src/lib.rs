use grid1d::Grid1dErasure;
use grid2d::Grid2dErasure;
use rand::{rngs::SmallRng, thread_rng, SeedableRng};
use traits::Reconstructable;

pub use base_grid::SampleStrategy;

mod base_grid;
mod grid1d;
mod grid2d;
mod traits;

#[derive(Debug)]
pub struct ExperimentConfig {
    pub n: usize,
    pub dims: usize,
    pub n_clients: usize,
    pub percent_censored: f64,
    pub n_samples: usize,
    pub sample_strategy: SampleStrategy,
}

impl ExperimentConfig {
    pub fn run(&self) -> f32 {
        if self.dims == 1 {
            self.run_generic::<Grid1dErasure>()
        } else if self.dims == 2 {
            self.run_generic::<Grid2dErasure>()
        } else {
            unimplemented!()
        }
    }

    pub fn run_generic<R: Reconstructable>(&self) -> f32 {
        let (mask, censor_target) = R::new_mask(&mut thread_rng(), self.n);

        let mut recon_count = 0;
        const N_EXPERIMENTS: usize = 500;
        let n_censored = (self.n_clients as f64 * self.percent_censored).floor() as usize;
        for _ in 0..N_EXPERIMENTS {
            let mut rng = SmallRng::from_entropy();
            // Grid that mimmics n_censored clients each making n_samples with censorship
            let mut censor_grid = R::new(self.n);
            censor_grid.sample_exclusion(
                &mut rng,
                self.n_samples * n_censored, // n_censored nodes making n_samples requests
                &self.sample_strategy,
                &mask,
            );
            // Grid that mimmics n_clients - n_censored clients making n_samples with censorship
            let mut honest_grid = R::new(self.n);
            honest_grid.sample(
                &mut rng,
                self.n_samples * (self.n_clients - n_censored),
                &self.sample_strategy,
            );
            let res = censor_grid.merge(honest_grid);

            let recon = res.can_reconstruct(censor_target.clone());
            recon_count += recon as i32;
        }
        (recon_count as f32) / (N_EXPERIMENTS as f32)
    }

    pub fn header() -> &'static [&'static str] {
        &[
            "dims",
            "n",
            "n_clients",
            "percent_censored",
            "n_samples",
            "strategy",
            "box_width",
            "box_height",
            "prob",
        ]
    }

    pub fn to_row(&self, prob: f32) -> Vec<String> {
        let (box_width, box_height) = match self.sample_strategy {
            SampleStrategy::Box { width, height } => (width, height),
            SampleStrategy::RandomPoints => (1, 1),
        };
        vec![
            self.dims.to_string(),
            self.n.to_string(),
            self.n_clients.to_string(),
            self.percent_censored.to_string(),
            self.n_samples.to_string(),
            self.sample_strategy.to_string(),
            box_width.to_string(),
            box_height.to_string(),
            format!("{:.10}", prob),
        ]
    }
}
