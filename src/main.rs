use rand::{rngs::SmallRng, thread_rng, SeedableRng};
use traits::GridParams;

use crate::{grid1d::Grid1dErasure, traits::Reconstructable};

mod grid1d;
mod traits;

struct GridParams256 {}
impl GridParams for GridParams256 {
    const WIDTH: usize = 256;
    const HEIGHT: usize = 256;
}

fn main() {
    use rayon::prelude::*;
    const NUM_EXPERIMENTS: usize = 2_000;
    const N_CLIENTS: usize = 2_000;
    const N_CLIENTS_CENSORED: usize = (2 * N_CLIENTS) / 3;
    const SAMPLES_PER_CLIENT: usize = 30;

    type Grid = Grid1dErasure<GridParams256>;

    let (mask, censor_target) = Grid::new_mask(&mut thread_rng());

    let mut recon_count = 0;
    for _ in 0..NUM_EXPERIMENTS {
        let censor_iter = (0..N_CLIENTS_CENSORED).into_par_iter().map_init(
            || SmallRng::from_entropy(),
            |rng, _| {
                let mut grid = Grid::new();
                grid.sample_exclusion(rng, SAMPLES_PER_CLIENT, &mask);
                grid
            },
        );
        let honest_iter = (0..N_CLIENTS - N_CLIENTS_CENSORED)
            .into_par_iter()
            .map_init(
                || SmallRng::from_entropy(),
                |rng, _| {
                    let mut grid = Grid::new();
                    grid.sample(rng, SAMPLES_PER_CLIENT);
                    grid
                },
            );

        let res = censor_iter
            .chain(honest_iter)
            .reduce(Grid::new, Grid::merge);

        let recon = res.can_reconstruct(censor_target);
        println!("Reconstruction success: {}", recon);
        recon_count += recon as i32;
    }

    println!(
        "Reconstruction rate: {}",
        (recon_count as f32) / (NUM_EXPERIMENTS as f32)
    )
}
