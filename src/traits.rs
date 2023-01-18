use rand::RngCore;

use crate::base_grid::{Grid, SampleStrategy};

pub trait Reconstructable: Send + Sync {
    type Index: Clone + Send + Sync;
    // The dimension of the reconstruction
    fn dims() -> usize;
    // Returns an index to censor and a mask representing the points which can be passed
    // to `sample_exclusion` that tells it what coordinates to exclude due to censoring
    fn new_mask<R: RngCore>(rng: &mut R, n: usize) -> (Grid, Self::Index);

    fn new(n: usize) -> Self;
    fn grid_size(&self) -> usize;
    fn can_reconstruct(&self, i: Self::Index) -> bool;
    fn sample<R: RngCore>(&mut self, rng: &mut R, amount: usize, strategy: &SampleStrategy);
    fn sample_exclusion<R: RngCore>(&mut self, rng: &mut R, amount: usize, strategy: &SampleStrategy, mask: &Grid);
    fn merge(self, other: Self) -> Self;
}
