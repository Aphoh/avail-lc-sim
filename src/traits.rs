use rand::RngCore;

pub trait Reconstructable: Send + Sync {
    type Index: Clone + Send + Sync;
    type Mask: Send + Sync;
    // The dimension of the reconstruction
    fn dims() -> usize;
    // The size of the underlying grid
    fn rand_index<R: RngCore>(rng: &mut R, n: usize) -> Self::Index;
    // Returns an index to censor and a mask representing the points which can be passed
    // to `sample_exclusion` that tells it what coordinates to exclude due to censoring
    fn new_mask<R: RngCore>(rng: &mut R, n: usize) -> (Self::Mask, Self::Index);

    fn new(n: usize) -> Self;
    fn grid_size(&self) -> usize;
    fn can_reconstruct(&self, i: Self::Index) -> bool;
    fn sample<R: RngCore>(&mut self, rng: &mut R, amount: usize);
    fn sample_exclusion<R: RngCore>(&mut self, rng: &mut R, amount: usize, mask: &Self::Mask);
    fn merge(self, other: Self) -> Self;
}

#[inline(always)]
pub fn coord_to_ind(i: usize, j: usize, h: usize) -> usize {
    i + (j * h)
}
#[inline(always)]
pub fn ind_to_col(z: usize, h: usize) -> usize {
    z / h
}

#[inline(always)]
fn ind_to_row(z: usize, h: usize) -> usize {
    z % h
}
