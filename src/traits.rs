use rand::RngCore;

pub trait Reconstructable: Send + Sync {
    type Index: Clone + Send + Sync;
    type Mask: Send + Sync;
    fn new() -> Self;
    fn rand_index<R: RngCore>(rng: &mut R) -> Self::Index;
    // Returns an index to censor and a mask representing the points which can be passed
    // to `sample_exclusion` that tells it what coordinates to exclude due to censoring
    fn new_mask<R: RngCore>(rng: &mut R) -> (Self::Mask, Self::Index);
    fn can_reconstruct(&self, i: Self::Index) -> bool;
    fn sample<R: RngCore>(&mut self, rng: &mut R, amount: usize);
    fn sample_exclusion<R: RngCore>(&mut self, rng: &mut R, amount: usize, mask: &Self::Mask);
    fn merge(self, other: Self) -> Self;
}

pub trait GridParams: Send + Sync {
    const WIDTH: usize;
    const HEIGHT: usize;
    const WH: usize = Self::WIDTH * Self::HEIGHT;
    #[inline(always)]
    fn coord_to_ind(i: usize, j: usize) -> usize {
        i + j * Self::HEIGHT
    }

    #[inline(always)]
    fn ind_to_col(z: usize) -> usize {
        z / Self::HEIGHT
    }

    #[inline(always)]
    fn ind_to_row(z: usize) -> usize {
        z % Self::HEIGHT
    }
}
