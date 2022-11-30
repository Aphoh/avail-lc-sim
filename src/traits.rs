use rand::RngCore;

pub trait Reconstructable {
    type Index;
    type Mask;
    fn new() -> Self;
    fn rand_index<R: RngCore>(rng: &mut R) -> Self::Index;
    // Returns an index to censor and a mask representing the points which can be passed
    // to `sample_exclusion` that tells it what coordinates to exclude due to censoring
    fn new_mask<R: RngCore>(rng: &mut R) -> (Self::Mask, Self::Index);
    fn has_index(&self, i: Self::Index) -> bool;
    fn can_reconstruct(&self, i: Self::Index) -> bool;
    fn sample<R: RngCore>(&mut self, rng: &mut R, amount: usize);
    fn sample_exclusion<R: RngCore>(&mut self, rng: &mut R, amount: usize, mask: &Self::Mask);
    fn merge(self, other: Self) -> Self;
}

pub trait GridParams {
    const WIDTH: usize;
    const HEIGHT: usize;
    const WH: usize = Self::WIDTH * Self::HEIGHT;
}
