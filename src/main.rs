use std::{marker::PhantomData, ops::Not};

use bitvec_simd::BitVec;
use rand::{distributions::Uniform, prelude::Distribution, RngCore};

trait Reconstructable {
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

trait GridParams {
    const WIDTH: usize;
    const HEIGHT: usize;
    const WH: usize = Self::WIDTH * Self::HEIGHT;
}

struct Grid1dErasure<P: GridParams> {
    // the grid stored column wise to make adding along columns more efficient
    grid: BitVec,
    // counts of each column
    col_counts: Vec<u16>,
    _phantom: PhantomData<P>,
}

impl<P: GridParams> Grid1dErasure<P> {
    #[inline(always)]
    fn coord_to_ind(i: usize, j: usize) -> usize {
        i * P::HEIGHT + j
    }

    #[inline(always)]
    fn ind_to_col(z: usize) -> usize {
        z % P::HEIGHT
    }

    #[inline(always)]
    fn ind_to_row(z: usize) -> usize {
        z / P::HEIGHT
    }
}

// Height in P must be even
impl<P: GridParams> Reconstructable for Grid1dErasure<P> {
    type Index = usize;
    type Mask = BitVec;

    fn new() -> Self {
        Grid1dErasure {
            grid: BitVec::zeros(P::WH),
            col_counts: vec![0; P::WIDTH],
            _phantom: PhantomData,
        }
    }

    fn rand_index<R: RngCore>(rng: &mut R) -> Self::Index {
        Uniform::new(0, P::WIDTH * P::HEIGHT).sample(rng)
    }

    fn new_mask<R: RngCore>(rng: &mut R) -> (Self::Mask, Self::Index) {
        let col = Uniform::from(0..P::WIDTH).sample(rng);
        let mut mask = BitVec::zeros(P::WH);
        // Set (0, col)..(n, col) true
        for i in 0..(P::HEIGHT / 2) + 1 {
            mask.set(Self::coord_to_ind(i, col), true);
        }
        (mask.not(), Self::coord_to_ind(0, col))
    }

    fn has_index(&self, i: Self::Index) -> bool {
        self.grid.get_unchecked(i)
    }

    fn can_reconstruct(&self, i: Self::Index) -> bool {
        self.col_counts[Self::ind_to_col(i)] as usize > (P::HEIGHT / 2)
    }

    fn sample<R: RngCore>(&mut self, rng: &mut R, amount: usize) {
        let ufm = Uniform::new(0, P::WIDTH * P::HEIGHT);
        for _ in 0..amount {
            self.grid.set(ufm.sample(rng), true)
        }
    }

    fn sample_exclusion<R: RngCore>(&mut self, rng: &mut R, amount: usize, mask: &Self::Mask) {
        self.sample(rng, amount);
        self.grid.and_inplace(&mask)
    }

    fn merge(mut self, other: Self) -> Self {
        // Difference of bitvecs: where we need to update counts
        let xor = self.grid.xor_cloned(&other.grid);
        for i in 0..P::WH {
            if xor.get_unchecked(i) {
                self.col_counts[Self::ind_to_col(i)] += 1; 
            }
        }

        Self {
            grid: self.grid & other.grid,
            col_counts: self.col_counts,
            _phantom: PhantomData,
        }
    }
}


fn main() {
    println!("Hello, world!");
}
