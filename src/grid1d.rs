use std::{
    fmt::{Debug, Display},
    marker::PhantomData,
    ops::Not,
};

use bitvec_simd::BitVec;
use rand::{distributions::Uniform, prelude::Distribution, RngCore};

use crate::traits::{GridParams, Reconstructable};

#[derive(Debug, PartialEq)]
pub struct Grid1dErasure<P: GridParams> {
    // the grid stored column wise to make adding along columns more efficient
    grid: BitVec,
    // counts of each column
    _phantom: PhantomData<P>,
}

impl<P: GridParams> Grid1dErasure<P> {
    pub fn from_bitvec(grid: BitVec) -> Result<Self, ()> {
        if grid.len() != P::WH {
            return Err(());
        }
        Ok(Self {
            grid,
            _phantom: PhantomData,
        })
    }

    #[inline(always)]
    pub fn coord_to_ind(i: usize, j: usize) -> usize {
        i + j * P::HEIGHT
    }

    #[inline(always)]
    pub fn ind_to_col(z: usize) -> usize {
        z / P::HEIGHT
    }

    #[inline(always)]
    pub fn ind_to_row(z: usize) -> usize {
        z % P::HEIGHT
    }
}

impl<P: GridParams> Display for Grid1dErasure<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "\n")?;
        for i in 0..P::HEIGHT {
            for j in 0..P::WIDTH {
                write!(
                    f,
                    " {} ",
                    self.grid.get_unchecked(Self::coord_to_ind(i, j)) as u8
                )?;
            }
            write!(f, "\n")?;
        }
        Ok(())
    }
}

// Height in P must be even
impl<P: GridParams> Reconstructable for Grid1dErasure<P> {
    type Index = usize;
    type Mask = BitVec;

    fn new() -> Self {
        Grid1dErasure {
            grid: BitVec::zeros(P::WH),
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
        //self.col_counts[Self::ind_to_col(i)] as usize > (P::HEIGHT / 2)
        let col = Self::ind_to_col(i);
        let start_ind = Self::coord_to_ind(0, col);
        let end_ind = Self::coord_to_ind(0, col + 1);
        let n_has = self.grid.count_ones_before(end_ind) - self.grid.count_ones_before(start_ind);
        n_has > (P::HEIGHT / 2)
    }

    #[inline(always)]
    fn sample<R: RngCore>(&mut self, rng: &mut R, amount: usize) {
        let ufm = Uniform::new(0, P::WIDTH * P::HEIGHT);
        for _ in 0..amount {
            self.grid.set(ufm.sample(rng), true)
        }
    }

    #[inline(always)]
    fn sample_exclusion<R: RngCore>(&mut self, rng: &mut R, amount: usize, mask: &Self::Mask) {
        self.sample(rng, amount);
        self.grid.and_inplace(&mask)
    }

    #[inline(always)]
    fn merge(self, other: Self) -> Self {
        Self {
            grid: self.grid | other.grid,
            _phantom: PhantomData,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[derive(Debug, PartialEq)]
    struct SmallGridParams;
    impl GridParams for SmallGridParams {
        const WIDTH: usize = 4;
        const HEIGHT: usize = 4;
    }
    type SmallGrid = Grid1dErasure<SmallGridParams>;

    fn from_bool_grid(bools: [[bool; 4]; 4]) -> SmallGrid {
        let mut bv = BitVec::zeros(4 * 4);
        for i in 0..4 {
            for j in 0..4 {
                bv.set(SmallGrid::coord_to_ind(i, j), bools[i][j]);
            }
        }
        SmallGrid::from_bitvec(bv).unwrap()
    }

    #[test]
    fn test_merge() {
        let g1 = from_bool_grid([
            [true, true, false, false],
            [false, false, true, true],
            [false, false, false, false],
            [true, true, true, true],
        ]);
        assert!(!g1.can_reconstruct(2));
        println!("g1: {}", g1);
        let g2 = from_bool_grid([
            [true, true, true, false],
            [false, true, false, true],
            [false, false, true, false],
            [true, true, false, false],
        ]);
        assert!(!g2.can_reconstruct(2));
        println!("g2: {}", g2);
        let res_cmp = from_bool_grid([
            [true, true, true, false],
            [false, true, true, true],
            [false, false, true, false],
            [true, true, true, true],
        ]);
        assert!(!g2.can_reconstruct(1));
        assert!(!g2.can_reconstruct(2));
        let res = g1.merge(g2);
        println!("es: {}", res);
        assert_eq!(res, res_cmp);
    }
}
