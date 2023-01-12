use std::{
    fmt::{Debug, Display},
    marker::PhantomData,
    ops::Not,
};

use bitvec_simd::BitVec;
use rand::{distributions::Uniform, prelude::Distribution, RngCore};

use crate::traits::{coord_to_ind, ind_to_col, Reconstructable};

#[derive(Debug, PartialEq)]
pub struct Grid1dErasure {
    n: usize,
    // the grid stored column wise to make adding along columns more efficient
    grid: BitVec,
}

impl Grid1dErasure {
    pub fn from_bitvec(grid: BitVec, n: usize) -> Result<Self, ()> {
        if grid.len() != 2 * n * n {
            return Err(());
        }
        Ok(Self { n, grid })
    }

    #[inline(always)]
    fn w(&self) -> usize {
        self.n
    }
    #[inline(always)]
    fn h(&self) -> usize {
        2 * self.n
    }
    #[inline(always)]
    fn wh(&self) -> usize {
        4 * self.n * self.n
    }
}

impl Display for Grid1dErasure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "\n")?;
        for i in 0..self.h() {
            for j in 0..self.w() {
                write!(
                    f,
                    " {} ",
                    self.grid.get_unchecked(coord_to_ind(i, j, self.h())) as u8
                )?;
            }
            write!(f, "\n")?;
        }
        Ok(())
    }
}

// Height in P must be even
impl Reconstructable for Grid1dErasure {
    type Index = usize;
    type Mask = BitVec;

    fn dims() -> usize {
        1
    }

    fn rand_index<R: RngCore>(rng: &mut R, n: usize) -> Self::Index {
        let wh = 2 * n * n;
        Uniform::new(0, wh).sample(rng)
    }

    fn new_mask<R: RngCore>(rng: &mut R, n: usize) -> (Self::Mask, Self::Index) {
        let (w, h) = (n, 2 * n);
        let col = Uniform::from(0..w).sample(rng);
        let mut mask = BitVec::zeros(w * h);
        // Set (0, col)..(n, col) true
        for i in 0..(h / 2) + 1 {
            mask.set(coord_to_ind(i, col, h), true);
        }
        (mask.not(), coord_to_ind(0, col, h))
    }

    fn new(n: usize) -> Self {
        Grid1dErasure {
            n,
            grid: BitVec::zeros(2 * n * n),
        }
    }

    fn grid_size(&self) -> usize {
        // Average of width/height
        (self.w() + self.h()) / 2
    }

    fn can_reconstruct(&self, i: Self::Index) -> bool {
        if self.grid.get_unchecked(i) {
            return true;
        }
        let col = ind_to_col(i, self.h());
        let start_ind = coord_to_ind(0, col, self.h());
        let end_ind = coord_to_ind(0, col + 1, self.h());
        let n_has = self.grid.count_ones_before(end_ind) - self.grid.count_ones_before(start_ind);
        n_has >= (self.h() / 2)
    }

    #[inline(always)]
    fn sample<R: RngCore>(&mut self, rng: &mut R, amount: usize) {
        let ufm = Uniform::new(0, self.w() * self.h());
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
        assert_eq!(self.n, other.n);
        Self {
            n: self.n,
            grid: self.grid | other.grid,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn from_bool_grid(bools: [[bool; 2]; 4]) -> Grid1dErasure {
        let mut bv = BitVec::zeros(2 * 4);
        for i in 0..4 {
            for j in 0..2 {
                bv.set(coord_to_ind(i, j, 4), bools[i][j]);
            }
        }
        Grid1dErasure::from_bitvec(bv, 2).unwrap()
    }

    #[test]
    fn test_merge() {
        let g1 = from_bool_grid([[true, true], [false, false], [false, false], [true, false]]);
        assert!(g1.can_reconstruct(coord_to_ind(0, 0, g1.h())));
        assert!(!g1.can_reconstruct(coord_to_ind(1, 1, g1.h())));
        assert!(g1.can_reconstruct(coord_to_ind(0, 1, g1.h())));
        let g2 = from_bool_grid([
            [true, true],
            [false, true],
            [false, false],
            [true, true],
        ]);
        let res_cmp = from_bool_grid([
            [true, true],
            [false, true],
            [false, false],
            [true, true],
        ]);
        println!("{}", &res_cmp);
        let res = g1.merge(g2);
        println!("{}", &res);
        assert_eq!(res, res_cmp);
    }
}
