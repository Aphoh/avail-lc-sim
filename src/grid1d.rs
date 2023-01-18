use std::fmt::Debug;

use rand::{distributions::Uniform, prelude::Distribution, RngCore};

use crate::{
    base_grid::{Grid, SampleStrategy},
    traits::Reconstructable,
};

#[derive(Debug, PartialEq)]
pub struct Grid1dErasure {
    n: usize,
    // the grid stored column wise to make adding along columns more efficient
    grid: Grid,
}

impl Grid1dErasure {
    #[cfg(test)]
    pub fn from_grid(grid: Grid, n: usize) -> Result<Self, ()> {
        if grid.w() != n || grid.h() != 2 * n {
            return Err(());
        }
        Ok(Self { n, grid })
    }
}

// Height in P must be even
impl Reconstructable for Grid1dErasure {
    type Index = (usize, usize);

    fn dims() -> usize {
        1
    }

    fn new_mask<R: RngCore>(rng: &mut R, n: usize) -> (Grid, Self::Index) {
        let mut mask = Grid::new(n, 2 * n);
        // Pick a point in the lower half to censor
        let row = Uniform::from(0..n).sample(rng);
        let col = Uniform::from(0..n).sample(rng);
        mask.set(row, col, true);
        // Censor n points from the upper half of the grid:
        // that is (n, col)..(2*n, col)
        for i in n..2 * n {
            mask.set(i, col, true);
        }
        assert_eq!(mask.count_ones(), n + 1);
        (mask.not(), (row, col))
    }

    fn new(n: usize) -> Self {
        Grid1dErasure {
            n,
            grid: Grid::new(n, 2 * n),
        }
    }

    fn grid_size(&self) -> usize {
        self.n
    }

    fn can_reconstruct(&self, (row, col): Self::Index) -> bool {
        if self.grid.get(row, col) {
            return true;
        }
        self.grid.count_columnar(col) >= self.n
    }

    #[inline(always)]
    fn sample<R: RngCore>(&mut self, rng: &mut R, amount: usize) {
        self.grid.sample(rng, amount, &SampleStrategy::RandomPoints)
    }

    #[inline(always)]
    fn sample_exclusion<R: RngCore>(&mut self, rng: &mut R, amount: usize, mask: &Grid) {
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
        let grid = Grid::from_bool_grid(bools);
        Grid1dErasure::from_grid(grid, 2).unwrap()
    }

    #[test]
    fn test_merge() {
        let g1 = from_bool_grid([[true, true], [false, false], [false, false], [true, false]]);
        assert!(g1.can_reconstruct((0, 0)));
        assert!(!g1.can_reconstruct((1, 1)));
        assert!(g1.can_reconstruct((0, 1)));
        let g2 = from_bool_grid([[true, true], [false, true], [false, false], [true, true]]);
        let res_cmp = from_bool_grid([[true, true], [false, true], [false, false], [true, true]]);
        println!("{:?}", &res_cmp);
        let res = g1.merge(g2);
        println!("{:?}", &res);
        assert_eq!(res, res_cmp);
    }
}
