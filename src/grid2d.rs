use std::fmt::Debug;

use rand::{distributions::Uniform, prelude::Distribution, RngCore};

use crate::{
    base_grid::{Grid, SampleStrategy},
    traits::Reconstructable,
};

#[derive(Debug, PartialEq)]
/// This is a 2d grid with erasure encoding as follows
///     First, interpolate points on the 2d grid horizontally, doubling the width
///     Then interpolate this wide/short grid vertically,doubling the height
/// Points can be reconstructed if
/// - It exist in the grid
/// - It belongs to a row with at least N/2 points
/// - It belongs to a row where at leastNWIDTH/2 points can be reconstructed
pub struct Grid2dErasure {
    // undelying size of grid
    n: usize,
    // the grid stored column wise to make adding along columns more efficient
    grid: Grid,
}

impl Grid2dErasure {
    #[cfg(test)]
    pub fn from_grid(grid: Grid, n: usize) -> Result<Self, ()> {
        if grid.w() != 2 * n || grid.h() != 2 * n {
            return Err(());
        }
        Ok(Self { n, grid })
    }
}

fn reconstruct(grid: &mut Grid) -> bool {
    // Make a copy of the grid we started with for comparison later
    let starting_grid = grid.clone();
    // count number of cells in each column and row
    let (col_c, row_c) = grid.col_row_counts();
    // For each column
    for j in 0..grid.w() {
        // if we have enough at least n points
        if col_c[j] >= grid.w() / 2 {
            // Reconstruct the whole column
            for i in 0..grid.h() {
                grid.set(i, j, true);
            }
        }
    }
    // For each row
    for i in 0..grid.h() {
        // if we have enough
        if row_c[i] >= grid.h() / 2 {
            // reconstruct everything in the row
            for j in 0..grid.w() {
                grid.set(i, j, true);
            }
        }
    }
    // The grid changed if the grid we started with and the
    // reconstruction we did are not the same
    grid != &starting_grid
}

// Height in P must be even
impl Reconstructable for Grid2dErasure {
    type Index = (usize, usize);

    fn new(n: usize) -> Self {
        Grid2dErasure {
            n,
            grid: Grid::new(2 * n, 2 * n),
        }
    }

    fn new_mask<R: RngCore>(rng: &mut R, n: usize) -> (Grid, Self::Index) {
        let mut mask = Grid::new(2 * n, 2 * n);
        // pick a point to censor in the first quadrant of the grid
        let col = Uniform::from(0..n).sample(rng);
        let row = Uniform::from(0..n).sample(rng);
        mask.set(row, col, true);

        // Censor n more points in that specific row and column, so not enough
        // erasure encoded data is directly available
        for k in n..2 * n {
            mask.set(row, k, true);
            mask.set(k, col, true);
        }

        // Then censor the n x n block in the last quadrant of the grid
        for i in n..2 * n {
            for j in n..2 * n {
                mask.set(i, j, true);
            }
        }
        // Check we censor
        // 1. The n x n block
        // 2. The point itself
        // 3. The n points in the points' row/column
        assert!(mask.count_ones() == n * n + 2 * n + 1);

        (mask.not(), (row, col))
    }

    fn can_reconstruct(&self, (i, j): Self::Index) -> bool {
        // Is the cell present?
        if self.grid.get(i, j) {
            return true;
        }
        let mut rgrid = self.grid.clone();
        // Try to reconstruct repeatedly until the grid stops changing
        let mut changed = true;
        while changed {
            changed = reconstruct(&mut rgrid);
        }
        return rgrid.get(i, j);
    }

    #[inline(always)]
    fn sample<R: RngCore>(&mut self, rng: &mut R, amount: usize) {
        self.grid.sample(rng, amount, &SampleStrategy::RandomPoints);
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

    fn dims() -> usize {
        2
    }

    fn grid_size(&self) -> usize {
        self.n
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn from_bool_grid(bools: [[bool; 4]; 4]) -> Grid2dErasure {
        let grid = Grid::from_bool_grid(bools);
        Grid2dErasure::from_grid(grid, 2).unwrap()
    }

    // Example of reconstruction with less than (W/2 + 1) * (H/2 + 1) points
    #[test]
    fn test_reconstruct() {
        let mut g1 = from_bool_grid([
            [true, true, false, false],
            [false, false, true, false],
            [false, false, false, false],
            [false, false, false, true],
        ]);
        reconstruct(&mut g1.grid);
        let g2 = from_bool_grid([
            [true, true, true, true],
            [false, false, true, false],
            [false, false, false, false],
            [false, false, false, true],
        ]);
        assert_eq!(g1, g2);
        reconstruct(&mut g1.grid);
        let g3 = from_bool_grid([
            [true, true, true, true],
            [false, false, true, true],
            [false, false, true, true],
            [false, false, true, true],
        ]);
        assert_eq!(g1, g3);
    }

    #[test]
    fn test_merge() {
        let g1 = from_bool_grid([
            [true, false, false, false],
            [false, false, true, true],
            [false, true, false, false],
            [true, true, true, true],
        ]);
        println!("g1: {:?}", g1);
        let g2 = from_bool_grid([
            [true, true, true, false],
            [false, true, false, true],
            [false, false, true, false],
            [true, true, false, false],
        ]);
        println!("g2: {:?}", g2);
        let res_cmp = from_bool_grid([
            [true, true, true, false],
            [false, true, true, true],
            [false, true, true, false],
            [true, true, true, true],
        ]);
        let res = g1.merge(g2);
        println!("es: {:?}", res);
        println!("cm: {:?}", res_cmp);
        assert_eq!(res, res_cmp);
    }
}
