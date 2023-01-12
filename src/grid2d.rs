use std::{
    fmt::{Debug, Display},
    ops::Not,
};

use bitvec_simd::BitVec;
use rand::{distributions::Uniform, prelude::Distribution, RngCore};

use crate::traits::{coord_to_ind, Reconstructable};

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
    grid: BitVec,
}

impl Grid2dErasure {
    pub fn from_bitvec(grid: BitVec, n: usize) -> Result<Self, ()> {
        if grid.len() != 4 * n * n {
            return Err(());
        }
        Ok(Self { n, grid })
    }
    #[inline(always)]
    fn w(&self) -> usize {
        2 * self.n
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

impl Display for Grid2dErasure {
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
fn col_row_counts(grid: &BitVec, w: usize, h: usize) -> (Vec<usize>, Vec<usize>) {
    let mut col_counts = vec![0usize; w];
    let mut row_counts = vec![0usize; h];
    for i in 0..h {
        for j in 0..w {
            if grid.get_unchecked(coord_to_ind(i, j, h)) {
                col_counts[j] += 1;
                row_counts[i] += 1;
            }
        }
    }
    (col_counts, row_counts)
}

fn reconstruct(grid: &mut BitVec, w: usize, h: usize) -> bool {
    // count number of cells in each column and row
    let (col_c, row_c) = col_row_counts(&grid, w, h);
    // try to reconstruct each column/row
    let starting_grid = grid.clone();
    for j in 0..w {
        // if we have enough, reconstruct everything
        if col_c[j] >= w / 2 {
            for i in 0..h {
                grid.set(coord_to_ind(i, j, h), true);
            }
        }
    }
    for i in 0..h {
        // if we have enough, reconstruct everything
        if row_c[i] >= h / 2 {
            for j in 0..w {
                grid.set(coord_to_ind(i, j, h), true);
            }
        }
    }
    grid != starting_grid
}

// Height in P must be even
impl Reconstructable for Grid2dErasure {
    type Index = usize;
    type Mask = BitVec;

    fn new(n: usize) -> Self {
        Grid2dErasure {
            n,
            grid: BitVec::zeros(4 * n * n),
        }
    }

    fn rand_index<R: RngCore>(rng: &mut R, n: usize) -> Self::Index {
        Uniform::new(0, 4 * n * n).sample(rng)
    }

    fn new_mask<R: RngCore>(rng: &mut R, n: usize) -> (Self::Mask, Self::Index) {
        let (w, h, wh) = (2 * n, 2 * n, 4 * n * n);
        let mut mask = BitVec::zeros(wh);
        // pick our special censored point in first half of the first row
        let col = Uniform::from(0..w / 2).sample(rng);
        mask.set(coord_to_ind(0, col, h), true);

        // Censor an additional WIDTH/2 points from end of the grid
        for j in (w / 2)..w {
            mask.set(coord_to_ind(0, j, h), true);
            // and for each of those points, censor HEIGHT/2 points from that column
            for i in (h / 2)..h {
                mask.set(coord_to_ind(i, j, h), true);
            }
        }
        assert!(mask.count_ones() > (w / 2) * (h / 2));

        (mask.not(), coord_to_ind(0, col, h))
    }

    fn can_reconstruct(&self, i: Self::Index) -> bool {
        // Is the cell present?
        if self.grid.get_unchecked(i) {
            return true;
        }
        let mut rgrid = self.grid.clone();
        // Try to reconstruct repeatedly until the grid stops changing
        let mut changed = true;
        let mut i = 0;
        while changed {
            i += 1;
            changed = reconstruct(&mut rgrid, self.w(), self.h());
        }
        return rgrid.get_unchecked(i);
    }

    #[inline(always)]
    fn sample<R: RngCore>(&mut self, rng: &mut R, amount: usize) {
        let ufm = Uniform::new(0, self.wh());
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

    fn dims() -> usize {
        2
    }

    fn grid_size(&self) -> usize {
        // Average of width and height
        (self.h() + self.h()) / 2
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn from_bool_grid(bools: [[bool; 4]; 4]) -> Grid2dErasure {
        let mut bv = BitVec::zeros(4 * 4);
        for i in 0..4 {
            for j in 0..4 {
                bv.set(coord_to_ind(i, j, 4), bools[i][j]);
            }
        }
        Grid2dErasure::from_bitvec(bv, 2).unwrap()
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
        let (w, h) = (g1.w(), g1.h());
        reconstruct(&mut g1.grid, w, h);
        let g2 = from_bool_grid([
            [true, true, true, true],
            [false, false, true, false],
            [false, false, false, false],
            [false, false, false, true],
        ]);
        assert_eq!(g1, g2);
        reconstruct(&mut g1.grid, w, h);
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
        println!("g1: {}", g1);
        let g2 = from_bool_grid([
            [true, true, true, false],
            [false, true, false, true],
            [false, false, true, false],
            [true, true, false, false],
        ]);
        println!("g2: {}", g2);
        let res_cmp = from_bool_grid([
            [true, true, true, false],
            [false, true, true, true],
            [false, true, true, false],
            [true, true, true, true],
        ]);
        let res = g1.merge(g2);
        println!("es: {}", res);
        println!("cm: {}", res_cmp);
        assert_eq!(res, res_cmp);
    }
}
