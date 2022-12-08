use std::{
    fmt::{Debug, Display},
    marker::PhantomData,
    ops::Not,
};

use bitvec_simd::BitVec;
use rand::{distributions::Uniform, prelude::Distribution, RngCore};

use crate::traits::{GridParams, Reconstructable};

#[derive(Debug, PartialEq)]
/// This is a 2d grid with erasure encoding as follows
///     First, interpolate points on the 2d grid horizontally, doubling the width
///     Then interpolate this wide/short grid vertically,doubling the height
/// Points can be reconstructed if 
/// - It exist in the grid
/// - It belongs to a row with at least WIDTH/2 points
/// - It belongs to a row where at least WIDTH/2 points can be reconstructed
pub struct Grid2dErasure<P: GridParams> {
    // the grid stored column wise to make adding along columns more efficient
    grid: BitVec,
    // counts of each column
    _phantom: PhantomData<P>,
}

impl<P: GridParams> Grid2dErasure<P> {
    pub fn from_bitvec(grid: BitVec) -> Result<Self, ()> {
        if grid.len() != P::WH {
            return Err(());
        }
        Ok(Self {
            grid,
            _phantom: PhantomData,
        })
    }
}

impl<P: GridParams> Display for Grid2dErasure<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "\n")?;
        for i in 0..P::HEIGHT {
            for j in 0..P::WIDTH {
                write!(
                    f,
                    " {} ",
                    self.grid.get_unchecked(P::coord_to_ind(i, j)) as u8
                )?;
            }
            write!(f, "\n")?;
        }
        Ok(())
    }
}
fn col_row_counts<P: GridParams>(grid: &BitVec) -> (Vec<usize>, Vec<usize>) {
    let mut col_counts = vec![0usize; P::WIDTH];
    let mut row_counts = vec![0usize; P::HEIGHT];
    for i in 0..P::HEIGHT {
        for j in 0..P::WIDTH {
            if grid.get_unchecked(P::coord_to_ind(i, j)) {
                col_counts[j] += 1;
                row_counts[i] += 1;
            }
        }
    }
    (col_counts, row_counts)
}

fn reconstruct<P: GridParams>(grid: &mut BitVec) -> bool {
    // count number of cells in each column and row
    let (col_c, row_c) = col_row_counts::<P>(&grid);
    // try to reconstruct each column/row
    let starting_grid = grid.clone();
    for j in 0..P::WIDTH {
        // if we have enough, reconstruct everything
        if col_c[j] >= P::WIDTH / 2 {
            for i in 0..P::HEIGHT {
                grid.set(P::coord_to_ind(i, j), true);
            }
        }
    }
    for i in 0..P::HEIGHT {
        // if we have enough, reconstruct everything
        if row_c[i] >= P::HEIGHT / 2 {
            for j in 0..P::WIDTH {
                grid.set(P::coord_to_ind(i, j), true);
            }
        }
    }
    grid != starting_grid
}



// Height in P must be even
impl<P: GridParams> Reconstructable for Grid2dErasure<P> {
    type Index = usize;
    type Mask = BitVec;

    fn new() -> Self {
        Grid2dErasure {
            grid: BitVec::zeros(P::WH),
            _phantom: PhantomData,
        }
    }

    fn rand_index<R: RngCore>(rng: &mut R) -> Self::Index {
        Uniform::new(0, P::WIDTH * P::HEIGHT).sample(rng)
    }

    fn new_mask<R: RngCore>(rng: &mut R) -> (Self::Mask, Self::Index) {
        let mut mask = BitVec::zeros(P::WH);
        // pick our special censored point in first half of the first row
        let col = Uniform::from(0..P::WIDTH / 2).sample(rng);
        mask.set(P::coord_to_ind(0, col), true);

        // Censor an additional WIDTH/2 points from the first row
        for j in (P::WIDTH/2)..P::WIDTH {
            mask.set(P::coord_to_ind(0, j), true);
            // and for each of those points, censor HEIGHT/2 points from that column
            for i in (P::HEIGHT/2)..P::HEIGHT {
                mask.set(P::coord_to_ind(i, j), true);
            }
        }
        assert!(mask.count_ones() > (P::WIDTH/2)*(P::HEIGHT/2));

        (mask.not(), P::coord_to_ind(0, col))
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
            changed = reconstruct::<P>(&mut rgrid);
        }
        return rgrid.get_unchecked(i);
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
    type SmallGrid = Grid2dErasure<SmallGridParams>;

    fn from_bool_grid(bools: [[bool; 4]; 4]) -> SmallGrid {
        let mut bv = BitVec::zeros(4 * 4);
        for i in 0..4 {
            for j in 0..4 {
                bv.set(SmallGridParams::coord_to_ind(i, j), bools[i][j]);
            }
        }
        SmallGrid::from_bitvec(bv).unwrap()
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
        reconstruct::<SmallGridParams>(&mut g1.grid);
        let g2 = from_bool_grid([
            [true, true, true, true],
            [false, false, true, false],
            [false, false, false, false],
            [false, false, false, true],
        ]);
        assert_eq!(g1, g2);
        reconstruct::<SmallGridParams>(&mut g1.grid);
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
