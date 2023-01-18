use std::{
    fmt::Debug,
    ops::{BitOr, Not},
};

use bitvec_simd::BitVec;
use rand::{distributions::Uniform, prelude::Distribution, RngCore};

#[derive(PartialEq, Clone)]
pub struct Grid {
    bv: BitVec,
    w: usize,
    h: usize,
}

pub enum SampleStrategy {
    /// Split the grid into row_split rows and column_split columns, then sample those
    Box {
        row_split: usize,
        column_split: usize,
    },
    RandomPoints,
}

impl Grid {
    pub fn new(w: usize, h: usize) -> Self {
        Self {
            w,
            h,
            bv: BitVec::zeros(w * h),
        }
    }

    #[inline(always)]
    pub fn w(&self) -> usize {
        self.w
    }
    #[inline(always)]
    pub fn h(&self) -> usize {
        self.h
    }
    #[inline(always)]
    pub fn coord_to_ind(&self, row: usize, col: usize) -> usize {
        row + (col * self.h)
    }

    #[inline(always)]
    pub fn set(&mut self, row: usize, col: usize, value: bool) {
        self.bv.set(self.coord_to_ind(row, col), value);
    }

    #[inline(always)]
    pub fn get(&self, row: usize, col: usize) -> bool {
        self.bv.get_unchecked(self.coord_to_ind(row, col))
    }

    pub fn count_columnar(&self, col: usize) -> usize {
        let start_ind = self.coord_to_ind(0, col);
        let end_ind = self.coord_to_ind(0, col + 1);
        self.bv.count_ones_before(end_ind) - self.bv.count_ones_before(start_ind)
    }

    pub fn count_ones(&self) -> usize {
        self.bv.count_ones()
    }

    #[inline(always)]
    pub fn sample<R: RngCore>(&mut self, rng: &mut R, amount: usize, strategy: &SampleStrategy) {
        match strategy {
            SampleStrategy::Box {
                row_split: _,
                column_split: _,
            } => {
                unimplemented!()
            }
            SampleStrategy::RandomPoints => {
                let ufw = Uniform::new(0, self.bv.len());
                for _ in 0..amount {
                    self.bv.set(ufw.sample(rng), true);
                }
            }
        }
    }

    #[inline(always)]
    pub fn and_inplace(&mut self, mask: &Grid) {
        assert_eq!(self.w, mask.w);
        assert_eq!(self.h, mask.h);
        self.bv.and_inplace(&mask.bv);
    }

    pub fn not(self) -> Grid {
        Self {
            w: self.w,
            h: self.h,
            bv: self.bv.not(),
        }
    }

    #[cfg(test)]
    pub fn from_bool_grid<const W: usize, const H: usize>(bools: [[bool; W]; H]) -> Self {
        let mut grid = Self::new(W, H);
        for i in 0..H {
            for j in 0..W {
                grid.set(i, j, bools[i][j])
            }
        }
        grid
    }

    pub fn col_row_counts(&self) -> (Vec<usize>, Vec<usize>) {
        let mut col_counts = vec![0usize; self.w];
        let mut row_counts = vec![0usize; self.h];
        for i in 0..self.h {
            for j in 0..self.w {
                if self.get(i, j) {
                    row_counts[i] += 1;
                    col_counts[j] += 1;
                }
            }
        }
        (col_counts, row_counts)
    }
}

impl BitOr for Grid {
    type Output = Grid;

    fn bitor(self, rhs: Self) -> Self::Output {
        assert_eq!(self.w, rhs.w);
        assert_eq!(self.h, rhs.h);
        Self {
            w: self.w,
            h: self.h,
            bv: self.bv | rhs.bv,
        }
    }
}

impl Debug for Grid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "\n")?;
        for i in 0..self.h() {
            for j in 0..self.w() {
                write!(f, "{:^3}", self.get(i, j) as u8)?;
            }
            write!(f, "\n")?;
        }
        Ok(())
    }
}
