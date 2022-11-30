use std::{
    fmt::{Debug, Display},
    marker::PhantomData,
    ops::Not,
};

use bitvec_simd::BitVec;
use rand::{
    distributions::Uniform, prelude::Distribution, rngs::SmallRng, thread_rng, RngCore, SeedableRng,
};

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

#[derive(Debug, PartialEq)]
pub struct Grid1dErasure<P: GridParams> {
    // the grid stored column wise to make adding along columns more efficient
    grid: BitVec,
    // counts of each column
    col_counts: Vec<u16>,
    _phantom: PhantomData<P>,
}

impl<P: GridParams> Grid1dErasure<P> {
    pub fn from_bitvec(grid: BitVec) -> Result<Self, ()> {
        if grid.len() != P::WH {
            return Err(());
        }
        let mut col_counts = vec![0; P::WIDTH];
        for i in 0..P::WH {
            if grid.get_unchecked(i) {
                col_counts[Self::ind_to_col(i)] += 1;
            }
        }
        Ok(Self {
            grid,
            col_counts,
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
        self.col_counts.fmt(f)?;
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
        let diff = self.grid.clone().not().and_cloned(&other.grid);
        for i in 0..P::WH {
            if diff.get_unchecked(i) {
                self.col_counts[Self::ind_to_col(i)] += 1;
            }
        }

        Self {
            grid: self.grid | other.grid,
            col_counts: self.col_counts,
            _phantom: PhantomData,
        }
    }
}

struct GridParams256 {}
impl GridParams for GridParams256 {
    const WIDTH: usize = 256;
    const HEIGHT: usize = 256;
}

fn main() {
    use rayon::prelude::*;
    const NUM_EXPERIMENTS: usize = 2_000;
    const N_CLIENTS: usize = 2_000;
    const N_CLIENTS_CENSORED: usize = (2 * N_CLIENTS) / 3;
    const SAMPLES_PER_CLIENT: usize = 30;

    type Grid = Grid1dErasure<GridParams256>;

    let (mask, censor_target) = Grid::new_mask(&mut thread_rng());

    let mut recon_count = 0;
    for _ in 0..NUM_EXPERIMENTS {
        let censor_iter = (0..N_CLIENTS_CENSORED).into_par_iter().map_init(
            || SmallRng::from_entropy(),
            |rng, _| {
                let mut grid = Grid::new();
                grid.sample_exclusion(rng, SAMPLES_PER_CLIENT, &mask);
                grid
            },
        );
        let honest_iter = (0..N_CLIENTS - N_CLIENTS_CENSORED)
            .into_par_iter()
            .map_init(
                || SmallRng::from_entropy(),
                |rng, _| {
                    let mut grid = Grid::new();
                    grid.sample(rng, SAMPLES_PER_CLIENT);
                    grid
                },
            );

        let res = censor_iter
            .chain(honest_iter)
            .reduce(Grid::new, Grid::merge);

        let recon = res.can_reconstruct(censor_target);
        println!("Reconstruction success: {}", recon);
        recon_count += recon as i32;
    }

    println!(
        "Reconstruction rate: {}",
        (recon_count as f32) / (NUM_EXPERIMENTS as f32)
    )
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
            [false, false, true, false],
            [true, true, true, true],
        ]);
        let res = g1.merge(g2);
        println!("es: {}", res);
        assert_eq!(res, res_cmp);
    }
}
