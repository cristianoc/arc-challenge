//! Grids, symmetry generators, and capped closure. Cells are bytes in a flat buffer.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

pub const MAX_SIDE: usize = 30;

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Grid {
    pub h: usize,
    pub w: usize,
    pub cells: Vec<u8>,
}

impl Grid {
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> u8 {
        if i < self.h && j < self.w {
            self.cells[i * self.w + j]
        } else {
            0
        }
    }
    #[inline]
    pub fn of_fn<F: FnMut(usize, usize) -> u8>(h: usize, w: usize, mut f: F) -> Grid {
        let mut cells = Vec::with_capacity(h * w);
        for i in 0..h {
            for j in 0..w {
                cells.push(f(i, j));
            }
        }
        Grid { h, w, cells }
    }
    pub fn of_rows(rows: &[Vec<u8>]) -> Grid {
        let h = rows.len();
        let w = rows[0].len();
        let mut cells = Vec::with_capacity(h * w);
        for r in rows {
            cells.extend_from_slice(r);
        }
        Grid { h, w, cells }
    }
    pub fn rows(&self) -> Vec<Vec<u8>> {
        (0..self.h)
            .map(|i| self.cells[i * self.w..(i + 1) * self.w].to_vec())
            .collect()
    }
    pub fn map<F: Fn(u8) -> u8>(&self, f: F) -> Grid {
        Grid {
            h: self.h,
            w: self.w,
            cells: self.cells.iter().map(|&c| f(c)).collect(),
        }
    }
    pub fn sub(&self, top: usize, left: usize, h: usize, w: usize) -> Grid {
        Grid::of_fn(h, w, |i, j| self.get(top + i, left + j))
    }
    pub fn flip_h(&self) -> Grid {
        Grid::of_fn(self.h, self.w, |i, j| self.get(i, self.w - 1 - j))
    }
    pub fn flip_v(&self) -> Grid {
        Grid::of_fn(self.h, self.w, |i, j| self.get(self.h - 1 - i, j))
    }
    pub fn transpose(&self) -> Grid {
        Grid::of_fn(self.w, self.h, |i, j| self.get(j, i))
    }
    pub fn rot90(&self) -> Grid {
        Grid::of_fn(self.w, self.h, |i, j| self.get(self.h - 1 - j, i))
    }
    pub fn rot180(&self) -> Grid {
        self.flip_h().flip_v()
    }
    pub fn rot270(&self) -> Grid {
        self.rot90().rot90().rot90()
    }
    pub fn cycle_rows(&self) -> Grid {
        Grid::of_fn(self.h, self.w, |i, j| {
            self.get((i + self.h - 1) % self.h, j)
        })
    }
    pub fn cycle_cols(&self) -> Grid {
        Grid::of_fn(self.h, self.w, |i, j| {
            self.get(i, (j + self.w - 1) % self.w)
        })
    }
    /// Colours in first-seen order (row-major), as `Grid.colours`.
    pub fn colours(&self) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::new();
        for &c in &self.cells {
            if !out.contains(&c) {
                out.push(c);
            }
        }
        out
    }
    pub fn nonzero_colours(&self) -> Vec<u8> {
        self.colours().into_iter().filter(|&c| c != 0).collect()
    }
    pub fn count_colour(&self, c: u8) -> usize {
        self.cells.iter().filter(|&&x| x == c).count()
    }
    /// (top, left, h, w) of the non-background cells.
    pub fn bbox(&self) -> Option<(usize, usize, usize, usize)> {
        let (mut top, mut left, mut bot, mut right) = (self.h, self.w, 0usize, 0usize);
        let mut any = false;
        for i in 0..self.h {
            for j in 0..self.w {
                if self.get(i, j) != 0 {
                    any = true;
                    top = top.min(i);
                    left = left.min(j);
                    bot = bot.max(i);
                    right = right.max(j);
                }
            }
        }
        if any {
            Some((top, left, bot + 1 - top, right + 1 - left))
        } else {
            None
        }
    }
}

/// FxHash-style hasher, so grid keys are cheap.
#[derive(Default)]
pub struct FxHasher {
    hash: u64,
}
const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;
impl FxHasher {
    #[inline]
    fn add(&mut self, w: u64) {
        self.hash = (self.hash.rotate_left(5) ^ w).wrapping_mul(SEED);
    }
}
impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for c in &mut chunks {
            self.add(u64::from_le_bytes(c.try_into().unwrap()));
        }
        let mut tail = 0u64;
        for (k, &b) in chunks.remainder().iter().enumerate() {
            tail |= (b as u64) << (8 * k);
        }
        self.add(tail);
    }
    #[inline]
    fn write_usize(&mut self, n: usize) {
        self.add(n as u64);
    }
    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
}
pub type FxMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;

/// Symmetry hypotheses, in the order used by greedy selection.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Gen {
    FlipH,
    FlipV,
    Transpose,
    SwapRows(usize, usize),
    SwapCols(usize, usize),
    SwapColours(u8, u8),
    CycleRows,
    CycleCols,
}

impl Gen {
    pub fn act(&self, g: &Grid) -> Option<Grid> {
        match *self {
            Gen::FlipH => Some(g.flip_h()),
            Gen::FlipV => Some(g.flip_v()),
            Gen::Transpose => Some(g.transpose()),
            Gen::SwapRows(a, b) => {
                if a < g.h && b < g.h {
                    Some(Grid::of_fn(g.h, g.w, |i, j| {
                        g.get(
                            if i == a {
                                b
                            } else if i == b {
                                a
                            } else {
                                i
                            },
                            j,
                        )
                    }))
                } else {
                    None
                }
            }
            Gen::SwapCols(a, b) => {
                if a < g.w && b < g.w {
                    Some(Grid::of_fn(g.h, g.w, |i, j| {
                        g.get(
                            i,
                            if j == a {
                                b
                            } else if j == b {
                                a
                            } else {
                                j
                            },
                        )
                    }))
                } else {
                    None
                }
            }
            Gen::SwapColours(a, b) => Some(g.map(|c| {
                if c == a {
                    b
                } else if c == b {
                    a
                } else {
                    c
                }
            })),
            Gen::CycleRows => Some(g.cycle_rows()),
            Gen::CycleCols => Some(g.cycle_cols()),
        }
    }
}

/// `Gen.candidates`: dihedral, colour swaps, cyclic, row swaps, column swaps.
pub fn candidates(max_h: usize, max_w: usize, palette: &[u8]) -> Vec<Gen> {
    let mut out = vec![Gen::FlipH, Gen::FlipV, Gen::Transpose];
    for &a in palette {
        for &b in palette {
            if a < b {
                out.push(Gen::SwapColours(a, b));
            }
        }
    }
    out.push(Gen::CycleRows);
    out.push(Gen::CycleCols);
    for i in 0..max_h.saturating_sub(1) {
        out.push(Gen::SwapRows(i, i + 1));
    }
    for j in 0..max_w.saturating_sub(1) {
        out.push(Gen::SwapCols(j, j + 1));
    }
    out
}

/// Nonzero colours present, in first-seen order, plus the smallest absent one.
pub fn palette(grids: &[&Grid]) -> Vec<u8> {
    let mut present: Vec<u8> = Vec::new();
    for g in grids {
        for c in g.nonzero_colours() {
            if !present.contains(&c) {
                present.push(c);
            }
        }
    }
    if let Some(fresh) = (1u8..=9).find(|c| !present.contains(c)) {
        present.push(fresh);
    }
    present
}

pub type Example = (Grid, Grid);

pub struct Closure {
    pub pairs: Vec<Example>,
    pub out_of: FxMap<Grid, Grid>,
    pub functional: bool,
    pub capped: bool,
}

impl Closure {
    pub fn size(&self) -> usize {
        self.pairs.len()
    }
}

/// Breadth-first closure of `d` under `gens`, at most `cap` pairs.
pub fn closure(gens: &[Gen], d: &[Example], cap: usize) -> Closure {
    let mut out_of: FxMap<Grid, Grid> = FxMap::default();
    let mut pairs: Vec<Example> = Vec::new();
    let mut functional = true;
    for (x, y) in d {
        match out_of.get(x) {
            Some(y2) => {
                if y2 != y {
                    functional = false;
                }
            }
            None => {
                out_of.insert(x.clone(), y.clone());
                pairs.push((x.clone(), y.clone()));
            }
        }
    }
    let mut i = 0;
    let mut capped = false;
    while i < pairs.len() {
        if pairs.len() >= cap {
            capped = true;
            break;
        }
        let (x, y) = pairs[i].clone();
        for g in gens {
            if let (Some(x2), Some(y2)) = (g.act(&x), g.act(&y)) {
                match out_of.get(&x2) {
                    Some(y3) => {
                        if *y3 != y2 {
                            functional = false;
                        }
                    }
                    None => {
                        out_of.insert(x2.clone(), y2.clone());
                        pairs.push((x2, y2));
                    }
                }
            }
        }
        i += 1;
    }
    Closure {
        pairs,
        out_of,
        functional,
        capped,
    }
}

/// Greedy: keep a generator while the closure stays functional.
pub fn greedy_gens(cands: &[Gen], d: &[Example], cap: usize) -> (Vec<Gen>, Closure) {
    let mut acc: Vec<Gen> = Vec::new();
    let mut best = closure(&[], d, cap);
    for &g in cands {
        let mut trial = acc.clone();
        trial.push(g);
        let c = closure(&trial, d, cap);
        if c.functional {
            acc.push(g);
            best = c;
        }
    }
    (acc, best)
}

impl Gen {
    pub fn family(&self) -> &'static str {
        match self {
            Self::FlipH | Self::FlipV | Self::Transpose => "dihedral",
            Self::SwapRows(..) => "rows",
            Self::SwapCols(..) => "cols",
            Self::SwapColours(..) => "colours",
            Self::CycleRows | Self::CycleCols => "cyclic",
        }
    }
    pub fn name(&self) -> String {
        match self {
            Self::SwapRows(a, b) => format!("swapRows {a} {b}"),
            Self::SwapCols(a, b) => format!("swapCols {a} {b}"),
            Self::SwapColours(a, b) => format!("swapColours {a} {b}"),
            _ => {
                let s = format!("{self:?}");
                format!("{}{}", s[..1].to_lowercase(), &s[1..])
            }
        }
    }
}
impl std::fmt::Display for Grid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, row) in self.rows().iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            for c in row {
                write!(f, "{c}")?;
            }
        }
        Ok(())
    }
}
