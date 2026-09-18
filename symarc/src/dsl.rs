//! The program DSL. Pool order determines enumeration and search tie-breaking.

use crate::grid::{Example, Grid, MAX_SIDE};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BoolOp {
    And,
    Or,
    Xor,
    Nor,
}
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dir {
    Horiz,
    Vert,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Prim {
    Rot90,
    Rot180,
    Rot270,
    FlipH,
    FlipV,
    Transpose,
    HConcat(bool),
    VConcat(bool),
    Mirror4,
    Tile(usize, usize),
    Scale(usize),
    CropBBox,
    CropLargest,
    CropSmallest,
    LeftHalf,
    RightHalf,
    TopHalf,
    BottomHalf,
    DedupRows,
    DedupCols,
    GravityDown,
    ShiftDown,
    ShiftUp,
    ShiftLeft,
    ShiftRight,
    Recolour(u8, u8),
    RecolourNonzero(u8),
    KeepColour(u8),
    RemoveColour(u8),
    FillRows,
    FillCols,
    Box(u8),
    Split(Dir, BoolOp, u8),
    SymmetrizeH,
    SymmetrizeV,
    Symmetrize4,
    FillEnclosed(u8),
    CountRow,
    MajorityFill,
}

/// 4-connected components of non-background cells, in reverse discovery order
/// Ties in component selection follow this order.
fn components(g: &Grid) -> Vec<Vec<(usize, usize)>> {
    let mut seen = vec![false; g.h * g.w];
    let mut comps: Vec<Vec<(usize, usize)>> = Vec::new();
    for i in 0..g.h {
        for j in 0..g.w {
            if g.get(i, j) != 0 && !seen[i * g.w + j] {
                let mut stack = vec![(i, j)];
                let mut comp = Vec::new();
                seen[i * g.w + j] = true;
                while let Some((a, b)) = stack.pop() {
                    comp.push((a, b));
                    let mut nbrs = vec![(a + 1, b), (a, b + 1)];
                    if a > 0 {
                        nbrs.push((a - 1, b));
                    }
                    if b > 0 {
                        nbrs.push((a, b - 1));
                    }
                    for (c, d) in nbrs {
                        if c < g.h && d < g.w && g.get(c, d) != 0 && !seen[c * g.w + d] {
                            seen[c * g.w + d] = true;
                            stack.push((c, d));
                        }
                    }
                }
                comps.insert(0, comp);
            }
        }
    }
    comps
}

fn bbox_of(cells: &[(usize, usize)]) -> (usize, usize, usize, usize) {
    let top = cells.iter().map(|c| c.0).min().unwrap();
    let bot = cells.iter().map(|c| c.0).max().unwrap();
    let left = cells.iter().map(|c| c.1).min().unwrap();
    let right = cells.iter().map(|c| c.1).max().unwrap();
    (top, left, bot + 1 - top, right + 1 - left)
}

fn crop_comp(g: &Grid, largest: bool) -> Option<Grid> {
    let comps = components(g);
    let mut best: Option<&Vec<(usize, usize)>> = None;
    for c in &comps {
        best = match best {
            None => Some(c),
            Some(a) => {
                let take = if largest {
                    c.len() > a.len()
                } else {
                    c.len() < a.len()
                };
                if take {
                    Some(c)
                } else {
                    Some(a)
                }
            }
        };
    }
    let c = best?;
    let (t, l, h, w) = bbox_of(c);
    Some(g.sub(t, l, h, w))
}

fn dedup(rows: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
    let mut out: Vec<Vec<u8>> = Vec::new();
    for r in rows {
        if out.last() != Some(&r) {
            out.push(r);
        }
    }
    out
}

fn halves(g: &Grid, d: Dir) -> Option<(Grid, Grid)> {
    match d {
        Dir::Horiz => {
            let n = g.w / 2;
            if n == 0 {
                None
            } else {
                Some((g.sub(0, 0, g.h, n), g.sub(0, g.w - n, g.h, n)))
            }
        }
        Dir::Vert => {
            let n = g.h / 2;
            if n == 0 {
                None
            } else {
                Some((g.sub(0, 0, n, g.w), g.sub(g.h - n, 0, n, g.w)))
            }
        }
    }
}

fn bool_op(op: BoolOp, a: bool, b: bool) -> bool {
    match op {
        BoolOp::And => a && b,
        BoolOp::Or => a || b,
        BoolOp::Xor => a != b,
        BoolOp::Nor => !(a || b),
    }
}

/// Background cells reachable from the border.
fn outside(g: &Grid) -> Vec<bool> {
    let mut seen = vec![false; g.h * g.w];
    let mut stack = Vec::new();
    for i in 0..g.h {
        for j in 0..g.w {
            if (i == 0 || j == 0 || i + 1 == g.h || j + 1 == g.w) && g.get(i, j) == 0 {
                stack.push((i, j));
                seen[i * g.w + j] = true;
            }
        }
    }
    while let Some((a, b)) = stack.pop() {
        let mut nbrs = vec![(a + 1, b), (a, b + 1)];
        if a > 0 {
            nbrs.push((a - 1, b));
        }
        if b > 0 {
            nbrs.push((a, b - 1));
        }
        for (c, d) in nbrs {
            if c < g.h && d < g.w && g.get(c, d) == 0 && !seen[c * g.w + d] {
                seen[c * g.w + d] = true;
                stack.push((c, d));
            }
        }
    }
    seen
}

fn majority(g: &Grid) -> Option<u8> {
    let mut best: Option<u8> = None;
    for c in g.nonzero_colours() {
        best = match best {
            None => Some(c),
            Some(a) => {
                if g.count_colour(c) > g.count_colour(a) {
                    Some(c)
                } else {
                    Some(a)
                }
            }
        };
    }
    best
}

impl Prim {
    pub fn eval(&self, g: &Grid) -> Option<Grid> {
        use Prim::*;
        Some(match *self {
            Rot90 => g.rot90(),
            Rot180 => g.rot180(),
            Rot270 => g.rot270(),
            FlipH => g.flip_h(),
            FlipV => g.flip_v(),
            Transpose => g.transpose(),
            HConcat(f) => {
                let r = if f { g.flip_h() } else { g.clone() };
                Grid::of_fn(g.h, 2 * g.w, |i, j| {
                    if j < g.w {
                        g.get(i, j)
                    } else {
                        r.get(i, j - g.w)
                    }
                })
            }
            VConcat(f) => {
                let b = if f { g.flip_v() } else { g.clone() };
                Grid::of_fn(2 * g.h, g.w, |i, j| {
                    if i < g.h {
                        g.get(i, j)
                    } else {
                        b.get(i - g.h, j)
                    }
                })
            }
            Mirror4 => Grid::of_fn(2 * g.h, 2 * g.w, |i, j| {
                let i2 = if i < g.h { i } else { 2 * g.h - 1 - i };
                let j2 = if j < g.w { j } else { 2 * g.w - 1 - j };
                g.get(i2, j2)
            }),
            Tile(m, n) => Grid::of_fn(m * g.h, n * g.w, |i, j| g.get(i % g.h, j % g.w)),
            Scale(k) => Grid::of_fn(k * g.h, k * g.w, |i, j| g.get(i / k, j / k)),
            CropBBox => {
                let (t, l, h, w) = g.bbox()?;
                g.sub(t, l, h, w)
            }
            CropLargest => crop_comp(g, true)?,
            CropSmallest => crop_comp(g, false)?,
            LeftHalf => halves(g, Dir::Horiz)?.0,
            RightHalf => halves(g, Dir::Horiz)?.1,
            TopHalf => halves(g, Dir::Vert)?.0,
            BottomHalf => halves(g, Dir::Vert)?.1,
            DedupRows => Grid::of_rows(&dedup(g.rows())),
            DedupCols => Grid::of_rows(&dedup(g.transpose().rows())).transpose(),
            GravityDown => {
                let mut cols: Vec<Vec<u8>> = Vec::with_capacity(g.w);
                for j in 0..g.w {
                    cols.push((0..g.h).map(|i| g.get(i, j)).filter(|&c| c != 0).collect());
                }
                Grid::of_fn(g.h, g.w, |i, j| {
                    let col = &cols[j];
                    let pad = g.h - col.len();
                    if i < pad {
                        0
                    } else {
                        col[i - pad]
                    }
                })
            }
            ShiftDown => Grid::of_fn(g.h, g.w, |i, j| if i == 0 { 0 } else { g.get(i - 1, j) }),
            ShiftUp => Grid::of_fn(
                g.h,
                g.w,
                |i, j| {
                    if i + 1 < g.h {
                        g.get(i + 1, j)
                    } else {
                        0
                    }
                },
            ),
            ShiftLeft => Grid::of_fn(
                g.h,
                g.w,
                |i, j| {
                    if j + 1 < g.w {
                        g.get(i, j + 1)
                    } else {
                        0
                    }
                },
            ),
            ShiftRight => Grid::of_fn(g.h, g.w, |i, j| if j == 0 { 0 } else { g.get(i, j - 1) }),
            Recolour(a, b) => g.map(|c| if c == a { b } else { c }),
            RecolourNonzero(k) => g.map(|c| if c == 0 { 0 } else { k }),
            KeepColour(c0) => g.map(|x| if x == c0 { x } else { 0 }),
            RemoveColour(c0) => g.map(|x| if x == c0 { 0 } else { x }),
            FillRows => {
                let first: Vec<Option<u8>> = (0..g.h)
                    .map(|i| (0..g.w).map(|j| g.get(i, j)).find(|&c| c != 0))
                    .collect();
                Grid::of_fn(g.h, g.w, |i, j| first[i].unwrap_or_else(|| g.get(i, j)))
            }
            FillCols => {
                let first: Vec<Option<u8>> = (0..g.w)
                    .map(|j| (0..g.h).map(|i| g.get(i, j)).find(|&c| c != 0))
                    .collect();
                Grid::of_fn(g.h, g.w, |i, j| first[j].unwrap_or_else(|| g.get(i, j)))
            }
            Box(k) => Grid::of_fn(g.h, g.w, |i, j| {
                if g.get(i, j) != 0 {
                    return g.get(i, j);
                }
                let mut near = vec![(i, j + 1), (i + 1, j), (i + 1, j + 1)];
                if i > 0 {
                    near.push((i - 1, j));
                    near.push((i - 1, j + 1));
                }
                if j > 0 {
                    near.push((i, j - 1));
                    near.push((i + 1, j - 1));
                }
                if i > 0 && j > 0 {
                    near.push((i - 1, j - 1));
                }
                if near
                    .iter()
                    .any(|&(a, b)| a < g.h && b < g.w && g.get(a, b) != 0)
                {
                    k
                } else {
                    0
                }
            }),
            Split(d, op, k) => {
                let (a, b) = halves(g, d)?;
                Grid::of_fn(a.h, a.w, |i, j| {
                    if bool_op(op, a.get(i, j) != 0, b.get(i, j) != 0) {
                        k
                    } else {
                        0
                    }
                })
            }
            SymmetrizeH => Grid::of_fn(g.h, g.w, |i, j| {
                if g.get(i, j) != 0 {
                    g.get(i, j)
                } else {
                    g.get(i, g.w - 1 - j)
                }
            }),
            SymmetrizeV => Grid::of_fn(g.h, g.w, |i, j| {
                if g.get(i, j) != 0 {
                    g.get(i, j)
                } else {
                    g.get(g.h - 1 - i, j)
                }
            }),
            Symmetrize4 => {
                if g.h != g.w {
                    return None;
                }
                Grid::of_fn(g.h, g.w, |i, j| {
                    let cands = [
                        g.get(i, j),
                        g.get(i, g.w - 1 - j),
                        g.get(g.h - 1 - i, j),
                        g.get(g.h - 1 - i, g.w - 1 - j),
                        g.get(j, i),
                        g.get(g.w - 1 - j, g.h - 1 - i),
                    ];
                    cands.into_iter().find(|&c| c != 0).unwrap_or(0)
                })
            }
            FillEnclosed(k) => {
                let out = outside(g);
                Grid::of_fn(g.h, g.w, |i, j| {
                    if g.get(i, j) == 0 && !out[i * g.w + j] {
                        k
                    } else {
                        g.get(i, j)
                    }
                })
            }
            CountRow => {
                let cs: Vec<u8> = g.cells.iter().copied().filter(|&c| c != 0).collect();
                if cs.is_empty() {
                    return None;
                }
                Grid {
                    h: 1,
                    w: cs.len(),
                    cells: cs,
                }
            }
            MajorityFill => {
                let c = majority(g)?;
                Grid::of_fn(g.h, g.w, |_, _| c)
            }
        })
    }
}

/// `Prim.pool`, in the same order as Lean.
pub fn pool(colours: &[u8]) -> Vec<Prim> {
    use BoolOp::*;
    use Dir::*;
    use Prim::*;
    let cs: Vec<u8> = colours.iter().copied().filter(|&c| c != 0).collect();
    let mut all = vec![0u8];
    all.extend_from_slice(&cs);
    let mut out = vec![
        Rot90,
        Rot180,
        Rot270,
        FlipH,
        FlipV,
        Transpose,
        HConcat(false),
        HConcat(true),
        VConcat(false),
        VConcat(true),
        Mirror4,
        Tile(1, 2),
        Tile(2, 1),
        Tile(2, 2),
        Tile(1, 3),
        Tile(3, 1),
        Tile(3, 3),
        Scale(2),
        Scale(3),
        CropBBox,
        CropLargest,
        CropSmallest,
        LeftHalf,
        RightHalf,
        TopHalf,
        BottomHalf,
        DedupRows,
        DedupCols,
        GravityDown,
        ShiftDown,
        ShiftUp,
        ShiftLeft,
        ShiftRight,
        FillRows,
        FillCols,
        SymmetrizeH,
        SymmetrizeV,
        Symmetrize4,
        CountRow,
        MajorityFill,
    ];
    for &a in &all {
        for &b in &all {
            if a != b {
                out.push(Recolour(a, b));
            }
        }
    }
    out.extend(cs.iter().map(|&k| RecolourNonzero(k)));
    out.extend(cs.iter().map(|&c| KeepColour(c)));
    out.extend(cs.iter().map(|&c| RemoveColour(c)));
    out.extend(cs.iter().map(|&k| Box(k)));
    out.extend(cs.iter().map(|&k| FillEnclosed(k)));
    for &k in &cs {
        for d in [Horiz, Vert] {
            for op in [And, Or, Xor, Nor] {
                out.push(Split(d, op, k));
            }
        }
    }
    out
}

#[inline]
pub fn eval_bounded(p: Prim, g: &Grid) -> Option<Grid> {
    let g2 = p.eval(g)?;
    if g2.h == 0 || g2.w == 0 || g2.h > MAX_SIDE || g2.w > MAX_SIDE {
        None
    } else {
        Some(g2)
    }
}

/// All programs of length at most `len` over `pool` that fit `s`, with prefix
/// outputs memoised.
pub fn enumerate(pool: &[Prim], s: &[Example], len: usize) -> Vec<Vec<Prim>> {
    let targets: Vec<&Grid> = s.iter().map(|(_, y)| y).collect();
    let inputs: Vec<Grid> = s.iter().map(|(x, _)| x.clone()).collect();
    let hits = |outs: &[Grid]| outs.iter().zip(&targets).all(|(a, b)| a == *b);
    let mut found: Vec<Vec<Prim>> = Vec::new();
    if hits(&inputs) {
        found.push(Vec::new());
    }
    let mut layer: Vec<(Vec<Prim>, Vec<Grid>)> = vec![(Vec::new(), inputs)];
    for _ in 0..len {
        let mut next: Vec<(Vec<Prim>, Vec<Grid>)> = Vec::new();
        for (p, outs) in &layer {
            for &pr in pool {
                let mut outs2: Vec<Grid> = Vec::with_capacity(outs.len());
                let mut ok = true;
                for g in outs {
                    match eval_bounded(pr, g) {
                        Some(g2) => outs2.push(g2),
                        None => {
                            ok = false;
                            break;
                        }
                    }
                }
                if ok {
                    let mut q = p.clone();
                    q.push(pr);
                    if hits(&outs2) {
                        found.push(q.clone());
                    }
                    next.push((q, outs2));
                }
            }
        }
        layer = next;
    }
    found
}

/// Stable program spelling used in experiment output.
pub fn program_name(p: &[Prim]) -> String {
    if p.is_empty() {
        "id".into()
    } else {
        p.iter().map(|p| p.name()).collect::<Vec<_>>().join(" ; ")
    }
}
impl Prim {
    pub fn name(&self) -> String {
        use Prim::*;
        match *self {
            HConcat(f) => if f { "hconcatFlip" } else { "hconcat" }.into(),
            VConcat(f) => if f { "vconcatFlip" } else { "vconcat" }.into(),
            Tile(m, n) => format!("tile {m} {n}"),
            Scale(k) => format!("scale {k}"),
            Recolour(a, b) => format!("recolour {a} {b}"),
            RecolourNonzero(k) => format!("recolourNonzero {k}"),
            KeepColour(k) => format!("keepColour {k}"),
            RemoveColour(k) => format!("removeColour {k}"),
            Box(k) => format!("box {k}"),
            FillEnclosed(k) => format!("fillEnclosed {k}"),
            Split(d, op, k) => format!(
                "split{} {} {k}",
                if d == Dir::Horiz { "H" } else { "V" },
                match op {
                    BoolOp::And => "and",
                    BoolOp::Or => "or",
                    BoolOp::Xor => "xor",
                    BoolOp::Nor => "nor",
                }
            ),
            _ => {
                let s = format!("{self:?}");
                format!("{}{}", s[..1].to_lowercase(), &s[1..])
            }
        }
    }
}
