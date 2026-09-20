use std::collections::HashMap;
use symarc::grid::Grid;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Segment {
    pub modal_background: bool,
    pub diagonal: bool,
    pub monochrome: bool,
}
impl Segment {
    pub fn all() -> Vec<Self> {
        let mut out = vec![];
        for modal_background in [false, true] {
            for diagonal in [false, true] {
                for monochrome in [true, false] {
                    out.push(Self {
                        modal_background,
                        diagonal,
                        monochrome,
                    });
                }
            }
        }
        out
    }
    pub fn name(self) -> String {
        format!(
            "{}-{}-{}",
            if self.modal_background {
                "modal"
            } else {
                "zero"
            },
            if self.diagonal { "8" } else { "4" },
            if self.monochrome { "mono" } else { "multi" }
        )
    }
}
#[derive(Clone, Debug)]
pub struct Object {
    pub cells: Vec<(usize, usize, u8)>,
    pub top: usize,
    pub left: usize,
    pub height: usize,
    pub width: usize,
    pub colour: u8,
    pub shape_count: usize,
    pub colour_count: usize,
}
#[derive(Clone)]
pub struct Scene {
    pub grid: Grid,
    pub background: u8,
    pub objects: Vec<Object>,
}
fn modal(cs: impl Iterator<Item = u8>) -> u8 {
    let mut counts = [0usize; 256];
    for c in cs {
        counts[c as usize] += 1;
    }
    (0..256)
        .max_by_key(|&c| (counts[c], std::cmp::Reverse(c)))
        .unwrap() as u8
}
impl Scene {
    pub fn parse(grid: &Grid, seg: Segment) -> Self {
        let background = if seg.modal_background {
            modal(grid.cells.iter().copied())
        } else {
            0
        };
        let mut seen = vec![false; grid.cells.len()];
        let mut objects = vec![];
        for row in 0..grid.h {
            for col in 0..grid.w {
                let colour = grid.get(row, col);
                if colour == background || seen[row * grid.w + col] {
                    continue;
                }
                seen[row * grid.w + col] = true;
                let mut stack = vec![(row, col)];
                let mut cells = vec![];
                while let Some((r, c)) = stack.pop() {
                    cells.push((r, c, grid.get(r, c)));
                    for dr in -1isize..=1 {
                        for dc in -1isize..=1 {
                            if (dr == 0 && dc == 0) || (!seg.diagonal && dr != 0 && dc != 0) {
                                continue;
                            }
                            let rr = r as isize + dr;
                            let cc = c as isize + dc;
                            if rr < 0 || cc < 0 || rr >= grid.h as isize || cc >= grid.w as isize {
                                continue;
                            }
                            let rr = rr as usize;
                            let cc = cc as usize;
                            let value = grid.get(rr, cc);
                            if !seen[rr * grid.w + cc]
                                && value != background
                                && (!seg.monochrome || value == colour)
                            {
                                seen[rr * grid.w + cc] = true;
                                stack.push((rr, cc));
                            }
                        }
                    }
                }
                cells.sort_unstable();
                let top = cells.iter().map(|p| p.0).min().unwrap();
                let left = cells.iter().map(|p| p.1).min().unwrap();
                let height = cells.iter().map(|p| p.0).max().unwrap() - top + 1;
                let width = cells.iter().map(|p| p.1).max().unwrap() - left + 1;
                let colour = modal(cells.iter().map(|p| p.2));
                objects.push(Object {
                    cells,
                    top,
                    left,
                    height,
                    width,
                    colour,
                    shape_count: 0,
                    colour_count: 0,
                });
            }
        }
        let shapes: Vec<Vec<_>> = objects
            .iter()
            .map(|o| {
                o.cells
                    .iter()
                    .map(|&(r, c, _)| (r - o.top, c - o.left))
                    .collect()
            })
            .collect();
        let mut sc = HashMap::new();
        let mut cc = HashMap::new();
        for (o, shape) in objects.iter().zip(&shapes) {
            *sc.entry(shape).or_insert(0) += 1;
            *cc.entry(o.colour).or_insert(0) += 1;
        }
        for (o, shape) in objects.iter_mut().zip(&shapes) {
            o.shape_count = sc[shape];
            o.colour_count = cc[&o.colour];
        }
        Self {
            grid: grid.clone(),
            background,
            objects,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Select {
    All,
    Largest,
    Smallest,
    Widest,
    Tallest,
    UniqueShape,
    RepeatedShape,
    UniqueColour,
    RepeatedColour,
    Colour(u8),
}
impl Select {
    pub fn all(palette: &[u8]) -> Vec<Self> {
        let mut s = vec![
            Self::All,
            Self::Largest,
            Self::Smallest,
            Self::Widest,
            Self::Tallest,
            Self::UniqueShape,
            Self::RepeatedShape,
            Self::UniqueColour,
            Self::RepeatedColour,
        ];
        s.extend(palette.iter().map(|&c| Self::Colour(c)));
        s
    }
    pub fn indices(self, scene: &Scene) -> Vec<usize> {
        let os = &scene.objects;
        let max_area = os.iter().map(|o| o.cells.len()).max().unwrap_or(0);
        let min_area = os.iter().map(|o| o.cells.len()).min().unwrap_or(0);
        let max_width = os.iter().map(|o| o.width).max().unwrap_or(0);
        let max_height = os.iter().map(|o| o.height).max().unwrap_or(0);
        os.iter()
            .enumerate()
            .filter(|(_, o)| match self {
                Self::All => true,
                Self::Largest => o.cells.len() == max_area,
                Self::Smallest => o.cells.len() == min_area,
                Self::Widest => o.width == max_width,
                Self::Tallest => o.height == max_height,
                Self::UniqueShape => o.shape_count == 1,
                Self::RepeatedShape => o.shape_count > 1,
                Self::UniqueColour => o.colour_count == 1,
                Self::RepeatedColour => o.colour_count > 1,
                Self::Colour(c) => o.colour == c,
            })
            .map(|(i, _)| i)
            .collect()
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Identity,
    Erase,
    Recolour(u8),
    Top,
    Bottom,
    Left,
    Right,
    MirrorH,
    MirrorV,
}
impl Action {
    pub fn all(palette: &[u8]) -> Vec<Self> {
        let mut a = vec![Self::Identity, Self::Erase];
        a.extend(palette.iter().map(|&c| Self::Recolour(c)));
        a.extend([
            Self::Top,
            Self::Bottom,
            Self::Left,
            Self::Right,
            Self::MirrorH,
            Self::MirrorV,
        ]);
        a
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Render {
    Original,
    Selected,
    Crop,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Program {
    pub segment: Segment,
    pub select: Select,
    pub action: Action,
    pub render: Render,
}
impl Program {
    pub fn name(&self) -> String {
        format!(
            "{} / {:?} / {:?} / {:?}",
            self.segment.name(),
            self.select,
            self.action,
            self.render
        )
    }
    #[cfg(test)]
    pub fn eval(&self, g: &Grid) -> Option<Grid> {
        let s = Scene::parse(g, self.segment);
        render(&s, &self.select.indices(&s), self.action, self.render)
    }
}
pub fn render(scene: &Scene, selected: &[usize], action: Action, render: Render) -> Option<Grid> {
    if selected.is_empty() {
        return None;
    }
    if action == Action::Erase && render != Render::Original {
        return None;
    }
    let grid = &scene.grid;
    let bg = scene.background;
    let mut out = if render == Render::Original {
        grid.clone()
    } else {
        Grid::of_fn(grid.h, grid.w, |_, _| bg)
    };
    for &i in selected {
        for &(r, c, _) in &scene.objects[i].cells {
            out.cells[r * grid.w + c] = bg;
        }
    }
    if action == Action::Erase {
        return Some(out);
    }
    let mut occupied = vec![false; grid.cells.len()];
    let mut top = grid.h;
    let mut left = grid.w;
    let mut bottom = 0;
    let mut right = 0;
    for &i in selected {
        let o = &scene.objects[i];
        for &(r, c, v) in &o.cells {
            let (rr, cc) = match action {
                Action::Top => (r - o.top, c),
                Action::Bottom => (r - o.top + grid.h - o.height, c),
                Action::Left => (r, c - o.left),
                Action::Right => (r, c - o.left + grid.w - o.width),
                Action::MirrorH => (r, o.left + o.width - 1 - (c - o.left)),
                Action::MirrorV => (o.top + o.height - 1 - (r - o.top), c),
                _ => (r, c),
            };
            if rr >= grid.h || cc >= grid.w {
                return None;
            }
            let at = rr * grid.w + cc;
            if occupied[at] || (render == Render::Original && out.cells[at] != bg) {
                return None;
            }
            occupied[at] = true;
            out.cells[at] = if let Action::Recolour(k) = action {
                k
            } else {
                v
            };
            top = top.min(rr);
            left = left.min(cc);
            bottom = bottom.max(rr);
            right = right.max(cc);
        }
    }
    if render == Render::Crop {
        Some(out.sub(top, left, bottom - top + 1, right - left + 1))
    } else {
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn grid(rs: &[&[u8]]) -> Grid {
        Grid::of_rows(&rs.iter().map(|r| r.to_vec()).collect::<Vec<_>>())
    }
    fn seg() -> Segment {
        Segment {
            modal_background: false,
            diagonal: false,
            monochrome: true,
        }
    }
    #[test]
    fn partitions() {
        let g = grid(&[&[1, 2, 0], &[0, 0, 1]]);
        assert_eq!(Scene::parse(&g, seg()).objects.len(), 3);
        assert_eq!(
            Scene::parse(
                &g,
                Segment {
                    monochrome: false,
                    ..seg()
                }
            )
            .objects
            .len(),
            2
        );
        assert_eq!(
            Scene::parse(
                &g,
                Segment {
                    diagonal: true,
                    monochrome: false,
                    ..seg()
                }
            )
            .objects
            .len(),
            1
        );
        assert_eq!(
            Scene::parse(
                &g,
                Segment {
                    diagonal: true,
                    ..seg()
                }
            )
            .objects
            .len(),
            3
        );
    }
    #[test]
    fn modal_background_and_ties() {
        let g = grid(&[&[2, 2, 2], &[2, 1, 2]]);
        let s = Scene::parse(
            &g,
            Segment {
                modal_background: true,
                ..seg()
            },
        );
        assert_eq!(s.background, 2);
        assert_eq!(s.objects[0].cells.len(), 1);
        assert_eq!(modal([2, 1].into_iter()), 1);
    }
    #[test]
    fn shape_relations_and_all_ties() {
        let s = Scene::parse(&grid(&[&[1, 0, 2, 0, 3, 3]]), seg());
        assert_eq!(Select::RepeatedShape.indices(&s), vec![0, 1]);
        assert_eq!(Select::UniqueShape.indices(&s), vec![2]);
        assert_eq!(Select::Smallest.indices(&s), vec![0, 1]);
    }
    #[test]
    fn crop_masks_unselected_cells() {
        let g = grid(&[&[1, 1, 1], &[1, 2, 1], &[1, 1, 1]]);
        let p = Program {
            segment: seg(),
            select: Select::Largest,
            action: Action::Identity,
            render: Render::Crop,
        };
        assert_eq!(
            p.eval(&g),
            Some(grid(&[&[1, 1, 1], &[1, 0, 1], &[1, 1, 1]]))
        );
    }
    #[test]
    fn edit_and_move() {
        let g = grid(&[&[0, 0, 0], &[0, 2, 0], &[0, 0, 0]]);
        let p = Program {
            segment: seg(),
            select: Select::All,
            action: Action::Top,
            render: Render::Original,
        };
        assert_eq!(
            p.eval(&g),
            Some(grid(&[&[0, 2, 0], &[0, 0, 0], &[0, 0, 0]]))
        );
        assert_eq!(
            Program {
                action: Action::Recolour(3),
                ..p
            }
            .eval(&g),
            Some(grid(&[&[0, 0, 0], &[0, 3, 0], &[0, 0, 0]]))
        );
    }
    #[test]
    fn collisions_and_empty_selection() {
        let g = grid(&[&[1], &[0], &[2]]);
        let p = Program {
            segment: seg(),
            select: Select::Colour(2),
            action: Action::Top,
            render: Render::Original,
        };
        assert!(p.eval(&g).is_none());
        assert!(Program {
            select: Select::All,
            ..p.clone()
        }
        .eval(&g)
        .is_none());
        assert!(Program {
            select: Select::Colour(4),
            ..p
        }
        .eval(&g)
        .is_none());
    }
    #[test]
    fn erase_and_reflection() {
        let g = grid(&[&[1, 1, 0], &[1, 0, 0]]);
        let p = Program {
            segment: seg(),
            select: Select::All,
            action: Action::MirrorH,
            render: Render::Original,
        };
        assert_eq!(p.eval(&g), Some(grid(&[&[1, 1, 0], &[0, 1, 0]])));
        assert_eq!(
            Program {
                action: Action::Erase,
                ..p
            }
            .eval(&g),
            Some(Grid::of_fn(2, 3, |_, _| 0))
        );
    }
}
