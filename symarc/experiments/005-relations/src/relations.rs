use std::collections::HashMap;
use symarc::grid::Grid;
use symarc_exp_003_objects::{
    objects::{self, Action, Render, Scene, Segment, Select},
    Inputs, Pool,
};
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Relation {
    Colour,
    MoveAbove,
    MoveBelow,
    MoveLeft,
    MoveRight,
    CopyAbove,
    CopyBelow,
    CopyLeft,
    CopyRight,
}
impl Relation {
    pub fn all() -> [Self; 9] {
        [
            Self::Colour,
            Self::MoveAbove,
            Self::MoveBelow,
            Self::MoveLeft,
            Self::MoveRight,
            Self::CopyAbove,
            Self::CopyBelow,
            Self::CopyLeft,
            Self::CopyRight,
        ]
    }
}
pub fn apply(scene: &Scene, source: usize, reference: usize, relation: Relation) -> Option<Grid> {
    if source == reference {
        return None;
    }
    let s = scene.objects.get(source)?;
    let t = scene.objects.get(reference)?;
    if relation == Relation::Colour {
        return objects::render(
            scene,
            &[source],
            Action::Recolour(t.colour),
            Render::Original,
        );
    }
    let position = match (relation as usize - 1) % 4 {
        0 => (t.top as isize - s.height as isize, t.left as isize),
        1 => ((t.top + t.height) as isize, t.left as isize),
        2 => (t.top as isize, t.left as isize - s.width as isize),
        _ => (t.top as isize, (t.left + t.width) as isize),
    };
    let copy = relation as usize >= 5;
    let mut out = scene.grid.clone();
    if !copy {
        for &(r, c, _) in &s.cells {
            out.cells[r * out.w + c] = scene.background;
        }
    }
    for &(r, c, value) in &s.cells {
        let rr = position.0 + (r - s.top) as isize;
        let cc = position.1 + (c - s.left) as isize;
        if rr < 0 || cc < 0 || rr >= out.h as isize || cc >= out.w as isize {
            return None;
        }
        let at = rr as usize * out.w + cc as usize;
        if out.cells[at] != scene.background {
            return None;
        }
        out.cells[at] = value;
    }
    Some(out)
}
struct Cached {
    scene: Scene,
    unique: Vec<Option<usize>>,
    outputs: HashMap<(usize, usize, Relation), Option<Grid>>,
}
impl Cached {
    fn new(g: &Grid, seg: Segment, selectors: &[Select]) -> Self {
        let scene = Scene::parse(g, seg);
        let unique = selectors
            .iter()
            .map(|s| {
                let is = s.indices(&scene);
                if is.len() == 1 {
                    Some(is[0])
                } else {
                    None
                }
            })
            .collect();
        Self {
            scene,
            unique,
            outputs: HashMap::new(),
        }
    }
    fn eval(&mut self, s: usize, t: usize, r: Relation) -> Option<Grid> {
        let s = self.unique[s]?;
        let t = self.unique[t]?;
        self.outputs
            .entry((s, t, r))
            .or_insert_with(|| apply(&self.scene, s, t, r))
            .clone()
    }
}
pub fn infer(input: &Inputs) -> Pool {
    let mut colours = input.colours();
    colours.sort_unstable();
    let selectors = Select::all(&colours);
    let mut pool = Pool::default();
    for (si, segment) in Segment::all().into_iter().enumerate() {
        let mut train: Vec<_> = input
            .train
            .iter()
            .map(|(x, _)| Cached::new(x, segment, &selectors))
            .collect();
        let mut queries: Vec<_> = input
            .queries
            .iter()
            .map(|x| Cached::new(x, segment, &selectors))
            .collect();
        for (s, source) in selectors.iter().enumerate() {
            for (t, reference) in selectors.iter().enumerate() {
                for relation in Relation::all() {
                    pool.candidates += 1;
                    let fits = train.iter_mut().zip(&input.train).all(|(scene, (_, y))| {
                        pool.checks += 1;
                        scene.eval(s, t, relation).as_ref() == Some(y)
                    });
                    if !fits {
                        continue;
                    }
                    pool.names.push(format!(
                        "{} / source {:?} / reference {:?} / {:?}",
                        segment.name(),
                        source,
                        reference,
                        relation
                    ));
                    pool.segments.push(si);
                    pool.length_weights.push(1.);
                    pool.preds
                        .push(queries.iter_mut().map(|q| q.eval(s, t, relation)).collect());
                }
            }
        }
    }
    pool
}
#[cfg(test)]
mod tests {
    use super::*;
    fn scene() -> Scene {
        Scene::parse(
            &Grid::of_rows(&[
                vec![1, 1, 0, 0, 0],
                vec![0, 0, 0, 0, 0],
                vec![0, 0, 2, 0, 0],
                vec![0, 0, 0, 0, 0],
            ]),
            Segment {
                modal_background: false,
                diagonal: false,
                monochrome: true,
            },
        )
    }
    #[test]
    fn colour_from_reference() {
        let s = scene();
        let out = apply(&s, 0, 1, Relation::Colour).unwrap();
        assert_eq!(out.get(0, 0), 2);
        assert_eq!(out.get(0, 1), 2);
        assert_eq!(out.get(2, 2), 2);
    }
    #[test]
    fn move_and_copy_positions() {
        let s = scene();
        let moved = apply(&s, 0, 1, Relation::MoveAbove).unwrap();
        let copied = apply(&s, 0, 1, Relation::CopyAbove).unwrap();
        assert_eq!(moved.get(1, 2), 1);
        assert_eq!(moved.get(1, 3), 1);
        assert_eq!(moved.get(0, 0), 0);
        assert_eq!(copied.get(0, 0), 1);
        assert_eq!(moved.get(2, 2), 2);
        assert_eq!(apply(&s, 0, 1, Relation::MoveLeft).unwrap().get(2, 0), 1);
    }
    #[test]
    fn bounds_collision_and_self_reference() {
        let mut s = scene();
        assert!(apply(&s, 0, 0, Relation::Colour).is_none());
        assert!(apply(&s, 1, 0, Relation::MoveAbove).is_none());
        s.grid.cells[1 * 5 + 2] = 3;
        assert!(apply(&s, 0, 1, Relation::MoveAbove).is_none());
    }
    #[test]
    fn selectors_must_be_unique() {
        let s = scene();
        let mut c = Cached::new(
            &s.grid,
            Segment {
                modal_background: false,
                diagonal: false,
                monochrome: true,
            },
            &[Select::All, Select::Largest, Select::Smallest],
        );
        assert!(c.eval(0, 2, Relation::Colour).is_none());
        assert!(c.eval(1, 2, Relation::Colour).is_some());
    }
}
