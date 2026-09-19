//! Bounded path-order representation and matched histogram ablation.
use std::collections::{BTreeMap, BTreeSet};
use symarc::grid::Grid;
use symarc_exp_003_objects::objects::{Scene, Segment, Select};
use symarc_exp_003_objects::{Inputs, Pool};

#[derive(Clone, Copy, Debug)]
pub enum Order {
    Histogram,
    Path4,
    Path8,
}

pub fn unfold(
    scene: &Scene,
    selected: &[usize],
    order: Order,
    reverse: bool,
    row: bool,
) -> Option<Grid> {
    let mut cells = BTreeMap::new();
    for &i in selected {
        for &(r, c, v) in &scene.objects[i].cells {
            cells.insert((r, c), v);
        }
    }
    if cells.is_empty() || cells.len() > 30 {
        return None;
    }
    let mut values = vec![];
    match order {
        Order::Histogram => {
            let mut colours = vec![];
            for &v in cells.values() {
                if !colours.contains(&v) {
                    colours.push(v);
                }
            }
            if reverse {
                colours.reverse();
            }
            for v in colours {
                values
                    .extend(std::iter::repeat(v).take(cells.values().filter(|&&c| c == v).count()));
            }
        }
        Order::Path4 | Order::Path8 => {
            let mut graph = BTreeMap::new();
            for &(r, c) in cells.keys() {
                let neighbours: Vec<_> = cells
                    .keys()
                    .copied()
                    .filter(|&(rr, cc)| {
                        let dr = r.abs_diff(rr);
                        let dc = c.abs_diff(cc);
                        match order {
                            Order::Path4 => dr + dc == 1,
                            _ => dr <= 1 && dc <= 1 && dr + dc > 0,
                        }
                    })
                    .collect();
                graph.insert((r, c), neighbours);
            }
            if cells.len() == 1 {
                values.push(*cells.values().next().unwrap());
            } else {
                let ends: Vec<_> = graph
                    .iter()
                    .filter(|(_, ns)| ns.len() == 1)
                    .map(|(&p, _)| p)
                    .collect();
                if ends.len() != 2 || graph.values().any(|ns| ns.len() != 1 && ns.len() != 2) {
                    return None;
                }
                let mut current = ends[usize::from(reverse)];
                let mut seen = BTreeSet::new();
                loop {
                    if !seen.insert(current) {
                        return None;
                    }
                    values.push(cells[&current]);
                    let next = graph[&current].iter().find(|p| !seen.contains(p));
                    match next {
                        Some(&p) => current = p,
                        None => break,
                    }
                }
                if seen.len() != cells.len() {
                    return None;
                }
            }
        }
    }
    Some(if row {
        Grid::of_fn(1, values.len(), |_, c| values[c])
    } else {
        Grid::of_fn(values.len(), 1, |r, _| values[r])
    })
}

pub fn infer(input: &Inputs, path: bool) -> Pool {
    let mut palette = input.colours();
    palette.sort_unstable();
    let orders = if path {
        vec![Order::Path4, Order::Path8]
    } else {
        vec![Order::Histogram]
    };
    let mut pool = Pool::default();
    for seg in Segment::all() {
        let trains: Vec<_> = input
            .train
            .iter()
            .map(|(x, _)| Scene::parse(x, seg))
            .collect();
        let queries: Vec<_> = input.queries.iter().map(|x| Scene::parse(x, seg)).collect();
        for select in Select::all(&palette) {
            let selected: Vec<_> = trains.iter().map(|s| select.indices(s)).collect();
            for &order in &orders {
                for reverse in [false, true] {
                    for row in [false, true] {
                        pool.candidates += 1;
                        let fits = trains.iter().zip(&selected).zip(&input.train).all(
                            |((scene, indices), (_, y))| {
                                pool.checks += 1;
                                unfold(scene, indices, order, reverse, row).as_ref() == Some(y)
                            },
                        );
                        if fits {
                            pool.names.push(format!(
                                "{} / {:?} / {:?} / {} / {}",
                                seg.name(),
                                select,
                                order,
                                if reverse { "last" } else { "first" },
                                if row { "row" } else { "column" }
                            ));
                            pool.preds.push(
                                queries
                                    .iter()
                                    .map(|s| unfold(s, &select.indices(s), order, reverse, row))
                                    .collect(),
                            );
                        }
                    }
                }
            }
        }
    }
    pool
}

#[cfg(test)]
mod tests {
    use super::*;
    fn grid(rs: &[&[u8]]) -> Grid {
        Grid::of_rows(&rs.iter().map(|r| r.to_vec()).collect::<Vec<_>>())
    }
    fn parse(g: &Grid) -> Scene {
        Scene::parse(
            g,
            Segment {
                modal_background: false,
                diagonal: false,
                monochrome: false,
            },
        )
    }
    fn all(g: &Grid, o: Order, rev: bool, row: bool) -> Option<Grid> {
        let s = parse(g);
        unfold(&s, &(0..s.objects.len()).collect::<Vec<_>>(), o, rev, row)
    }
    #[test]
    fn repeated_colour_and_layout() {
        let g = grid(&[&[1, 2, 0], &[0, 3, 1]]);
        assert_eq!(
            all(&g, Order::Path4, false, true),
            Some(grid(&[&[1, 2, 3, 1]]))
        );
        assert_eq!(
            all(&g, Order::Histogram, false, true),
            Some(grid(&[&[1, 1, 2, 3]]))
        );
        assert_eq!(
            all(&g, Order::Path4, true, false),
            Some(grid(&[&[1], &[3], &[2], &[1]]))
        );
    }
    #[test]
    fn reject_branches_cycles_and_disconnection() {
        for g in [
            grid(&[&[0, 1, 0], &[1, 1, 1]]),
            grid(&[&[1, 1], &[1, 1]]),
            grid(&[&[1, 0, 1]]),
        ] {
            assert!(all(&g, Order::Path4, false, true).is_none());
        }
    }
    #[test]
    fn adjacency_and_limits() {
        let g = grid(&[&[1, 0], &[0, 2]]);
        assert!(all(&g, Order::Path4, false, true).is_none());
        assert_eq!(all(&g, Order::Path8, false, true), Some(grid(&[&[1, 2]])));
        assert_eq!(
            all(&grid(&[&[3]]), Order::Path4, false, true),
            Some(grid(&[&[3]]))
        );
        assert!(all(&Grid::of_fn(1, 31, |_, _| 1), Order::Path4, false, true).is_none());
        assert!(all(&grid(&[&[0]]), Order::Path4, false, true).is_none());
    }
    #[test]
    fn extra_evidence_separates_histogram_from_path() {
        let first = grid(&[&[1, 2, 0], &[0, 3, 4]]);
        let second = grid(&[&[1, 2, 0], &[0, 3, 1]]);
        let mut input = Inputs {
            train: vec![(first, grid(&[&[1, 2, 3, 4]]))],
            queries: vec![second.clone()],
        };
        assert!(!infer(&input, false).preds.is_empty());
        assert!(!infer(&input, true).preds.is_empty());
        input.train.push((second, grid(&[&[1, 2, 3, 1]])));
        assert!(infer(&input, false).preds.is_empty());
        assert!(!infer(&input, true).preds.is_empty());
    }
    #[test]
    fn query_labels_cannot_change_discovery() {
        let mut task = symarc::Task {
            id: "synthetic".into(),
            train: vec![(grid(&[&[1, 2, 0], &[0, 3, 4]]), grid(&[&[1, 2, 3, 4]]))],
            test: vec![(grid(&[&[1, 2, 0], &[0, 3, 1]]), grid(&[&[9]]))],
        };
        let before = infer(&Inputs::from_task(&task, false), true);
        task.test[0].1 = grid(&[&[8, 8]]);
        let after = infer(&Inputs::from_task(&task, false), true);
        assert_eq!(before.names, after.names);
        assert_eq!(before.preds, after.preds);
    }
}
