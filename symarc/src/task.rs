//! ARC task loading and task-specific candidate construction.
use crate::grid::{candidates, palette, Example, Gen, Grid};
use crate::{dsl, json};

pub struct Task {
    pub id: String,
    pub train: Vec<Example>,
    pub test: Vec<Example>,
}

impl Task {
    pub fn all_grids(&self) -> Vec<&Grid> {
        let mut v: Vec<&Grid> = Vec::new();
        for (x, y) in &self.train {
            v.push(x);
            v.push(y);
        }
        for (x, _) in &self.test {
            v.push(x);
        }
        v
    }
}

fn to_grid(j: &json::Json) -> Grid {
    let rows: Vec<Vec<u8>> = j
        .arr()
        .iter()
        .map(|r| {
            r.arr()
                .iter()
                .map(|c| u8::try_from(c.num()).expect("colour outside byte range"))
                .collect()
        })
        .collect();
    assert!(
        !rows.is_empty() && !rows[0].is_empty() && rows.iter().all(|r| r.len() == rows[0].len()),
        "malformed grid"
    );
    Grid::of_rows(&rows)
}

pub fn load(path: &str, id: &str) -> Task {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let j = json::parse(&bytes);
    let pairs = |key: &str| -> Vec<Example> {
        j.get(key)
            .unwrap()
            .arr()
            .iter()
            .map(|p| {
                (
                    to_grid(p.get("input").unwrap()),
                    to_grid(p.get("output").unwrap()),
                )
            })
            .collect()
    };
    let task = Task {
        id: id.to_string(),
        train: pairs("train"),
        test: pairs("test"),
    };
    assert!(!task.train.is_empty(), "{path}: no training examples");
    task
}

impl Task {
    pub fn candidates_pool(&self) -> (Vec<Gen>, Vec<dsl::Prim>) {
        let all = self.all_grids();
        let cands = candidates(
            all.iter().map(|g| g.h).max().unwrap_or(0),
            all.iter().map(|g| g.w).max().unwrap_or(0),
            &palette(&all),
        );
        let mut cols = vec![];
        for g in all {
            for c in g.colours() {
                if !cols.contains(&c) {
                    cols.push(c);
                }
            }
        }
        (cands, dsl::pool(&cols))
    }
}
