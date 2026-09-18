//! Shared object experiment language and inference; imported by later studies.
pub mod objects;
use objects::{Action, Program, Render, Scene, Segment, Select};
use std::collections::HashMap;
use symarc::{
    dsl,
    grid::{Example, Grid},
    search, Task,
};

pub struct Inputs {
    pub train: Vec<Example>,
    pub queries: Vec<Grid>,
}
impl Inputs {
    pub fn from_task(t: &Task, reserve: bool) -> Self {
        let n = t.train.len() - usize::from(reserve);
        let mut queries = vec![];
        if reserve {
            queries.push(t.train[n].0.clone());
        }
        queries.extend(t.test.iter().map(|(x, _)| x.clone()));
        Self {
            train: t.train[..n].to_vec(),
            queries,
        }
    }
    pub fn colours(&self) -> Vec<u8> {
        let mut out = vec![];
        for g in self
            .train
            .iter()
            .flat_map(|(x, y)| [x, y])
            .chain(self.queries.iter())
        {
            for c in g.colours() {
                if !out.contains(&c) {
                    out.push(c);
                }
            }
        }
        out
    }
}
#[derive(Default)]
pub struct Pool {
    pub names: Vec<String>,
    pub preds: Vec<Vec<Option<Grid>>>,
    pub segments: Vec<usize>,
    pub length_weights: Vec<f64>,
    pub candidates: usize,
    pub checks: usize,
}
pub fn core(input: &Inputs, depth: usize) -> Pool {
    let primitives = dsl::pool(&input.colours());
    let ps = dsl::enumerate(&primitives, &input.train, depth);
    let mut out = Pool::default();
    out.candidates = (0..=depth).map(|l| primitives.len().pow(l as u32)).sum();
    for p in ps {
        out.names.push(dsl::program_name(&p));
        out.length_weights
            .push((0.5 / primitives.len() as f64).powi(p.len() as i32));
        out.preds
            .push(input.queries.iter().map(|x| search::eval(&p, x)).collect());
    }
    out
}
pub fn object(input: &Inputs) -> Pool {
    let mut palette = input.colours();
    palette.sort_unstable();
    let selectors = Select::all(&palette);
    let actions = Action::all(&palette);
    let mut out = Pool::default();
    for (si, segment) in Segment::all().into_iter().enumerate() {
        let train: Vec<_> = input
            .train
            .iter()
            .map(|(x, _)| Scene::parse(x, segment))
            .collect();
        let queries: Vec<_> = input
            .queries
            .iter()
            .map(|x| Scene::parse(x, segment))
            .collect();
        for &select in &selectors {
            let st: Vec<_> = train.iter().map(|s| select.indices(s)).collect();
            let sq: Vec<_> = queries.iter().map(|s| select.indices(s)).collect();
            for &action in &actions {
                for render in [Render::Original, Render::Selected, Render::Crop] {
                    if action == Action::Erase && render != Render::Original {
                        continue;
                    }
                    out.candidates += 1;
                    let fits =
                        train
                            .iter()
                            .zip(&st)
                            .zip(&input.train)
                            .all(|((s, indices), (_, y))| {
                                out.checks += 1;
                                objects::render(s, indices, action, render).as_ref() == Some(y)
                            });
                    if !fits {
                        continue;
                    }
                    let program = Program {
                        segment,
                        select,
                        action,
                        render,
                    };
                    out.names.push(program.name());
                    out.segments.push(si);
                    out.length_weights.push(1.);
                    out.preds.push(
                        queries
                            .iter()
                            .zip(&sq)
                            .map(|(s, indices)| objects::render(s, indices, action, render))
                            .collect(),
                    );
                }
            }
        }
    }
    out
}
pub fn distribution(pool: &Pool, query: usize, weighted: bool) -> HashMap<Option<Grid>, f64> {
    let mut mass = HashMap::new();
    if pool.preds.is_empty() {
        mass.insert(None, 1.);
        return mass;
    }
    let total = if weighted {
        pool.length_weights.iter().sum()
    } else {
        pool.preds.len() as f64
    };
    for (i, p) in pool.preds.iter().enumerate() {
        *mass.entry(p[query].clone()).or_insert(0.) += if weighted {
            pool.length_weights[i] / total
        } else {
            1. / total
        };
    }
    mass
}
pub fn entropy(masses: impl Iterator<Item = f64>) -> f64 {
    masses.filter(|&v| v > 0.).map(|v| -v * v.log2()).sum()
}
