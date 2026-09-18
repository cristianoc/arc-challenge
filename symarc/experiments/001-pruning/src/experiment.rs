use std::time::Instant;
use symarc::{
    dsl,
    grid::{closure, greedy_gens, Example, FxMap, Gen, Grid},
    random::Random,
    search::{self, Prog},
    Task,
};

#[derive(Clone, Debug)]
pub struct Config {
    pub cap: usize,
    pub sample: usize,
    pub budget: usize,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            cap: 300,
            sample: 64,
            budget: 500_000,
        }
    }
}

pub struct Problem {
    pub id: String,
    pub data: Vec<Example>,
    pub test_inputs: Vec<Grid>,
    pub progs: Vec<Prog>,
    pub gens: Vec<Gen>,
    pub answers: Vec<Vec<Option<Grid>>>,
}
pub struct Prepared {
    pub problem: Problem,
    pub targets: Vec<Grid>,
}
pub fn prepare(t: &Task) -> Prepared {
    let (candidates, pool) = t.candidates_pool();
    let progs = dsl::enumerate(&pool, &t.train, 2);
    let gens = if progs.is_empty() {
        vec![]
    } else {
        greedy_gens(&candidates, &t.train, 1000).0
    };
    let test_inputs: Vec<_> = t.test.iter().map(|(x, _)| x.clone()).collect();
    let answers = progs
        .iter()
        .map(|p| test_inputs.iter().map(|x| search::eval(p, x)).collect())
        .collect();
    Prepared {
        problem: Problem {
            id: t.id.clone(),
            data: t.train.clone(),
            test_inputs,
            progs,
            gens,
            answers,
        },
        targets: t.test.iter().map(|(_, y)| y.clone()).collect(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    None,
    Fixed,
    Random,
    Pruning,
}
impl Policy {
    pub const ALL: [Self; 4] = [Self::None, Self::Fixed, Self::Random, Self::Pruning];
    pub fn name(self) -> &'static str {
        match self {
            Self::None => "no-symmetry",
            Self::Fixed => "fixed-order",
            Self::Random => "random-order",
            Self::Pruning => "marginal-pruning",
        }
    }
}
#[derive(Default)]
pub struct Budget {
    pub used: usize,
    pub limit: usize,
    pub exhausted: bool,
}
impl Budget {
    fn charge(&mut self, n: usize) -> bool {
        if n > self.limit.saturating_sub(self.used) {
            self.exhausted = true;
            false
        } else {
            self.used += n;
            true
        }
    }
}
#[derive(Clone, Debug)]
pub struct Outcome {
    pub survivors: Vec<usize>,
    pub accepted: Vec<usize>,
    pub charges: usize,
    pub trials: usize,
    pub capped_trials: usize,
    pub conflicts: usize,
    pub exhausted: bool,
    pub selection_seconds: f64,
}

/// Canonical sets give the same sample regardless of the order of trial evaluation.
fn set_seed(seed: u64, ids: &[usize]) -> u64 {
    let mut h = seed ^ 0x9e3779b97f4a7c15;
    for &id in ids {
        h = (h ^ (id as u64 + 1)).wrapping_mul(0x100000001b3);
    }
    h
}
struct Trial {
    keep: Vec<usize>,
    capped: bool,
    conflict: bool,
}
fn trial(
    problem: &Problem,
    current: &[usize],
    accepted: &[usize],
    candidate: usize,
    cfg: &Config,
    seed: u64,
    budget: &mut Budget,
) -> Option<Trial> {
    let mut ids = accepted.to_vec();
    ids.push(candidate);
    ids.sort_unstable();
    let gens: Vec<_> = ids.iter().map(|&i| problem.gens[i]).collect();
    let c = closure(&gens, &problem.data, cfg.cap);
    if !c.functional {
        return Some(Trial {
            keep: vec![],
            capped: c.capped,
            conflict: true,
        });
    }
    let sample = if c.capped {
        search::fit_sample(
            &c,
            &gens,
            &problem.data,
            cfg.sample,
            &mut Random::new(set_seed(seed, &ids)),
        )
    } else {
        c.pairs.clone()
    };
    let mut keep = vec![];
    'program: for &i in current {
        let p = &problem.progs[i];
        // The current class already satisfies all previously accepted constraints.
        for x in &problem.test_inputs {
            if !budget.charge(2) {
                return None;
            }
            if !search::commutes_at(problem.gens[candidate], p, x) {
                continue 'program;
            }
        }
        for (x, y) in &sample {
            if !budget.charge(1) {
                return None;
            }
            if search::eval(p, x).as_ref() != Some(y) {
                continue 'program;
            }
        }
        keep.push(i);
    }
    Some(Trial {
        keep,
        capped: c.capped,
        conflict: false,
    })
}

pub fn answer_entropy(problem: &Problem, survivors: &[usize]) -> f64 {
    if survivors.is_empty() || problem.test_inputs.is_empty() {
        return 0.0;
    }
    let mut total = 0.0;
    for j in 0..problem.test_inputs.len() {
        let mut counts: FxMap<&Option<Grid>, usize> = FxMap::default();
        for &i in survivors {
            *counts.entry(&problem.answers[i][j]).or_default() += 1;
        }
        total += counts
            .values()
            .map(|&n| {
                let p = n as f64 / survivors.len() as f64;
                -p * p.log2()
            })
            .sum::<f64>();
    }
    total / problem.test_inputs.len() as f64
}
pub fn determined(problem: &Problem, survivors: &[usize]) -> bool {
    !survivors.is_empty()
        && !problem.test_inputs.is_empty()
        && (0..problem.test_inputs.len()).all(|j| {
            let first = &problem.answers[survivors[0]][j];
            first.is_some() && survivors.iter().all(|&i| &problem.answers[i][j] == first)
        })
}
pub fn select(problem: &Problem, policy: Policy, seed: u64, cfg: &Config) -> Outcome {
    let start = Instant::now();
    let mut out = Outcome {
        survivors: (0..problem.progs.len()).collect(),
        accepted: vec![],
        charges: 0,
        trials: 0,
        capped_trials: 0,
        conflicts: 0,
        exhausted: false,
        selection_seconds: 0.0,
    };
    if policy == Policy::None || out.survivors.is_empty() {
        return out;
    }
    let mut remaining: Vec<_> = (0..problem.gens.len()).collect();
    if policy == Policy::Random {
        let mut rng = Random::new(seed ^ 0xd1b54a32d192ed03);
        for i in (1..remaining.len()).rev() {
            let j = rng.range(0, i);
            remaining.swap(i, j);
        }
    }
    let mut budget = Budget {
        limit: cfg.budget,
        ..Default::default()
    };
    'selection: while !remaining.is_empty() && !determined(problem, &out.survivors) {
        if policy == Policy::Pruning {
            let mut best: Option<(usize, Vec<usize>)> = None;
            for (position, &candidate) in remaining.iter().enumerate() {
                out.trials += 1;
                let Some(t) = trial(
                    problem,
                    &out.survivors,
                    &out.accepted,
                    candidate,
                    cfg,
                    seed,
                    &mut budget,
                ) else {
                    break 'selection;
                };
                out.capped_trials += usize::from(t.capped);
                out.conflicts += usize::from(t.conflict);
                if !t.keep.is_empty()
                    && t.keep.len() < out.survivors.len()
                    && best
                        .as_ref()
                        .map_or(true, |(_, keep)| t.keep.len() < keep.len())
                {
                    best = Some((position, t.keep));
                }
            }
            if let Some((position, keep)) = best {
                out.accepted.push(remaining.remove(position));
                out.survivors = keep;
            } else {
                break;
            }
        } else {
            let candidate = remaining.remove(0);
            out.trials += 1;
            let Some(t) = trial(
                problem,
                &out.survivors,
                &out.accepted,
                candidate,
                cfg,
                seed,
                &mut budget,
            ) else {
                break;
            };
            out.capped_trials += usize::from(t.capped);
            out.conflicts += usize::from(t.conflict);
            if !t.keep.is_empty() {
                out.accepted.push(candidate);
                out.survivors = t.keep;
            }
        }
    }
    out.charges = budget.used;
    out.exhausted = budget.exhausted;
    out.selection_seconds = start.elapsed().as_secs_f64();
    out
}

#[derive(Clone, Debug)]
pub struct Score {
    pub eligible: bool,
    pub correct: bool,
    pub test_correct: usize,
    pub tests: usize,
    pub program_bits: f64,
    pub answer_bits: f64,
    pub determined: bool,
    pub oracle: bool,
}
/// Labels enter only here, after the selection policy has returned its survivors.
pub fn score(prepared: &Prepared, survivors: &[usize]) -> Score {
    let p = &prepared.problem;
    let matches = |i: usize| {
        p.answers[i]
            .iter()
            .zip(&prepared.targets)
            .filter(|(a, y)| a.as_ref() == Some(*y))
            .count()
    };
    let selected = survivors.iter().min_by_key(|&&i| p.progs[i].len()).copied();
    let test_correct = selected.map(matches).unwrap_or(0);
    Score {
        eligible: !p.progs.is_empty(),
        correct: selected.is_some() && test_correct == prepared.targets.len(),
        test_correct,
        tests: prepared.targets.len(),
        program_bits: if survivors.is_empty() {
            0.0
        } else {
            (survivors.len() as f64).log2()
        },
        answer_bits: answer_entropy(p, survivors),
        determined: determined(p, survivors),
        oracle: survivors
            .iter()
            .any(|&i| matches(i) == prepared.targets.len()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn toy() -> Task {
        Task {
            id: "toy".into(),
            train: vec![(
                Grid::of_rows(&[vec![0, 0, 0], vec![0, 7, 0], vec![0, 0, 0]]),
                Grid::of_rows(&[vec![0, 0, 0], vec![7, 7, 7], vec![0, 0, 0]]),
            )],
            test: vec![(
                Grid::of_rows(&[vec![0, 0, 0], vec![0, 0, 0], vec![0, 3, 0]]),
                Grid::of_rows(&[vec![0, 0, 0], vec![0, 0, 0], vec![3, 3, 3]]),
            )],
        }
    }
    #[test]
    fn toy_counts_and_entropy_match_exact_closure() {
        let t = toy();
        let pool = dsl::pool(&[0, 7]);
        let p0 = dsl::enumerate(&pool, &t.train, 2);
        assert_eq!(p0.len(), 30);
        let gens = greedy_gens(
            &symarc::grid::candidates(3, 3, &(1..=9).collect::<Vec<_>>()),
            &t.train,
            100000,
        )
        .0;
        let c = closure(&gens, &t.train, 100000);
        assert!(!c.capped);
        assert_eq!(c.size(), 81);
        assert_eq!(
            p0.iter().filter(|p| search::fits_all(p, &c.pairs)).count(),
            8
        );
    }
    #[test]
    fn undefined_is_an_entropy_category_but_not_a_determined_answer() {
        let mut p = prepare(&toy()).problem;
        p.answers = vec![vec![None], vec![Some(Grid::of_rows(&[vec![1]]))]];
        assert_eq!(answer_entropy(&p, &[0, 1]), 1.0);
        assert!(!determined(&p, &[0, 1]));
        assert_eq!(answer_entropy(&p, &[0]), 0.0);
        assert!(!determined(&p, &[0]));
        assert!(determined(&p, &[1]));
    }
    #[test]
    fn policies_keep_a_nonempty_subset_and_ignore_scoring_labels() {
        let t = toy();
        let a = prepare(&t);
        let mut t2 = toy();
        t2.test[0].1 = Grid::of_rows(&[vec![9]]);
        let b = prepare(&t2);
        let cfg = Config::default();
        for policy in Policy::ALL {
            let x = select(&a.problem, policy, 7, &cfg);
            let y = select(&b.problem, policy, 7, &cfg);
            assert_eq!(x.survivors, y.survivors);
            assert_eq!(x.accepted, y.accepted);
            assert!(!x.survivors.is_empty());
            assert!(x.survivors.iter().all(|&i| i < a.problem.progs.len()));
        }
    }
    #[test]
    fn zero_budget_does_not_commit_a_partial_ranking() {
        let a = prepare(&toy());
        let cfg = Config {
            budget: 0,
            ..Default::default()
        };
        let x = select(&a.problem, Policy::Pruning, 0, &cfg);
        assert_eq!(x.survivors.len(), a.problem.progs.len());
        assert!(x.accepted.is_empty());
        assert_eq!(x.charges, 0);
    }
    #[test]
    fn samples_and_survivors_are_repeatable_and_nested() {
        let a = prepare(&toy());
        let p = &a.problem;
        let cfg = Config {
            cap: 2,
            ..Default::default()
        };
        let initial: Vec<_> = (0..p.progs.len()).collect();
        let run = |candidate| {
            trial(
                p,
                &initial,
                &[],
                candidate,
                &cfg,
                3,
                &mut Budget {
                    limit: 100000,
                    ..Default::default()
                },
            )
            .unwrap()
            .keep
        };
        let x = run(0);
        let _ = run(1);
        assert_eq!(x, run(0));
        if !x.is_empty() {
            let y = trial(
                p,
                &x,
                &[0],
                1,
                &cfg,
                3,
                &mut Budget {
                    limit: 100000,
                    ..Default::default()
                },
            )
            .unwrap();
            assert!(y.keep.iter().all(|i| x.contains(i)));
        }
    }
}
