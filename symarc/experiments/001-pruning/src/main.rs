mod experiment;
use experiment::{Config, Outcome, Policy, Prepared, Score};
use std::{path::Path, time::Instant};
use symarc::report::number;

fn parallel<T: Sync, R: Send>(items: &[T], threads: usize, f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let next = AtomicUsize::new(0);
    let mut rows = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads.min(items.len()))
            .map(|_| {
                let f = &f;
                let next = &next;
                s.spawn(move || {
                    let mut v = vec![];
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= items.len() {
                            break;
                        }
                        v.push((i, f(&items[i])));
                    }
                    v
                })
            })
            .collect();
        hs.into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect::<Vec<_>>()
    });
    rows.sort_by_key(|(i, _)| *i);
    rows.into_iter().map(|(_, r)| r).collect()
}
struct Batch {
    policy: Policy,
    seed: u64,
    outcomes: Vec<Outcome>,
    scores: Vec<Score>,
    wall: f64,
}
fn mean(xs: impl Iterator<Item = f64>) -> f64 {
    let v: Vec<_> = xs.collect();
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}
fn sum_scores(b: &Batch, f: impl Fn(&Score) -> bool) -> usize {
    b.scores.iter().filter(|s| f(s)).count()
}
fn percent(n: f64, d: usize) -> String {
    if d == 0 {
        "n/a".into()
    } else {
        format!("{}%", number(n * 100.0 / d as f64))
    }
}
fn report(
    problems: &[Prepared],
    base: &[Score],
    batches: &[Batch],
    cfg: &Config,
    seeds: &[u64],
    workers: usize,
    prep: f64,
    data: &str,
) {
    let n = problems.len();
    let eligible = base.iter().filter(|s| s.eligible).count();
    let initial_correct = base.iter().filter(|s| s.correct).count();
    let initially_determined = base.iter().filter(|s| s.eligible && s.determined).count();
    println!("# 001-pruning: program-entropy policy comparison\n\nData: `{data}`. Tasks: {n}. Workers: {workers}. Seeds: {seeds:?}.\n");
    println!("## Before symmetry\n\nFixed depth-2 syntax class; no repair or search fallback. **{eligible}/{n} tasks have a fitting program**; {} are excluded from conditional metrics and unsolved in all-task accuracy.\n",n-eligible);
    println!("| Initial measurement | Value |\n|---|---:|\n| Correct selected programs | {initial_correct}/{n} ({}) |\n| Tasks with at least one correct program | {} |\n| Already determined among eligible | {initially_determined}/{eligible} |\n| Undetermined among eligible | {} |\n| Mean program entropy, eligible tasks | {} bits |\n| Mean answer entropy, eligible tasks | {} bits |\n",
        percent(initial_correct as f64,n),base.iter().filter(|s|s.oracle).count(),eligible-initially_determined,
        number(mean(base.iter().filter(|s|s.eligible).map(|s|s.program_bits))),number(mean(base.iter().filter(|s|s.eligible).map(|s|s.answer_bits))));
    println!("## Predictive results\n\nSingle chosen program, exact match on every test output in a task. Means and ranges vary the seed on the **same tasks**; seed repetitions are not independent task observations.\n\n| Policy | Mean correct tasks (min–max) | All-task accuracy | Eligible-task accuracy | Test-grid accuracy (mean) |\n|---|---:|---:|---:|---:|");
    for policy in Policy::ALL {
        let bs: Vec<_> = batches.iter().filter(|b| b.policy == policy).collect();
        let counts: Vec<_> = bs.iter().map(|b| sum_scores(b, |s| s.correct)).collect();
        let avg = mean(counts.iter().map(|&n| n as f64));
        let inputs = base.iter().map(|s| s.tests).sum();
        let grids = mean(
            bs.iter()
                .map(|b| b.scores.iter().map(|s| s.test_correct).sum::<usize>() as f64),
        );
        println!(
            "| {} | {} ({ }–{}) / {n} | {} | {} | {} |",
            policy.name(),
            number(avg),
            counts.iter().min().unwrap(),
            counts.iter().max().unwrap(),
            percent(avg, n),
            percent(avg, eligible),
            percent(grids, inputs)
        );
    }
    println!("\n## Entropy and failure diagnostics\n\nAnswer entropy includes undefinedness as an outcome; determination requires unanimous **defined** answers at every test input. Entropies average over eligible tasks (within each task, average over its test inputs). Lost-correct counts tasks where every initially available correct program was removed.\n\n| Policy | Mean pruning bits | Mean final program entropy | Mean final answer entropy | Newly determined tasks | Unanimous wrong tasks | Lost all correct programs |\n|---|---:|---:|---:|---:|---:|---:|");
    for policy in Policy::ALL {
        let bs: Vec<_> = batches.iter().filter(|b| b.policy == policy).collect();
        let h0 = mean(base.iter().filter(|s| s.eligible).map(|s| s.program_bits));
        let hp = mean(bs.iter().map(|b| {
            mean(
                b.scores
                    .iter()
                    .filter(|s| s.eligible)
                    .map(|s| s.program_bits),
            )
        }));
        let ha = mean(bs.iter().map(|b| {
            mean(
                b.scores
                    .iter()
                    .filter(|s| s.eligible)
                    .map(|s| s.answer_bits),
            )
        }));
        let newdet = mean(bs.iter().map(|b| {
            b.scores
                .iter()
                .zip(base)
                .filter(|(a, z)| a.eligible && a.determined && !z.determined)
                .count() as f64
        }));
        let wrong = mean(
            bs.iter()
                .map(|b| sum_scores(b, |s| s.eligible && s.determined && !s.correct) as f64),
        );
        let lost = mean(bs.iter().map(|b| {
            b.scores
                .iter()
                .zip(base)
                .filter(|(a, z)| z.oracle && !a.oracle)
                .count() as f64
        }));
        println!(
            "| {} | {} | {} | {} | {} | {} | {} |",
            policy.name(),
            number(h0 - hp),
            number(hp),
            number(ha),
            number(newdet),
            number(wrong),
            number(lost)
        );
    }
    println!("\n## Paired task changes\n\nWins/losses compare identical tasks and seeds; values are mean counts per seed.\n\n| Policy | Wins vs no symmetry | Losses vs no symmetry | Wins vs fixed | Losses vs fixed |\n|---|---:|---:|---:|---:|");
    for policy in Policy::ALL {
        let bs: Vec<_> = batches.iter().filter(|b| b.policy == policy).collect();
        let compare = |against_fixed: bool, wins: bool| {
            mean(bs.iter().map(|b| {
                let reference = if against_fixed {
                    &batches
                        .iter()
                        .find(|x| x.policy == Policy::Fixed && x.seed == b.seed)
                        .unwrap()
                        .scores
                } else {
                    base
                };
                b.scores
                    .iter()
                    .zip(reference)
                    .filter(|(a, z)| {
                        if wins {
                            a.correct && !z.correct
                        } else {
                            !a.correct && z.correct
                        }
                    })
                    .count() as f64
            }))
        };
        println!(
            "| {} | {} | {} | {} | {} |",
            policy.name(),
            number(compare(false, true)),
            number(compare(false, false)),
            number(compare(true, true)),
            number(compare(true, false))
        );
    }
    println!("\n## Work and approximation\n\nShared preparation (enumeration, functionality prefilter, cached program answers): {} s. Each policy/seed batch uses {workers} workers and is run separately. Charged work covers program-grid checks, not closure construction; actual work is not assumed equal merely because the ceilings match.\n\n| Policy | Mean selection wall seconds | Mean charged checks | Mean attempted trials | Mean capped completed trials | Mean budget-exhausted tasks |\n|---|---:|---:|---:|---:|---:|",number(prep));
    for policy in Policy::ALL {
        let bs: Vec<_> = batches.iter().filter(|b| b.policy == policy).collect();
        println!(
            "| {} | {} | {} | {} | {} | {} |",
            policy.name(),
            number(mean(bs.iter().map(|b| b.wall))),
            number(mean(bs.iter().map(
                |b| b.outcomes.iter().map(|o| o.charges).sum::<usize>() as f64
            ))),
            number(mean(bs.iter().map(
                |b| b.outcomes.iter().map(|o| o.trials).sum::<usize>() as f64
            ))),
            number(mean(bs.iter().map(
                |b| b.outcomes.iter().map(|o| o.capped_trials).sum::<usize>() as f64
            ))),
            number(mean(bs.iter().map(
                |b| b.outcomes.iter().filter(|o| o.exhausted).count() as f64
            )))
        );
    }
    println!("\nClosure cap: {}. Extra sample attempts: {}. Per-task/per-policy charge ceiling: {}. Full closures are checked exactly when completed; otherwise constraints are sampled. Surviving sets are nested, but they are **observed admissibility sets**, not proven full-closure version spaces. Selection also checks test-input commutation. Fixed/random can accept zero-pruning generators; marginal pruning stops when none strictly prunes.\n\nThis is development evidence from public training tasks only. The comparison disables stable-solver repair/fallback and stops on answer determination; it does not compare the complete stable solver or establish performance on public evaluation tasks.\n",cfg.cap,cfg.sample,cfg.budget);
    println!("## Per-seed task accuracy\n\n| Seed | Policy | Correct tasks |\n|---:|---|---:|");
    for b in batches {
        println!(
            "| {} | {} | {}/{n} |",
            b.seed,
            b.policy.name(),
            sum_scores(b, |s| s.correct)
        );
    }
    println!("\n## Outcomes for initially undetermined eligible tasks\n\nTasks whose answers already agree cannot change under the common stopping rule. `correct` scores the chosen shortest survivor; `lost` means all initially correct programs were removed.\n\n| Task | Seed | Policy | Programs before → after | Pruning bits | Answer bits before → after | Correct | Determined | Lost | Budget hit | Chosen program |\n|---|---:|---|---:|---:|---:|---|---|---|---|---|");
    for (i, p) in problems.iter().enumerate() {
        if !base[i].eligible || base[i].determined {
            continue;
        }
        for b in batches {
            let o = &b.outcomes[i];
            let s = &b.scores[i];
            let chosen = o
                .survivors
                .iter()
                .min_by_key(|&&j| p.problem.progs[j].len())
                .map(|&j| symarc::dsl::program_name(&p.problem.progs[j]))
                .unwrap_or("undefined".into());
            println!(
                "| {} | {} | {} | {} → {} | {} | {} → {} | {} | {} | {} | {} | `{}` |",
                p.problem.id,
                b.seed,
                b.policy.name(),
                p.problem.progs.len(),
                o.survivors.len(),
                number(base[i].program_bits - s.program_bits),
                number(base[i].answer_bits),
                number(s.answer_bits),
                s.correct,
                s.determined,
                base[i].oracle && !s.oracle,
                o.exhausted,
                chosen
            );
        }
    }
}
fn main() {
    let mut cfg = Config::default();
    let mut workers = 12;
    let mut seeds = vec![0, 1, 2, 3, 4];
    let mut data = "../data/training".to_string();
    let mut limit = usize::MAX;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let v = args.next().expect("option needs a value");
        match a.as_str() {
            "--data" => data = v,
            "--threads" => workers = v.parse().unwrap(),
            "--cap" => cfg.cap = v.parse().unwrap(),
            "--sample" => cfg.sample = v.parse().unwrap(),
            "--budget" => cfg.budget = v.parse().unwrap(),
            "--seeds" => seeds = v.split(',').map(|s| s.parse().unwrap()).collect(),
            "--limit" => limit = v.parse().unwrap(),
            _ => panic!("unknown option {a}"),
        }
    }
    assert!(workers > 0 && cfg.cap > 0 && !seeds.is_empty());
    let mut paths: Vec<_> = std::fs::read_dir(&data)
        .unwrap()
        .map(|p| p.unwrap().path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("json"))
        .collect();
    paths.sort();
    paths.truncate(limit);
    let tasks: Vec<_> = paths
        .iter()
        .map(|p| {
            symarc::task::load(
                p.to_str().unwrap(),
                p.file_stem().unwrap().to_str().unwrap(),
            )
        })
        .collect();
    assert!(
        !tasks.is_empty(),
        "no tasks in {}",
        Path::new(&data).display()
    );
    let start = Instant::now();
    let problems = parallel(&tasks, workers, experiment::prepare);
    let prep = start.elapsed().as_secs_f64();
    eprintln!("prepared {} tasks in {:.2}s", tasks.len(), prep);
    let base: Vec<_> = problems
        .iter()
        .map(|p| experiment::score(p, &(0..p.problem.progs.len()).collect::<Vec<_>>()))
        .collect();
    let mut batches = vec![];
    for &seed in &seeds {
        for policy in Policy::ALL {
            let start = Instant::now();
            let outcomes = parallel(&problems, workers, |p| {
                experiment::select(&p.problem, policy, seed, &cfg)
            });
            let wall = start.elapsed().as_secs_f64();
            let scores = problems
                .iter()
                .zip(&outcomes)
                .map(|(p, o)| experiment::score(p, &o.survivors))
                .collect();
            eprintln!("seed {seed} {} {:.2}s", policy.name(), wall);
            batches.push(Batch {
                policy,
                seed,
                outcomes,
                scores,
                wall,
            });
        }
    }
    report(
        &problems, &base, &batches, &cfg, &seeds, workers, prep, &data,
    );
}
