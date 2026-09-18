use std::{
    collections::{HashMap, HashSet},
    fs,
    io::{BufWriter, Write},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
    time::Instant,
};
use symarc::{grid::Grid, task, Task};
use symarc_exp_003_objects::{core, entropy, object, objects::Segment, Inputs, Pool};

#[derive(Clone)]
struct Candidate {
    name: String,
    predictions: Vec<Option<Grid>>,
    weight: f64,
}
struct Family {
    name: String,
    programs: Vec<Candidate>,
}
fn families(input: &Inputs) -> Vec<Family> {
    let mut fs = vec![Family {
        name: "grid".into(),
        programs: vec![],
    }];
    fs.extend(Segment::all().into_iter().map(|s| Family {
        name: s.name(),
        programs: vec![],
    }));
    for (pool, objects) in [(core(input, 2), false), (object(input), true)] {
        let Pool {
            names,
            preds,
            segments,
            length_weights,
            ..
        } = pool;
        for (i, ((name, predictions), weight)) in
            names.into_iter().zip(preds).zip(length_weights).enumerate()
        {
            let f = if objects { 1 + segments[i] } else { 0 };
            fs[f].programs.push(Candidate {
                name,
                predictions,
                weight,
            });
        }
    }
    fs
}
fn input(t: &Task, limited: bool) -> (Inputs, usize) {
    assert!(t.train.len() >= 2);
    let evidence = if limited { 1 } else { t.train.len() - 1 };
    let mut queries = vec![t.train[evidence].0.clone()];
    let extra = if limited { t.train.len() - 2 } else { 0 };
    if limited {
        queries.extend(t.train[2..].iter().map(|(x, _)| x.clone()));
    }
    queries.extend(t.test.iter().map(|(x, _)| x.clone()));
    (
        Inputs {
            train: t.train[..evidence].to_vec(),
            queries,
        },
        extra,
    )
}
#[derive(Clone)]
struct View {
    name: String,
    initial: usize,
    programs: Vec<Candidate>,
    evidence_p: f64,
    weighted_p: f64,
    answer_h: f64,
}
fn condition(families: Vec<Family>, evidence: &Grid, extra: usize, tests: usize) -> Vec<View> {
    families
        .into_iter()
        .map(|family| {
            let n = family.programs.len();
            let mass: f64 = family.programs.iter().map(|p| p.weight).sum();
            let programs: Vec<_> = family
                .programs
                .into_iter()
                .filter(|p| p.predictions[0].as_ref() == Some(evidence))
                .collect();
            let mut h = 0.;
            if !programs.is_empty() {
                for i in 1 + extra..1 + extra + tests {
                    let mut counts = HashMap::new();
                    for p in &programs {
                        *counts.entry(&p.predictions[i]).or_insert(0usize) += 1;
                    }
                    h += entropy(
                        counts
                            .into_values()
                            .map(|c| c as f64 / programs.len() as f64),
                    ) / tests as f64;
                }
            }
            let evidence_p = if n > 0 {
                programs.len() as f64 / n as f64
            } else {
                0.
            };
            let weighted_p = if mass > 0. {
                programs.iter().map(|p| p.weight).sum::<f64>() / mass
            } else {
                0.
            };
            View {
                name: family.name,
                initial: n,
                programs,
                evidence_p,
                weighted_p,
                answer_h: if n > 0 { h.max(0.) } else { f64::NAN },
            }
        })
        .collect()
}
// All selection inputs are predictions and discovery/evidence statistics.
// Scoring labels are intentionally absent from this API.
fn choices(views: &[View]) -> [Option<usize>; 4] {
    let mut fixed = None;
    let mut evidence = None;
    let mut minimum = None;
    let mut weighted = None;
    for (i, v) in views.iter().enumerate() {
        if v.programs.is_empty() {
            continue;
        }
        fixed.get_or_insert(i);
        if evidence.is_none_or(|j: usize| v.evidence_p > views[j].evidence_p + 1e-12) {
            evidence = Some(i);
        }
        if minimum.is_none_or(|j: usize| v.answer_h < views[j].answer_h - 1e-12) {
            minimum = Some(i);
        }
        if weighted.is_none_or(|j: usize| v.weighted_p > views[j].weighted_p + 1e-12) {
            weighted = Some(i);
        }
    }
    [fixed, evidence, minimum, weighted]
}
fn correct(preds: &[Option<Grid>], ys: &[Grid]) -> bool {
    preds.len() == ys.len() && preds.iter().zip(ys).all(|(p, y)| p.as_ref() == Some(y))
}
#[derive(Clone)]
struct FRow {
    name: String,
    initial: usize,
    remaining: usize,
    p: f64,
    h: f64,
    representative: String,
    predictions: Vec<Option<Grid>>,
    correct: bool,
    pool_oracle: bool,
}
#[derive(Clone, Default)]
struct Outcome {
    correct: bool,
    grid_correct: usize,
    predicted: usize,
    extra_correct: usize,
    extra_predicted: usize,
}
struct Row {
    id: String,
    limited: bool,
    extra: usize,
    tests: usize,
    initial_families: usize,
    final_families: usize,
    competitive: bool,
    family_oracle: bool,
    pool_oracle: bool,
    harm: bool,
    h_before: f64,
    h_after: f64,
    choices: [Option<usize>; 4],
    outcomes: [Outcome; 3],
    families: Vec<FRow>,
}
fn score(id: &str, limited: bool, views: Vec<View>, ys: &[Grid], extra_ys: &[Grid]) -> Row {
    let c = choices(&views);
    let extra = extra_ys.len();
    let initial = views.iter().filter(|v| v.initial > 0).count();
    let final_n = views.iter().filter(|v| !v.programs.is_empty()).count();
    let h_before = if initial > 0 {
        (initial as f64).log2()
    } else {
        f64::NAN
    };
    let mass: f64 = views.iter().map(|v| v.evidence_p).sum();
    let h_after = if mass > 0. {
        entropy(views.iter().map(|v| v.evidence_p / mass))
    } else {
        f64::NAN
    };
    let mut outcomes: [Outcome; 3] = std::array::from_fn(|_| Outcome::default());
    for (i, selected) in c[..3].iter().enumerate() {
        if let Some(j) = selected {
            let p = &views[*j].programs[0].predictions;
            outcomes[i] = Outcome {
                correct: correct(&p[1 + extra..], ys),
                grid_correct: p[1 + extra..]
                    .iter()
                    .zip(ys)
                    .filter(|(p, y)| p.as_ref() == Some(*y))
                    .count(),
                predicted: p[1 + extra..].iter().filter(|g| g.is_some()).count(),
                extra_correct: p[1..1 + extra]
                    .iter()
                    .zip(extra_ys)
                    .filter(|(p, y)| p.as_ref() == Some(*y))
                    .count(),
                extra_predicted: p[1..1 + extra].iter().filter(|g| g.is_some()).count(),
            };
        }
    }
    let distinct = views
        .iter()
        .filter_map(|v| v.programs.first())
        .map(|p| &p.predictions[1 + extra..])
        .collect::<HashSet<_>>()
        .len();
    let families: Vec<_> = views
        .into_iter()
        .map(|v| {
            let first = v.programs.first();
            FRow {
                name: v.name,
                initial: v.initial,
                remaining: v.programs.len(),
                p: v.evidence_p,
                h: v.answer_h,
                representative: first.map_or("—".into(), |p| p.name.clone()),
                predictions: first.map_or(vec![], |p| p.predictions[1 + extra..].to_vec()),
                correct: first.is_some_and(|p| correct(&p.predictions[1 + extra..], ys)),
                pool_oracle: v
                    .programs
                    .iter()
                    .any(|p| correct(&p.predictions[1 + extra..], ys)),
            }
        })
        .collect();
    Row {
        id: id.into(),
        limited,
        extra,
        tests: ys.len(),
        initial_families: initial,
        final_families: final_n,
        competitive: final_n >= 2 && distinct >= 2,
        family_oracle: families.iter().any(|f| f.correct),
        pool_oracle: families.iter().any(|f| f.pool_oracle),
        harm: outcomes[0].correct && families.iter().any(|f| f.remaining > 0 && !f.correct),
        h_before,
        h_after,
        choices: c,
        outcomes,
        families,
    }
}
fn hash(id: &str) -> u64 {
    id.bytes().fold(14695981039346656037, |h, b| {
        (h ^ b as u64).wrapping_mul(1099511628211)
    })
}
fn parallel<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let index = AtomicUsize::new(0);
    let results = Mutex::new(vec![]);
    std::thread::scope(|s| {
        for _ in 0..12 {
            let f = &f;
            let index = &index;
            let results = &results;
            s.spawn(move || loop {
                let i = index.fetch_add(1, Ordering::Relaxed);
                if i >= items.len() {
                    break;
                }
                let r = f(&items[i]);
                results.lock().unwrap().push((i, r));
            });
        }
    });
    let mut rs = results.into_inner().unwrap();
    rs.sort_by_key(|r| r.0);
    rs.into_iter().map(|r| r.1).collect()
}
fn number(v: f64) -> String {
    if !v.is_finite() {
        "n/a".into()
    } else if v.abs() < 0.0005 {
        "0.000".into()
    } else {
        format!("{v:.3}")
    }
}
fn ratio(n: usize, d: usize) -> String {
    if d == 0 {
        "n/a (0/0)".into()
    } else {
        format!("{n}/{d} ({:.2}%)", 100. * n as f64 / d as f64)
    }
}
fn grid_json(g: &Option<Grid>) -> String {
    g.as_ref().map_or("null".into(), |g| {
        format!("{{\"h\":{},\"w\":{},\"cells\":{:?}}}", g.h, g.w, g.cells)
    })
}
fn save(path: &str, rows: &[Row]) {
    let mut w = BufWriter::new(fs::File::create(path).unwrap());
    for r in rows {
        for f in &r.families {
            if f.initial == 0 {
                continue;
            }
            writeln!(w,"{{\"task\":\"{}\",\"condition\":\"{}\",\"family\":\"{}\",\"initial\":{},\"remaining\":{},\"evidence_probability\":{},\"answer_entropy\":{},\"representative\":\"{}\",\"predictions\":[{}],\"representative_correct\":{},\"pool_oracle\":{}}}",r.id,if r.limited{"two-pair"}else{"full"},f.name,f.initial,f.remaining,f.p,if f.remaining==0{"null".into()}else{f.h.to_string()},f.representative,f.predictions.iter().map(grid_json).collect::<Vec<_>>().join(","),f.correct,f.pool_oracle).unwrap();
        }
    }
    w.flush().unwrap();
}
fn family_name(r: &Row, i: usize) -> &str {
    r.choices[i].map_or("abstain", |j| r.families[j].name.as_str())
}
fn report(rows: &[Row], mode: &str, secs: f64) {
    println!("# Competing interpretations run report\n\nPublic training only; {mode}; 12 workers; fixed grid DSL depth 2 plus eight frozen object interpretations. Shared first surviving program within each family. Wall time including both conditions and diagnostics: {secs:.2}s.\n\nFull uses all but the last training pair for discovery and the last as evidence. Two-pair uses only the first two pairs; extra training outputs are deliberately withheld. Program vocabularies are frozen before evidence, unlike full refitting in 003. These are controlled inference experiments, not complete production-solver scores.\n\n## Was improvement possible?\n\n| Condition | Tasks | Any surviving family | ≥2 surviving families | Competitive (disagree) | Fixed-order correct | Family-choice ceiling | Recoverable baseline errors | Correct baseline vulnerable to switching | Full-program-pool ceiling |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    for limited in [false, true] {
        let rs: Vec<_> = rows.iter().filter(|r| r.limited == limited).collect();
        println!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            if limited { "two-pair" } else { "full" },
            rs.len(),
            rs.iter().filter(|r| r.final_families > 0).count(),
            rs.iter().filter(|r| r.final_families >= 2).count(),
            rs.iter().filter(|r| r.competitive).count(),
            rs.iter().filter(|r| r.outcomes[0].correct).count(),
            rs.iter().filter(|r| r.family_oracle).count(),
            rs.iter()
                .filter(|r| r.family_oracle && !r.outcomes[0].correct)
                .count(),
            rs.iter().filter(|r| r.harm).count(),
            rs.iter().filter(|r| r.pool_oracle).count()
        );
    }
    println!("\nCompetitive membership uses predictions only, never scoring labels. The family-choice ceiling chooses among the SAME representatives available to every policy. The full-program ceiling additionally changes programs and is not attainable merely by selecting a family. Recoverable errors are the actual improvement opportunity; fewer than five supports descriptive cases only.\n\n## Accuracy and paired comparisons\n\n| Condition | Cohort | Policy | Tasks | Task exact match | Test-grid exact match | Prediction coverage | Wins vs fixed | Losses vs fixed | Unused training grids correct |\n|---|---|---|---:|---:|---:|---:|---:|---:|---:|");
    for limited in [false, true] {
        for competitive in [false, true] {
            let rs: Vec<_> = rows
                .iter()
                .filter(|r| r.limited == limited && (!competitive || r.competitive))
                .collect();
            for (i, name) in [
                "fixed order",
                "predictive evidence",
                "minimum answer entropy",
            ]
            .iter()
            .enumerate()
            {
                let grid_total = rs.iter().map(|r| r.tests).sum();
                let extra_total = rs.iter().map(|r| r.extra).sum();
                println!(
                    "| {} | {} | {name} | {} | {} | {} | {} | {} | {} | {} |",
                    if limited { "two-pair" } else { "full" },
                    if competitive { "competitive" } else { "all" },
                    rs.len(),
                    ratio(
                        rs.iter().filter(|r| r.outcomes[i].correct).count(),
                        rs.len()
                    ),
                    ratio(
                        rs.iter().map(|r| r.outcomes[i].grid_correct).sum(),
                        grid_total
                    ),
                    ratio(rs.iter().map(|r| r.outcomes[i].predicted).sum(), grid_total),
                    rs.iter()
                        .filter(|r| !r.outcomes[0].correct && r.outcomes[i].correct)
                        .count(),
                    rs.iter()
                        .filter(|r| r.outcomes[0].correct && !r.outcomes[i].correct)
                        .count(),
                    ratio(
                        rs.iter().map(|r| r.outcomes[i].extra_correct).sum(),
                        extra_total
                    )
                );
            }
        }
    }
    println!("\nEvidence selects the largest surviving/discovery program ratio. Minimum answer entropy selects the lowest mean per-test-input entropy under uniform surviving syntax. Fixed-order tie breaking (tolerance 1e-12) applies throughout. Neither sees scoring outputs. Undefined is a prediction category for entropy, but never correct; empty families are unselectable.\n\n## Representation uncertainty\n\n| Condition | Tasks retaining any family | Mean initial viable families | Mean final viable families | Mean initial family H | Mean posterior family H | Evidence-choice changes with grid length prior | Unused training prediction coverage (fixed/evidence/min-H) |\n|---|---:|---:|---:|---:|---:|---:|---:|");
    for limited in [false, true] {
        let all: Vec<_> = rows.iter().filter(|r| r.limited == limited).collect();
        let rs: Vec<_> = all.iter().filter(|r| r.final_families > 0).collect();
        let n = rs.len();
        let mean = |f: fn(&Row) -> f64| rs.iter().map(|r| f(r)).sum::<f64>() / n as f64;
        let extra = all.iter().map(|r| r.extra).sum();
        println!(
            "| {} | {n} | {} | {} | {} | {} | {} | {} / {} / {} |",
            if limited { "two-pair" } else { "full" },
            number(mean(|r| r.initial_families as f64)),
            number(mean(|r| r.final_families as f64)),
            number(mean(|r| r.h_before)),
            number(mean(|r| r.h_after)),
            all.iter().filter(|r| r.choices[1] != r.choices[3]).count(),
            ratio(
                all.iter().map(|r| r.outcomes[0].extra_predicted).sum(),
                extra
            ),
            ratio(
                all.iter().map(|r| r.outcomes[1].extra_predicted).sum(),
                extra
            ),
            ratio(
                all.iter().map(|r| r.outcomes[2].extra_predicted).sum(),
                extra
            )
        );
    }
    println!("\nInitial family prior is uniform over discovery-viable families, conditional on discovery; posterior mass is proportional to evidence probability. Entropies are bits over named families, not semantic equivalence classes. Some object labels describe identical partitions on these inputs. A single observed answer can increase or decrease entropy. Grid-prior sensitivity uses q(length) ∝ 2^-length / K^length per program; object priors remain uniform.\n\n## Competing cases and missed within-family solutions\n\nIncludes every competitive task and every task with a pool-correct program but no correct family representative.\n\n| Condition | Task | Surviving families | Fixed / evidence / min-H family | Correct fixed/evidence/min-H | Family / pool oracle | Initial → posterior family H |\n|---|---|---:|---|---|---|---:|");
    for r in rows {
        if r.competitive || (r.pool_oracle && !r.family_oracle) {
            println!(
                "| {} | {} | {} | {} / {} / {} | {}/{}/{} | {}/{} | {}→{} |",
                if r.limited { "two-pair" } else { "full" },
                r.id,
                r.final_families,
                family_name(r, 0),
                family_name(r, 1),
                family_name(r, 2),
                r.outcomes[0].correct,
                r.outcomes[1].correct,
                r.outcomes[2].correct,
                r.family_oracle,
                r.pool_oracle,
                number(r.h_before),
                number(r.h_after)
            );
        }
    }
    println!("\n## Recoverable errors and harmful switches\n\nFor each recoverable baseline error or actual harmful switch, list ALL surviving representatives. Correctness is post-hoc scoring, never a selection input.\n\n| Condition | Task | Family | Discovery → surviving programs | Evidence probability | Answer entropy | Representative correct | Program |\n|---|---|---|---:|---:|---:|---|---|");
    for r in rows {
        if (r.family_oracle && !r.outcomes[0].correct)
            || (r.outcomes[0].correct && r.outcomes[1..].iter().any(|o| !o.correct))
        {
            for f in &r.families {
                if f.remaining > 0 {
                    println!(
                        "| {} | {} | {} | {}→{} | {} | {} | {} | `{}` |",
                        if r.limited { "two-pair" } else { "full" },
                        r.id,
                        f.name,
                        f.initial,
                        f.remaining,
                        number(f.p),
                        number(f.h),
                        f.correct,
                        f.representative
                    );
                }
            }
        }
    }
    println!("\nCompact family diagnostics and representative predictions for all tasks are saved in families.jsonl. No new DSL features or policies were added after observing results.");
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let mode = &args[1];
    let dest = &args[2];
    assert!(mode == "pilot" || mode == "full");
    let start = Instant::now();
    let mut paths: Vec<_> = fs::read_dir("../data/training")
        .unwrap()
        .map(|p| p.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    assert_eq!(paths.len(), 400);
    paths.sort_by_key(|p| (hash(p.file_stem().unwrap().to_str().unwrap()), p.clone()));
    if mode == "pilot" {
        paths.truncate(24);
    }
    fs::create_dir_all(dest).unwrap();
    fs::write(
        format!("{dest}/selection.txt"),
        paths
            .iter()
            .map(|p| format!("{}\n", p.file_stem().unwrap().to_str().unwrap()))
            .collect::<String>(),
    )
    .unwrap();
    let batches = parallel(&paths, |path| {
        let id = path.file_stem().unwrap().to_str().unwrap();
        let t = task::load(path.to_str().unwrap(), id);
        let ys: Vec<_> = t.test.iter().map(|(_, y)| y.clone()).collect();
        let mut out = vec![];
        for limited in [false, true] {
            let (input, extra) = input(&t, limited);
            let ev = if limited { 1 } else { t.train.len() - 1 };
            let fs = families(&input);
            let views = condition(fs, &t.train[ev].1, extra, ys.len());
            let extras: Vec<_> = if limited {
                t.train[2..].iter().map(|(_, y)| y.clone()).collect()
            } else {
                vec![]
            };
            let r = score(id, limited, views, &ys, &extras);
            eprintln!(
                "{id} {}: {} surviving, competitive {}, recoverable {}",
                if limited { "two-pair" } else { "full" },
                r.final_families,
                r.competitive,
                r.family_oracle && !r.outcomes[0].correct
            );
            out.push(r);
        }
        out
    });
    let rows: Vec<_> = batches.into_iter().flatten().collect();
    save(&format!("{dest}/families.jsonl"), &rows);
    report(&rows, mode, start.elapsed().as_secs_f64());
}
#[cfg(test)]
mod tests {
    use super::*;
    fn g(c: u8) -> Grid {
        Grid::of_rows(&[vec![c]])
    }
    fn candidate(ev: u8, test: u8) -> Candidate {
        Candidate {
            name: format!("{ev}-{test}"),
            predictions: vec![Some(g(ev)), Some(g(test))],
            weight: 1.,
        }
    }
    #[test]
    fn evidence_can_recover_a_wrong_fixed_choice() {
        let fs = vec![
            Family {
                name: "first".into(),
                programs: vec![candidate(1, 2), candidate(3, 2)],
            },
            Family {
                name: "second".into(),
                programs: vec![candidate(1, 1)],
            },
        ];
        let vs = condition(fs, &g(1), 0, 1);
        assert_eq!(vs[0].programs.len(), 1);
        let r = score("test", false, vs, &[g(1)], &[]);
        assert!(r.competitive && r.family_oracle);
        assert!(!r.outcomes[0].correct);
        assert!(r.outcomes[1].correct);
        assert_eq!(r.choices[2], Some(0));
    }
    #[test]
    fn full_pool_ceiling_is_not_family_choice_ceiling() {
        let fs = vec![Family {
            name: "only".into(),
            programs: vec![candidate(1, 2), candidate(1, 1)],
        }];
        let r = score("test", false, condition(fs, &g(1), 0, 1), &[g(1)], &[]);
        assert!(r.pool_oracle);
        assert!(!r.family_oracle);
        assert!(!r.competitive);
    }
    #[test]
    fn certainty_can_harm() {
        let fs = vec![
            Family {
                name: "first".into(),
                programs: vec![candidate(1, 1), candidate(1, 2)],
            },
            Family {
                name: "second".into(),
                programs: vec![candidate(1, 2)],
            },
        ];
        let r = score("test", false, condition(fs, &g(1), 0, 1), &[g(1)], &[]);
        assert!(r.outcomes[0].correct && r.harm);
        assert!(!r.outcomes[2].correct);
    }
    #[test]
    fn undefined_is_not_a_correct_answer() {
        let mut c = candidate(1, 1);
        c.predictions[1] = None;
        let vs = condition(
            vec![Family {
                name: "undefined".into(),
                programs: vec![c],
            }],
            &g(1),
            0,
            1,
        );
        assert_eq!(vs[0].answer_h, 0.);
        let r = score("test", false, vs, &[g(1)], &[]);
        assert!(!r.family_oracle);
    }
    #[test]
    fn scoring_outputs_do_not_change_choices() {
        let fs = || {
            vec![
                Family {
                    name: "a".into(),
                    programs: vec![candidate(1, 1), candidate(1, 2)],
                },
                Family {
                    name: "b".into(),
                    programs: vec![candidate(1, 2)],
                },
            ]
        };
        let a = score("test", false, condition(fs(), &g(1), 0, 1), &[g(1)], &[]);
        let b = score("test", false, condition(fs(), &g(1), 0, 1), &[g(2)], &[]);
        assert_eq!(a.choices, b.choices);
    }
    #[test]
    fn withheld_outputs_do_not_enter_vocabulary() {
        let mut t = Task {
            id: "test".into(),
            train: vec![(g(1), g(1)), (g(1), g(8)), (g(1), g(9))],
            test: vec![(g(1), g(7))],
        };
        let (a, _) = input(&t, true);
        assert_eq!(a.colours(), vec![1]);
        t.train[1].1 = g(6);
        t.train[2].1 = g(5);
        t.test[0].1 = g(4);
        let (b, _) = input(&t, true);
        assert_eq!(a.colours(), b.colours());
    }
}
