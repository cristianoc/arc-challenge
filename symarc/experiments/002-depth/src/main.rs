use std::{
    collections::HashMap,
    fs,
    io::{BufWriter, Write},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
    time::Instant,
};
use symarc::{
    dsl::{self, Prim},
    grid::Grid,
    task, Task,
};

fn predict(p: &[Prim], x: &Grid) -> Option<Grid> {
    p.iter()
        .try_fold(x.clone(), |g, &pr| dsl::eval_bounded(pr, &g))
}
fn predictions(ps: &[Vec<Prim>], xs: &[Grid]) -> Vec<Vec<Option<Grid>>> {
    ps.iter()
        .map(|p| xs.iter().map(|x| predict(p, x)).collect())
        .collect()
}
fn hash(id: &str) -> u64 {
    id.bytes().fold(14695981039346656037, |h, b| {
        (h ^ b as u64).wrapping_mul(1099511628211)
    })
}
fn entropy(counts: impl Iterator<Item = usize>, n: usize) -> f64 {
    counts
        .map(|c| {
            let p = c as f64 / n as f64;
            -p * p.log2()
        })
        .sum()
}
#[derive(Clone)]
struct Stats {
    n: usize,
    unique: usize,
    hp: f64,
    ha: f64,
    determined: bool,
    correct: bool,
    oracle: bool,
    covered: bool,
    correct_grids: usize,
    undefined: usize,
}
fn stats(preds: &[Vec<Option<Grid>>], ys: &[Grid]) -> Stats {
    let n = preds.len();
    let mut groups = HashMap::new();
    for p in preds {
        *groups.entry(p).or_insert(0usize) += 1;
    }
    let mut ha = 0.;
    if n > 0 {
        for i in 0..ys.len() {
            let mut cs = HashMap::new();
            for p in preds {
                *cs.entry(&p[i]).or_insert(0usize) += 1;
            }
            ha += entropy(cs.into_values(), n) / ys.len() as f64;
        }
    }
    let matches = |p: &Vec<Option<Grid>>| p.iter().zip(ys).all(|(a, b)| a.as_ref() == Some(b));
    Stats {
        n,
        unique: groups.len(),
        hp: if n > 0 { (n as f64).log2() } else { f64::NAN },
        ha: if n > 0 { ha } else { f64::NAN },
        determined: n > 0 && groups.len() == 1 && preds[0].iter().all(Option::is_some),
        correct: preds.first().is_some_and(matches),
        oracle: preds.iter().any(matches),
        covered: preds.first().is_some_and(|p| p.iter().all(Option::is_some)),
        correct_grids: preds.first().map_or(0, |p| {
            p.iter()
                .zip(ys)
                .filter(|(a, b)| a.as_ref() == Some(*b))
                .count()
        }),
        undefined: preds
            .iter()
            .filter(|p| p.iter().any(Option::is_none))
            .count(),
    }
}
fn parallel<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let index = AtomicUsize::new(0);
    let results = Mutex::new(Vec::new());
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
struct Prepared {
    task: Task,
    pool: Vec<Prim>,
    ps: Vec<Vec<Prim>>,
    preds: Vec<Vec<Option<Grid>>>,
    stats: Stats,
}
struct Row {
    id: String,
    cohort: &'static str,
    k: usize,
    before: Stats,
    after: Stats,
    new_predictions: usize,
    secs: f64,
    grids: usize,
}
fn grid_json(g: &Option<Grid>) -> String {
    g.as_ref().map_or("null".into(), |g| {
        format!("{{\"h\":{},\"w\":{},\"cells\":{:?}}}", g.h, g.w, g.cells)
    })
}
fn save(path: &str, pool: &[Prim], ps: &[Vec<Prim>], preds: &[Vec<Option<Grid>>]) {
    let mut w = BufWriter::new(fs::File::create(path).unwrap());
    for (p, outs) in ps.iter().zip(preds) {
        let indices: Vec<_> = p
            .iter()
            .map(|pr| pool.iter().position(|x| x == pr).unwrap())
            .collect();
        writeln!(
            w,
            "{{\"length\":{},\"indices\":{:?},\"program\":\"{}\",\"predictions\":[{}]}}",
            p.len(),
            indices,
            dsl::program_name(p),
            outs.iter().map(grid_json).collect::<Vec<_>>().join(",")
        )
        .unwrap();
    }
    w.flush().unwrap();
}
fn number(v: f64) -> String {
    if v.is_nan() {
        "n/a".into()
    } else {
        format!("{v:.3}")
    }
}
fn summary(rows: &[Row], cohort: &str, depth: usize) {
    let rs: Vec<_> = rows
        .iter()
        .filter(|r| cohort == "all selected" || r.cohort == cohort)
        .collect();
    let ss: Vec<_> = rs
        .iter()
        .map(|r| if depth == 2 { &r.before } else { &r.after })
        .collect();
    let fit = ss.iter().filter(|s| s.n > 0).count();
    let mean = |f: fn(&Stats) -> f64| {
        if fit == 0 {
            f64::NAN
        } else {
            ss.iter().filter(|s| s.n > 0).map(|s| f(s)).sum::<f64>() / fit as f64
        }
    };
    let solved = ss.iter().filter(|s| s.correct).count();
    let oracle = ss.iter().filter(|s| s.oracle).count();
    let grids: usize = rs.iter().map(|r| r.grids).sum();
    println!("| {cohort} | {depth} | {} | {fit} | {solved} ({:.2}%) | {}/{} | {} | {oracle} | {} | {} | {} | {} | {} |",rs.len(),100.*solved as f64/rs.len() as f64,ss.iter().map(|s|s.correct_grids).sum::<usize>(),grids,ss.iter().filter(|s|s.covered).count(),oracle-solved,ss.iter().filter(|s|s.determined).count(),number(mean(|s|s.n as f64)),number(mean(|s|s.hp)),number(mean(|s|s.ha)));
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let mode = &args[1];
    let dest = &args[2];
    assert!(mode == "pilot" || mode == "selected");
    fs::create_dir_all(format!("{dest}/pools")).unwrap();
    let start = Instant::now();
    let mut paths: Vec<_> = fs::read_dir("../data/training")
        .unwrap()
        .map(|p| p.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    paths.sort();
    assert_eq!(paths.len(), 400);
    let prep = parallel(&paths, |path| {
        let id = path.file_stem().unwrap().to_str().unwrap();
        let task = task::load(path.to_str().unwrap(), id);
        let pool = task.candidates_pool().1;
        let ps = dsl::enumerate(&pool, &task.train, 2);
        let xs = task.test.iter().map(|(x, _)| x.clone()).collect::<Vec<_>>();
        let preds = predictions(&ps, &xs);
        let ys = task.test.iter().map(|(_, y)| y.clone()).collect::<Vec<_>>();
        let stats = stats(&preds, &ys);
        Prepared {
            task,
            pool,
            ps,
            preds,
            stats,
        }
    });
    assert_eq!(prep.iter().filter(|p| p.stats.n > 0).count(), 55);
    assert_eq!(prep.iter().filter(|p| p.stats.determined).count(), 32);
    let mut ambiguous: Vec<_> = prep
        .iter()
        .filter(|p| p.stats.n > 0 && !p.stats.determined)
        .collect();
    let mut unfitted: Vec<_> = prep.iter().filter(|p| p.stats.n == 0).collect();
    for c in [&mut ambiguous, &mut unfitted] {
        c.sort_by_key(|p| (hash(&p.task.id), &p.task.id));
    }
    unfitted.truncate(24);
    let mut selection = String::from("task\tcohort\n");
    for (c, ps) in [("ambiguous", &ambiguous), ("unfitted", &unfitted)] {
        for p in ps {
            selection += &format!("{}\t{c}\n", p.task.id);
        }
    }
    fs::write(format!("{dest}/selection.txt"), selection).unwrap();
    if mode == "pilot" {
        ambiguous.truncate(6);
        unfitted.truncate(6);
    }
    let selected: Vec<_> = ambiguous
        .into_iter()
        .map(|p| (p, "ambiguous"))
        .chain(unfitted.into_iter().map(|p| (p, "unfitted")))
        .collect();
    let prep_time = start.elapsed().as_secs_f64();
    eprintln!(
        "Census done in {prep_time:.2}s; depth 3 on {} tasks",
        selected.len()
    );
    let rows = parallel(&selected, |(p, cohort)| {
        let t = Instant::now();
        let ps = dsl::enumerate(&p.pool, &p.task.train, 3);
        assert_eq!(
            ps.iter()
                .filter(|p| p.len() <= 2)
                .cloned()
                .collect::<Vec<_>>(),
            p.ps
        );
        let xs = p
            .task
            .test
            .iter()
            .map(|(x, _)| x.clone())
            .collect::<Vec<_>>();
        let preds = predictions(&ps, &xs);
        let new_predictions = preds
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .filter(|v| !p.preds.contains(v))
            .count();
        save(
            &format!("{dest}/pools/{}.jsonl", p.task.id),
            &p.pool,
            &ps,
            &preds,
        );
        let ys = p
            .task
            .test
            .iter()
            .map(|(_, y)| y.clone())
            .collect::<Vec<_>>();
        let after = stats(&preds, &ys);
        let secs = t.elapsed().as_secs_f64();
        eprintln!(
            "{} {cohort}: {} -> {} programs, {secs:.2}s",
            p.task.id, p.stats.n, after.n
        );
        Row {
            id: p.task.id.clone(),
            cohort,
            k: p.pool.len(),
            before: p.stats.clone(),
            after,
            new_predictions,
            secs,
            grids: ys.len(),
        }
    });
    println!("# Depth sensitivity run report\n\nPublic training only; {mode} cohort; 12 workers; exhaustive depth 2 versus 3, uniform syntax prior. No symmetry or search fallback. Shortest fitting program predicts; original enumeration order breaks ties.\n\nFull depth-2 census: 400 tasks, 55 fitting, 32 determined. Census wall time: {prep_time:.2}s. Total computation and pool-writing wall time: {:.2}s.\n",start.elapsed().as_secs_f64());
    println!("| Cohort | Depth | Tasks | Fitting | Task exact match | Grid exact match | Fully predicted tasks | Oracle tasks | Oracle headroom | Determined | Mean programs | Mean program H | Mean answer H |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    for c in ["ambiguous", "unfitted", "all selected"] {
        for d in [2, 3] {
            summary(&rows, c, d);
        }
    }
    println!("\nMeans condition on fitting tasks; depths may have different denominators. Entropies are bits. Oracle is any single fitting program correct on all test outputs; it is a post-hoc ceiling, not a selection rule. Empty pools count as incorrect/unpredicted. Grid scores use all test grids. This deliberately selected cohort is not a corpus accuracy estimate.\n\n| Task | Cohort | Primitives | Programs d2→d3 | Joint predictions d2→d3 | New joint predictions | Answer H d2→d3 | Selected correct d2→d3 | Oracle d2→d3 | Undefined programs d3 | Depth-3 task seconds |\n|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|");
    for r in &rows {
        println!(
            "| {} | {} | {} | {}→{} | {}→{} | {} | {}→{} | {}→{} | {}→{} | {} | {:.2} |",
            r.id,
            r.cohort,
            r.k,
            r.before.n,
            r.after.n,
            r.before.unique,
            r.after.unique,
            r.new_predictions,
            number(r.before.ha),
            number(r.after.ha),
            r.before.correct,
            r.after.correct,
            r.before.oracle,
            r.after.oracle,
            r.after.undefined,
            r.secs
        );
    }
    println!("\nJoint prediction counts include undefined and identify agreement only on the observed test inputs. Saved pools contain every fitting program at length ≤3 and its predictions; length filters recover depth 2 without rerunning. Per-task seconds overlap across workers and include scoring/writing; process peak RSS is recorded in stderr.txt by the runner.");
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn enumeration_matches_brute_force_in_order() {
        let pool = vec![
            Prim::Rot90,
            Prim::Recolour(1, 2),
            Prim::CropBBox,
            Prim::Scale(2),
        ];
        let x = Grid::of_rows(&[vec![1, 0], vec![0, 0]]);
        for target in [
            x.clone(),
            predict(&[Prim::CropBBox, Prim::Recolour(1, 2)], &x).unwrap(),
        ] {
            for depth in 0..=3 {
                let mut all = vec![vec![]];
                let mut layer = vec![vec![]];
                for _ in 0..depth {
                    layer = layer
                        .iter()
                        .flat_map(|p| {
                            pool.iter().map(move |&pr| {
                                let mut q = p.clone();
                                q.push(pr);
                                q
                            })
                        })
                        .collect();
                    all.extend(layer.clone());
                }
                let expected: Vec<_> = all
                    .into_iter()
                    .filter(|p| predict(p, &x).as_ref() == Some(&target))
                    .collect();
                assert_eq!(
                    dsl::enumerate(&pool, &[(x.clone(), target.clone())], depth),
                    expected
                );
            }
        }
    }
    #[test]
    fn undefined_is_not_determination() {
        let y = Grid::of_rows(&[vec![1]]);
        let s = stats(&[vec![None], vec![None]], &[y.clone()]);
        assert!(!s.determined);
        assert_eq!(s.ha, 0.);
        let s = stats(&[vec![None], vec![Some(y.clone())]], &[y]);
        assert_eq!(s.ha, 1.);
        assert!(s.oracle);
        assert!(!s.correct);
    }
}
