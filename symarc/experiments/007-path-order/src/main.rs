use std::{
    fs,
    io::Write,
    path::PathBuf,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
    time::Instant,
};
use symarc::{grid::Grid, task};
use symarc_exp_003_objects::{core, object, Inputs, Pool};
use symarc_exp_007_path_order::infer;
fn hash(s: &str) -> u64 {
    s.bytes().fold(14695981039346656037, |h, b| {
        (h ^ b as u64).wrapping_mul(1099511628211)
    })
}
fn parallel<T: Sync, R: Send>(xs: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let next = AtomicUsize::new(0);
    let out = Mutex::new(vec![]);
    std::thread::scope(|scope| {
        for _ in 0..12 {
            let f = &f;
            let next = &next;
            let out = &out;
            scope.spawn(move || loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= xs.len() {
                    break;
                }
                let result = f(&xs[i]);
                out.lock().unwrap().push((i, result));
            });
        }
    });
    let mut out = out.into_inner().unwrap();
    out.sort_by_key(|r| r.0);
    out.into_iter().map(|(_, r)| r).collect()
}
#[derive(Clone, Default)]
struct Score {
    fits: usize,
    correct: bool,
    oracle: bool,
    grids: usize,
    defined: usize,
    distinct: usize,
    name: String,
    witness: String,
}
fn score(p: &Pool, ys: &[Grid]) -> Score {
    let correct =
        |preds: &Vec<Option<Grid>>| preds.iter().zip(ys).all(|(x, y)| x.as_ref() == Some(y));
    Score {
        fits: p.preds.len(),
        correct: p.preds.first().is_some_and(correct),
        oracle: p.preds.iter().any(correct),
        grids: p.preds.first().map_or(0, |ps| {
            ps.iter()
                .zip(ys)
                .filter(|(x, y)| x.as_ref() == Some(y))
                .count()
        }),
        defined: p
            .preds
            .first()
            .map_or(0, |ps| ps.iter().filter(|x| x.is_some()).count()),
        distinct: p
            .preds
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len(),
        name: p.names.first().cloned().unwrap_or_default(),
        witness: p
            .preds
            .iter()
            .position(correct)
            .map_or(String::new(), |i| p.names[i].clone()),
    }
}
fn fallback(scores: &[&Score]) -> Score {
    let mut s = scores
        .iter()
        .find(|s| s.fits > 0)
        .map(|s| (*s).clone())
        .unwrap_or_default();
    s.oracle = scores.iter().any(|s| s.oracle);
    s
}
fn q(s: &str) -> String {
    format!(
        "\"{}\"",
        s.replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
    )
}
fn grid(g: &Option<Grid>) -> String {
    g.as_ref().map_or("null".into(), |g| {
        format!("{{\"h\":{},\"w\":{},\"cells\":{:?}}}", g.h, g.w, g.cells)
    })
}
fn save(dest: &str, id: &str, label: &str, p: &Pool) {
    let mut f = fs::File::create(format!("{dest}/pools/{id}-{label}.jsonl")).unwrap();
    for (name, ps) in p.names.iter().zip(&p.preds) {
        writeln!(
            f,
            "{{\"program\":{},\"predictions\":[{}]}}",
            q(name),
            ps.iter().map(grid).collect::<Vec<_>>().join(",")
        )
        .unwrap();
    }
}
struct Row {
    id: String,
    s: Vec<Score>,
    checks: usize,
    candidates: usize,
    seconds: f64,
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let mode = &args[1];
    let dest = &args[2];
    let start = Instant::now();
    assert!(["pilot", "full", "development"].contains(&mode.as_str()));
    fs::create_dir_all(format!("{dest}/pools")).unwrap();
    let mut paths: Vec<PathBuf> = if mode == "development" {
        vec![PathBuf::from(&args[3])]
    } else {
        fs::read_dir("../data/training")
            .unwrap()
            .map(|p| p.unwrap().path())
            .filter(|p| p.extension().is_some_and(|x| x == "json"))
            .collect()
    };
    paths.sort_by_key(|p| (hash(p.file_stem().unwrap().to_str().unwrap()), p.clone()));
    if mode != "development" {
        assert_eq!(paths.len(), 400);
    }
    if mode == "pilot" {
        paths.truncate(24);
    }
    fs::write(
        format!("{dest}/selection.txt"),
        paths
            .iter()
            .map(|p| format!("{}\n", p.file_stem().unwrap().to_str().unwrap()))
            .collect::<String>(),
    )
    .unwrap();
    let rows = parallel(&paths, |path| {
        let st = Instant::now();
        let id = path.file_stem().unwrap().to_str().unwrap();
        let t = task::load(path.to_str().unwrap(), id);
        let input = Inputs::from_task(&t, false);
        let g = core(&input, 2);
        let o = object(&input);
        let h = infer(&input, false);
        let p = infer(&input, true);
        // All candidate generation and selection complete before reading labels for scoring.
        let ys: Vec<_> = t.test.iter().map(|(_, y)| y.clone()).collect();
        let mut s = vec![
            score(&g, &ys),
            score(&o, &ys),
            score(&h, &ys),
            score(&p, &ys),
        ];
        s.push(fallback(&[&s[0], &s[1], &s[2]]));
        s.push(fallback(&[&s[0], &s[1], &s[2], &s[3]]));
        s.push(fallback(&[&s[3], &s[0], &s[1], &s[2]]));
        for (label, pool) in [
            ("grid", &g),
            ("objects", &o),
            ("histogram", &h),
            ("path", &p),
        ] {
            save(dest, id, label, pool);
        }
        Row {
            id: id.into(),
            s,
            checks: p.checks + h.checks,
            candidates: p.candidates + h.candidates,
            seconds: st.elapsed().as_secs_f64(),
        }
    });
    if mode == "full" {
        assert_eq!(rows.iter().filter(|r| r.s[0].fits > 0).count(), 55);
        assert_eq!(rows.iter().filter(|r| r.s[0].correct).count(), 53);
        assert_eq!(rows.iter().filter(|r| r.s[0].oracle).count(), 54);
        assert_eq!(rows.iter().filter(|r| r.s[1].fits > 0).count(), 17);
        assert_eq!(rows.iter().filter(|r| r.s[1].correct).count(), 16);
    }
    let mut audit: Vec<_> = rows
        .iter()
        .filter(|r| r.s[3].fits > 0 && r.s[..3].iter().all(|s| s.fits == 0))
        .map(|r| r.id.clone())
        .collect();
    audit.sort_by_key(|id| (hash(id), id.clone()));
    audit.truncate(if mode == "full" { 16 } else { 0 });
    fs::write(format!("{dest}/audit-selection.txt"), audit.join("\n")).unwrap();
    let audits = parallel(&audit, |id| {
        let t = task::load(&format!("../data/training/{id}.json"), id);
        let p = core(&Inputs::from_task(&t, false), 3);
        save(dest, id, "grid-d3", &p);
        let ys = t.test.iter().map(|(_, y)| y.clone()).collect::<Vec<_>>();
        (id.clone(), score(&p, &ys))
    });
    println!("# 007 path-order: {mode}\n\n{} tasks; 12 workers; {:.3}s including depth-3 audit.\n\nTraining-only enumeration; test labels used only for scoring. Public development data; known ARC-AGI-2 witness is not a fresh holdout.\n",rows.len(),start.elapsed().as_secs_f64());
    println!("| Arm | Fitting tasks | Correct tasks | Correct test grids | Defined test grids | Oracle tasks |\n|---|---:|---:|---:|---:|---:|");
    for (i, name) in [
        "grid d2",
        "003 objects",
        "histogram",
        "path",
        "control: grid/objects/histogram",
        "treatment: control/path",
        "secondary: path/control",
    ]
    .iter()
    .enumerate()
    {
        println!(
            "| {name} | {} | {} | {} | {} | {} |",
            rows.iter().filter(|r| r.s[i].fits > 0).count(),
            rows.iter().filter(|r| r.s[i].correct).count(),
            rows.iter().map(|r| r.s[i].grids).sum::<usize>(),
            rows.iter().map(|r| r.s[i].defined).sum::<usize>(),
            rows.iter().filter(|r| r.s[i].oracle).count()
        );
    }
    println!("\nTreatment wins/losses vs control: {}/{}. Path-first wins/losses: {}/{}. New oracle tasks: {}.\n",rows.iter().filter(|r|!r.s[4].correct&&r.s[5].correct).count(),rows.iter().filter(|r|r.s[4].correct&&!r.s[5].correct).count(),rows.iter().filter(|r|!r.s[4].correct&&r.s[6].correct).count(),rows.iter().filter(|r|r.s[4].correct&&!r.s[6].correct).count(),rows.iter().filter(|r|!r.s[4].oracle&&r.s[3].oracle).count());
    println!("## Path-fitting tasks\n\n| Task | Histogram fits | Path fits | Distinct path predictions | Path selected correct | Path oracle | Control oracle | First path | Correct path witness |\n|---|---:|---:|---:|---|---|---|---|---|");
    for r in &rows {
        if r.s[3].fits > 0 {
            println!(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} |",
                r.id,
                r.s[2].fits,
                r.s[3].fits,
                r.s[3].distinct,
                r.s[3].correct,
                r.s[3].oracle,
                r.s[4].oracle,
                r.s[3].name,
                r.s[3].witness
            );
        }
    }
    println!(
        "\n## Depth-3 audit\n\n| Task | d3 fits | d3 oracle | Path oracle |\n|---|---:|---|---|"
    );
    for (id, s) in &audits {
        println!(
            "| {id} | {} | {} | {} |",
            s.fits,
            s.oracle,
            rows.iter().find(|r| r.id == *id).unwrap().s[3].oracle
        );
    }
    println!("\nHistogram+path candidate ASTs: {}; actual training example checks: {}. Summed task seconds (overlapping workers): {:.3}. Counts are not equal-cost primitive-operation budgets.\n",rows.iter().map(|r|r.candidates).sum::<usize>(),rows.iter().map(|r|r.checks).sum::<usize>(),rows.iter().map(|r|r.seconds).sum::<f64>());
    println!("No post-run grammar tuning or test-label-driven selection. A gain means coverage within these bounded classes, not automatic discovery of graph concepts or an all-depth separation.");
    let mut f = fs::File::create(format!("{dest}/tasks.jsonl")).unwrap();
    for r in &rows {
        writeln!(f,"{{\"id\":{},\"arms\":[{}]}}",q(&r.id),r.s.iter().map(|s|format!("{{\"fits\":{},\"correct\":{},\"oracle\":{},\"grids\":{},\"defined\":{},\"distinct\":{},\"first\":{},\"witness\":{}}}",s.fits,s.correct,s.oracle,s.grids,s.defined,s.distinct,q(&s.name),q(&s.witness))).collect::<Vec<_>>().join(",")).unwrap();
    }
}
