mod controlled;
mod relations;
use std::{
    fs,
    io::{BufWriter, Write},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
    time::Instant,
};
use symarc::{grid::Grid, task};
use symarc_exp_003_objects::{core, object, Inputs, Pool};
#[derive(Clone)]
struct Score {
    fit: usize,
    correct: bool,
    oracle: bool,
    predicted: usize,
    grid_correct: usize,
    grids: usize,
}
fn score(p: &Pool, ys: &[Grid]) -> Score {
    let good = |v: &Vec<Option<Grid>>| {
        v.len() == ys.len() && v.iter().zip(ys).all(|(x, y)| x.as_ref() == Some(y))
    };
    Score {
        fit: p.preds.len(),
        correct: p.preds.first().is_some_and(good),
        oracle: p.preds.iter().any(good),
        predicted: p
            .preds
            .first()
            .map_or(0, |v| v.iter().filter(|x| x.is_some()).count()),
        grid_correct: p.preds.first().map_or(0, |v| {
            v.iter()
                .zip(ys)
                .filter(|(x, y)| x.as_ref() == Some(*y))
                .count()
        }),
        grids: ys.len(),
    }
}
struct Row {
    id: String,
    baseline: Score,
    old_oracle: bool,
    relation: Score,
    name: String,
    witness: String,
    candidates: usize,
    checks: usize,
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
fn grid_json(g: &Grid) -> String {
    format!("{{\"h\":{},\"w\":{},\"cells\":{:?}}}", g.h, g.w, g.cells)
}
fn maybe_json(g: &Option<Grid>) -> String {
    g.as_ref().map_or("null".into(), grid_json)
}
fn save(path: &str, p: &Pool) {
    let mut w = BufWriter::new(fs::File::create(path).unwrap());
    for (name, preds) in p.names.iter().zip(&p.preds) {
        writeln!(
            w,
            "{{\"program\":\"{}\",\"predictions\":[{}]}}",
            name,
            preds.iter().map(maybe_json).collect::<Vec<_>>().join(",")
        )
        .unwrap();
    }
    w.flush().unwrap();
}
fn save_cases(path: &str, cs: &[controlled::Case]) {
    let mut w = BufWriter::new(fs::File::create(path).unwrap());
    for c in cs {
        writeln!(w,"{{\"seed\":{},\"object_truth\":{},\"initial_input\":{},\"initial_output\":{},\"menu_inputs\":[{}],\"menu_outputs\":[{}],\"test_input\":{},\"test_output\":{},\"ig_choice\":{}}}",c.seed,c.object_truth,grid_json(&c.initial),grid_json(&controlled::teacher(c,&c.initial,0)),c.menu.iter().map(grid_json).collect::<Vec<_>>().join(","),c.menu.iter().enumerate().map(|(i,g)|grid_json(&controlled::teacher(c,g,i+1))).collect::<Vec<_>>().join(","),grid_json(&c.test),grid_json(&controlled::teacher(c,&c.test,5)),controlled::choose(c)).unwrap();
    }
    w.flush().unwrap();
}
fn ratio(n: usize, d: usize) -> String {
    if d == 0 {
        "n/a".into()
    } else {
        format!("{n}/{d} ({:.2}%)", 100. * n as f64 / d as f64)
    }
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
    fs::create_dir_all(format!("{dest}/pools")).unwrap();
    fs::write(
        format!("{dest}/selection.txt"),
        paths
            .iter()
            .map(|p| format!("{}\n", p.file_stem().unwrap().to_str().unwrap()))
            .collect::<String>(),
    )
    .unwrap();
    let rows = parallel(&paths, |path| {
        let id = path.file_stem().unwrap().to_str().unwrap();
        let t = task::load(path.to_str().unwrap(), id);
        let input = Inputs::from_task(&t, false);
        let g = core(&input, 2);
        let o = object(&input);
        let r = relations::infer(&input);
        let ys: Vec<_> = t.test.iter().map(|(_, y)| y.clone()).collect();
        let gs = score(&g, &ys);
        let os = score(&o, &ys);
        let relation = score(&r, &ys);
        let old_oracle = gs.oracle || os.oracle;
        let baseline = if gs.fit > 0 { gs } else { os };
        let witness = r
            .preds
            .iter()
            .position(|p| p.iter().zip(&ys).all(|(x, y)| x.as_ref() == Some(y)))
            .map_or("—".into(), |i| r.names[i].clone());
        let name = r.names.first().cloned().unwrap_or("—".into());
        save(&format!("{dest}/pools/{id}-relations.jsonl"), &r);
        eprintln!(
            "{id}: baseline fit {}, relation fit {}, new oracle {}",
            baseline.fit,
            relation.fit,
            relation.oracle && !old_oracle
        );
        Row {
            id: id.into(),
            baseline,
            old_oracle,
            relation,
            name,
            witness,
            candidates: r.candidates,
            checks: r.checks,
        }
    });
    let primary = start.elapsed().as_secs_f64();
    if mode == "full" {
        assert_eq!(rows.iter().filter(|r| r.baseline.fit > 0).count(), 63);
        assert_eq!(rows.iter().filter(|r| r.baseline.correct).count(), 60);
        assert_eq!(rows.iter().filter(|r| r.old_oracle).count(), 61);
    }
    let mut audit_ids: Vec<_> = rows
        .iter()
        .filter(|r| r.baseline.fit == 0 && r.relation.fit > 0)
        .map(|r| r.id.clone())
        .collect();
    audit_ids.sort_by_key(|id| (hash(id), id.clone()));
    audit_ids.truncate(if mode == "full" { 16 } else { 0 });
    fs::write(
        format!("{dest}/audit-selection.txt"),
        audit_ids.join("\n") + "\n",
    )
    .unwrap();
    let audit = parallel(&audit_ids, |id| {
        let t = task::load(&format!("../data/training/{id}.json"), id);
        let st = Instant::now();
        let p = core(&Inputs::from_task(&t, false), 3);
        let ys: Vec<_> = t.test.iter().map(|(_, y)| y.clone()).collect();
        let s = score(&p, &ys);
        save(&format!("{dest}/pools/{id}-grid-d3.jsonl"), &p);
        eprintln!(
            "Audit {id}: d3 fit {}, oracle {}, {:.2}s",
            s.fit,
            s.oracle,
            st.elapsed().as_secs_f64()
        );
        (id.clone(), s, st.elapsed().as_secs_f64())
    });
    let cases = controlled::checked_cases();
    save_cases(&format!("{dest}/controlled-cases.jsonl"), &cases);
    println!("# Relational objects and informative evidence\n\nPublic training tasks only; {mode}; {} real tasks, 64 controlled cases; 12 workers. Core and 003 object language unchanged. Primary real-task search and saving: {primary:.2}s; total including depth-3 audit and constructed cases: {:.2}s. Resource measurements are in stderr.txt.\n\n## Real-task coverage\n\n| Arm | Training-fit tasks | Task exact match | Test-grid exact match | Prediction coverage (grids) | Accuracy given fit | Oracle tasks |\n|---|---:|---:|---:|---:|---:|---:|",rows.len(),start.elapsed().as_secs_f64());
    for arm in 0..3 {
        let ss: Vec<_> = rows
            .iter()
            .map(|r| match arm {
                0 => &r.baseline,
                1 => &r.relation,
                _ => {
                    if r.baseline.fit > 0 {
                        &r.baseline
                    } else {
                        &r.relation
                    }
                }
            })
            .collect();
        let fit = ss.iter().filter(|s| s.fit > 0).count();
        let solved = ss.iter().filter(|s| s.correct).count();
        let grids = ss.iter().map(|s| s.grids).sum();
        let oracle = if arm == 0 {
            rows.iter().filter(|r| r.old_oracle).count()
        } else if arm == 1 {
            rows.iter().filter(|r| r.relation.oracle).count()
        } else {
            rows.iter()
                .filter(|r| r.old_oracle || r.relation.oracle)
                .count()
        };
        println!(
            "| {} | {} | {} | {} | {} | {} | {} |",
            [
                "grid + 003 objects",
                "relations only",
                "control then relations fallback"
            ][arm],
            ratio(fit, rows.len()),
            ratio(solved, rows.len()),
            ratio(ss.iter().map(|s| s.grid_correct).sum(), grids),
            ratio(ss.iter().map(|s| s.predicted).sum(), grids),
            ratio(solved, fit),
            ratio(oracle, rows.len())
        );
    }
    println!("\nControl prefers a fitting grid program, then a fitting 003 object program. The new fallback uses relations only when neither control family fits. Within a language, first fitting program in the fixed order predicts. Oracle columns count any correct program across the listed arm's pools, including unselected families, and require all test outputs correct. These are enumeration experiments, not the complete production solver with repair/symmetry.\n\nRelations add {} training-fitting tasks and {} oracle-correct tasks beyond the control. Candidate ASTs checked: {}; actual training program/example checks: {}. Candidate counts are syntax, not unique semantics or equivalent operation costs.\n",rows.iter().filter(|r|r.baseline.fit==0&&r.relation.fit>0).count(),rows.iter().filter(|r|!r.old_oracle&&r.relation.oracle).count(),rows.iter().map(|r|r.candidates).sum::<usize>(),rows.iter().map(|r|r.checks).sum::<usize>());
    println!("## Targeted depth-3 audit\n\nSelected by newly fitting training data, before considering test correctness; at most 16 tasks.\n\n| Task | Grid d3 fits | Grid d3 oracle | Relational oracle | Relation selected correct | Audit task seconds |\n|---|---:|---|---|---|---:|");
    for (id, s, secs) in &audit {
        let r = rows.iter().find(|r| &r.id == id).unwrap();
        println!(
            "| {id} | {} | {} | {} | {} | {secs:.2} |",
            s.fit, s.oracle, r.relation.oracle, r.relation.correct
        );
    }
    println!("\n{} audited tasks have a relational correct program absent from both the control and the grid depth-3 pool. This is a targeted audit, not full-corpus depth-3 accuracy.\n",audit.iter().filter(|(id,s,_)|!s.oracle&&rows.iter().find(|r|&r.id==id).unwrap().relation.oracle).count());
    println!("{}", controlled::report(&cases));
    println!("## Real tasks with relational fits\n\nCorrect witnesses are obtained by post-hoc scoring, never used by search or first-program selection.\n\n| Task | Control / relational fits | Control / relational selected correct | Control / relational oracle | Relational first program | Correct relational witness |\n|---|---:|---|---|---|---|");
    for r in &rows {
        if r.relation.fit > 0 {
            println!(
                "| {} | {}/{} | {}/{} | {}/{} | `{}` | `{}` |",
                r.id,
                r.baseline.fit,
                r.relation.fit,
                r.baseline.correct,
                r.relation.correct,
                r.old_oracle,
                r.relation.oracle,
                r.name,
                r.witness
            );
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn oracle_requires_one_program_correct_on_all_queries() {
        let a = Grid::of_rows(&[vec![1]]);
        let b = Grid::of_rows(&[vec![2]]);
        let p = Pool {
            preds: vec![
                vec![Some(a.clone()), Some(a.clone())],
                vec![Some(b.clone()), Some(b.clone())],
            ],
            ..Pool::default()
        };
        assert!(!score(&p, &[a, b]).oracle);
    }
}
