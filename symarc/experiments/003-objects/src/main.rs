mod objects;
use objects::{Action, Program, Render, Scene, Segment, Select};
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
use symarc::{
    dsl,
    grid::{Example, Grid},
    search, task, Task,
};

struct Inputs {
    train: Vec<Example>,
    queries: Vec<Grid>,
}
impl Inputs {
    fn from_task(t: &Task, reserve: bool) -> Self {
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
    fn colours(&self) -> Vec<u8> {
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
struct Pool {
    names: Vec<String>,
    preds: Vec<Vec<Option<Grid>>>,
    segments: Vec<usize>,
    length_weights: Vec<f64>,
    candidates: usize,
    checks: usize,
}
fn core(input: &Inputs, depth: usize) -> Pool {
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
fn object(input: &Inputs) -> Pool {
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
fn distribution(pool: &Pool, query: usize, weighted: bool) -> HashMap<Option<Grid>, f64> {
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
fn entropy(masses: impl Iterator<Item = f64>) -> f64 {
    masses.filter(|&v| v > 0.).map(|v| -v * v.log2()).sum()
}
#[derive(Clone, Default)]
struct Score {
    fit: usize,
    correct: bool,
    oracle: bool,
    predicted: usize,
    grids: usize,
    grid_correct: usize,
    unique: usize,
    answer_h: f64,
    program_h: f64,
    segments: usize,
    segment_h: f64,
}
fn score(pool: &Pool, ys: &[Grid]) -> Score {
    let n = pool.preds.len();
    let matches = |p: &Vec<Option<Grid>>| p.iter().zip(ys).all(|(a, b)| a.as_ref() == Some(b));
    let unique = pool.preds.iter().collect::<HashSet<_>>().len();
    let mut sc = [0usize; 8];
    for &si in &pool.segments {
        sc[si] += 1;
    }
    Score {
        fit: n,
        correct: pool.preds.first().is_some_and(matches),
        oracle: pool.preds.iter().any(matches),
        grids: ys.len(),
        predicted: pool
            .preds
            .first()
            .map_or(0, |p| p.iter().filter(|g| g.is_some()).count()),
        grid_correct: pool.preds.first().map_or(0, |p| {
            p.iter()
                .zip(ys)
                .filter(|(a, b)| a.as_ref() == Some(*b))
                .count()
        }),
        unique,
        answer_h: if n == 0 {
            f64::NAN
        } else {
            (0..ys.len())
                .map(|i| entropy(distribution(pool, i, false).into_values()))
                .sum::<f64>()
                / ys.len() as f64
        },
        program_h: if n == 0 { f64::NAN } else { (n as f64).log2() },
        segments: sc.iter().filter(|&&n| n > 0).count(),
        segment_h: if pool.segments.is_empty() {
            f64::NAN
        } else {
            entropy(sc.iter().map(|&v| v as f64 / n as f64))
        },
    }
}
#[derive(Clone, Default)]
struct Validation {
    available: bool,
    grid_fit: usize,
    obj_fit: usize,
    grid_p: f64,
    obj_p: f64,
    grid_length_p: f64,
    grid_h: f64,
    obj_h: f64,
    grid_brier: f64,
    obj_brier: f64,
    grid_correct: bool,
    obj_correct: bool,
}
fn validation(input: &Inputs, y: &Grid) -> Validation {
    let g = core(input, 2);
    let o = object(input);
    let measure = |pool: &Pool, weighted: bool| {
        let ds = distribution(pool, 0, weighted);
        let p = *ds.get(&Some(y.clone())).unwrap_or(&0.);
        let h = entropy(ds.values().copied());
        let brier = 1. - 2. * p + ds.values().map(|v| v * v).sum::<f64>();
        (p, h, brier)
    };
    let (gp, gh, gb) = measure(&g, false);
    let (op, oh, ob) = measure(&o, false);
    let (glp, _, _) = measure(&g, true);
    Validation {
        available: true,
        grid_fit: g.preds.len(),
        obj_fit: o.preds.len(),
        grid_p: gp,
        obj_p: op,
        grid_length_p: glp,
        grid_h: gh,
        obj_h: oh,
        grid_brier: gb,
        obj_brier: ob,
        grid_correct: g.preds.first().is_some_and(|p| p[0].as_ref() == Some(y)),
        obj_correct: o.preds.first().is_some_and(|p| p[0].as_ref() == Some(y)),
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
    std::thread::scope(|scope| {
        for _ in 0..12 {
            let f = &f;
            let index = &index;
            let results = &results;
            scope.spawn(move || loop {
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
fn json_grid(g: &Option<Grid>) -> String {
    g.as_ref().map_or("null".into(), |g| {
        format!("{{\"h\":{},\"w\":{},\"cells\":{:?}}}", g.h, g.w, g.cells)
    })
}
fn save(path: &str, pool: &Pool) {
    let mut w = BufWriter::new(fs::File::create(path).unwrap());
    for (i, (name, preds)) in pool.names.iter().zip(&pool.preds).enumerate() {
        writeln!(
            w,
            "{{\"program\":\"{}\",\"segment\":{},\"length_weight\":{},\"predictions\":[{}]}}",
            name,
            pool.segments
                .get(i)
                .map_or("null".into(), |s| s.to_string()),
            pool.length_weights[i],
            preds.iter().map(json_grid).collect::<Vec<_>>().join(",")
        )
        .unwrap();
    }
    w.flush().unwrap();
}
struct Row {
    id: String,
    grid: Score,
    obj: Score,
    validation: Validation,
    grid_name: String,
    obj_name: String,
    obj_witness: String,
    grid_candidates: usize,
    obj_candidates: usize,
    obj_checks: usize,
    grid_seconds: f64,
    obj_seconds: f64,
}
fn ratio(n: usize, d: usize) -> String {
    if d == 0 {
        "n/a (0/0)".into()
    } else {
        format!("{n}/{d} ({:.2}%)", 100. * n as f64 / d as f64)
    }
}
fn num(v: f64) -> String {
    if v.is_nan() {
        "n/a".into()
    } else if v.abs() < 0.0005 {
        "0.000".into()
    } else {
        format!("{v:.3}")
    }
}
fn arm<'a>(r: &'a Row, name: &str) -> &'a Score {
    match name {
        "grid" => &r.grid,
        "objects" => &r.obj,
        "fallback" => {
            if r.grid.fit > 0 {
                &r.grid
            } else {
                &r.obj
            }
        }
        "validation gate" => {
            if r.validation.available && r.validation.obj_p > r.validation.grid_p + 1e-12 {
                &r.obj
            } else {
                &r.grid
            }
        }
        _ => unreachable!(),
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
    paths.sort_by_key(|p| (hash(p.file_stem().unwrap().to_str().unwrap()), p.clone()));
    assert_eq!(paths.len(), 400);
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
        let gt = Instant::now();
        let g = core(&input, 2);
        let gs = gt.elapsed().as_secs_f64();
        let ot = Instant::now();
        let o = object(&input);
        let os = ot.elapsed().as_secs_f64();
        let v = if t.train.len() >= 2 {
            validation(&Inputs::from_task(&t, true), &t.train.last().unwrap().1)
        } else {
            Validation::default()
        };
        let ys: Vec<_> = t.test.iter().map(|(_, y)| y.clone()).collect();
        let grid = score(&g, &ys);
        let obj = score(&o, &ys);
        let obj_witness = o
            .preds
            .iter()
            .position(|p| p.iter().zip(&ys).all(|(x, y)| x.as_ref() == Some(y)))
            .map_or("—".into(), |i| o.names[i].clone());
        save(&format!("{dest}/pools/{id}-grid.jsonl"), &g);
        save(&format!("{dest}/pools/{id}-objects.jsonl"), &o);
        eprintln!(
            "{id}: grid fits {} object fits {} oracle {}/{} validation p {:.3}/{:.3}",
            grid.fit, obj.fit, grid.oracle, obj.oracle, v.grid_p, v.obj_p
        );
        Row {
            id: id.into(),
            grid,
            obj,
            validation: v,
            grid_name: g.names.first().cloned().unwrap_or("—".into()),
            obj_name: o.names.first().cloned().unwrap_or("—".into()),
            obj_witness,
            grid_candidates: g.candidates,
            obj_candidates: o.candidates,
            obj_checks: o.checks,
            grid_seconds: gs,
            obj_seconds: os,
        }
    });
    let primary_seconds = start.elapsed().as_secs_f64();
    if mode == "full" {
        assert_eq!(rows.iter().filter(|r| r.grid.fit > 0).count(), 55);
        assert_eq!(rows.iter().filter(|r| r.grid.correct).count(), 53);
        assert_eq!(rows.iter().filter(|r| r.grid.oracle).count(), 54);
    }
    let mut audit_ids: Vec<_> = rows
        .iter()
        .filter(|r| r.grid.fit == 0 && r.obj.fit > 0)
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
        save(&format!("{dest}/pools/{id}-grid-d3.jsonl"), &p);
        let ys = t.test.iter().map(|(_, y)| y.clone()).collect::<Vec<_>>();
        let s = score(&p, &ys);
        eprintln!(
            "Audit {id}: d3 fit {}, oracle {}, {:.2}s",
            s.fit,
            s.oracle,
            st.elapsed().as_secs_f64()
        );
        (id.clone(), s, st.elapsed().as_secs_f64())
    });
    println!("# Object abstraction run report\n\nPublic training only; {mode}; {} tasks; 12 workers. Core depth 2 versus a finite object AST: segment → select → act → render. No symmetry, repair, or fitted-pool cap.\n\nPrimary search, reserved-example validation, and pool-writing wall time: {primary_seconds:.2}s. Including depth-3 audit: {:.2}s. Resource measurements are in stderr.txt.\n",rows.len(),start.elapsed().as_secs_f64());
    println!("## Predicting unseen test answers after fitting all training examples\n\n| Arm | Training-fit tasks | Task exact match | Grid exact match | Prediction coverage (grids) | Accuracy given fit | Oracle tasks | Oracle-minus-selected |\n|---|---:|---:|---:|---:|---:|---:|---:|");
    for name in ["grid", "objects", "fallback", "validation gate"] {
        let ss: Vec<_> = rows.iter().map(|r| arm(r, name)).collect();
        let fit = ss.iter().filter(|s| s.fit > 0).count();
        let correct = ss.iter().filter(|s| s.correct).count();
        let oracle = ss.iter().filter(|s| s.oracle).count();
        let grids = ss.iter().map(|s| s.grids).sum();
        println!(
            "| {name} | {} | {} | {} | {} | {} | {} | {} |",
            ratio(fit, rows.len()),
            ratio(correct, rows.len()),
            ratio(ss.iter().map(|s| s.grid_correct).sum(), grids),
            ratio(ss.iter().map(|s| s.predicted).sum(), grids),
            ratio(correct, fit),
            ratio(oracle, rows.len()),
            oracle - correct
        );
    }
    let union = rows
        .iter()
        .filter(|r| r.grid.oracle || r.obj.oracle)
        .count();
    let new_fit = rows
        .iter()
        .filter(|r| r.grid.fit == 0 && r.obj.fit > 0)
        .count();
    let new_oracle = rows
        .iter()
        .filter(|r| !r.grid.oracle && r.obj.oracle)
        .count();
    println!("\nUnion oracle: {}. Objects add {new_fit} training-fitting tasks and {new_oracle} oracle-correct tasks beyond core depth 2. Oracle means at least one SINGLE program gives every test output correctly; it is not a selection method. Grid selects first shortest; objects select first AST in the registered order. Fallback prefers any fitting core program; validation gate prefers objects only when reserved-training-answer probability is strictly higher (tolerance 1e-12), otherwise core. The oracle columns for fallback/gate concern the chosen family, not the union.\n",ratio(union,rows.len()));
    println!("## Is an object interpretation predictively useful?\n\nFor each task with ≥2 training examples, fit all except the last; predict the last input. Its output is excluded from inference and palette construction. Test outputs are never used for this decision. Refit all training pairs for the test report above. These are within-task validation examples, not the public evaluation split.\n\n| Family | Eligible tasks | Discovery fit | First-program validation accuracy | Correct-answer support | Mean correct-answer probability | Mean predictive entropy (bits) | Mean multiclass Brier loss |\n|---|---:|---:|---:|---:|---:|---:|---:|");
    let vs: Vec<_> = rows
        .iter()
        .filter(|r| r.validation.available)
        .map(|r| &r.validation)
        .collect();
    let nv = vs.len();
    for obj in [false, true] {
        let fit = vs
            .iter()
            .filter(|v| if obj { v.obj_fit > 0 } else { v.grid_fit > 0 })
            .count();
        let correct = vs
            .iter()
            .filter(|v| if obj { v.obj_correct } else { v.grid_correct })
            .count();
        let support = vs
            .iter()
            .filter(|v| if obj { v.obj_p > 0. } else { v.grid_p > 0. })
            .count();
        let avg = |f: fn(&Validation) -> f64| vs.iter().map(|v| f(v)).sum::<f64>() / nv as f64;
        println!(
            "| {} | {nv} | {} | {} | {} | {} | {} | {} |",
            if obj { "objects" } else { "grid" },
            ratio(fit, nv),
            ratio(correct, nv),
            ratio(support, nv),
            num(if obj {
                avg(|v| v.obj_p)
            } else {
                avg(|v| v.grid_p)
            }),
            num(if obj {
                avg(|v| v.obj_h)
            } else {
                avg(|v| v.grid_h)
            }),
            num(if obj {
                avg(|v| v.obj_brier)
            } else {
                avg(|v| v.grid_brier)
            })
        );
    }
    let stronger = vs.iter().filter(|v| v.obj_p > v.grid_p + 1e-12).count();
    let weaker = vs.iter().filter(|v| v.grid_p > v.obj_p + 1e-12).count();
    let sensitive = vs
        .iter()
        .filter(|v| (v.obj_p > v.grid_p + 1e-12) != (v.obj_p > v.grid_length_p + 1e-12))
        .count();
    println!("\nObjects assign more probability to the reserved answer on {stronger} tasks; grid on {weaker}; ties on {}. Changing ONLY the grid prior to normalized length mass q(l) ∝ 2^-l, distributed uniformly over K^l sequences at length l, changes the object-preference decision on {sensitive} tasks.\n\nPrimary probabilities use a uniform prior over each family's syntax conditioned on discovery fits. Empty pools abstain (unit mass on undefined), giving zero correct-answer support, entropy zero, and Brier loss 2. Low entropy can therefore mean failure, and must be read alongside support and accuracy. Brier is the sum of squared probability errors (0 best, 2 worst). No calibrated probability that a puzzle is intrinsically about objects is claimed; these are comparisons under explicit languages and priors.\n",nv-stronger-weaker);
    println!("| Validation preference | Tasks | Core test correct | Object test correct | Object-only wins | Core-only wins |\n|---|---:|---:|---:|---:|---:|");
    for (label, want) in [("objects higher", true), ("core higher or tie", false)] {
        let rr: Vec<_> = rows
            .iter()
            .filter(|r| {
                r.validation.available && (r.validation.obj_p > r.validation.grid_p + 1e-12) == want
            })
            .collect();
        println!(
            "| {label} | {} | {} | {} | {} | {} |",
            rr.len(),
            rr.iter().filter(|r| r.grid.correct).count(),
            rr.iter().filter(|r| r.obj.correct).count(),
            rr.iter()
                .filter(|r| r.obj.correct && !r.grid.correct)
                .count(),
            rr.iter()
                .filter(|r| r.grid.correct && !r.obj.correct)
                .count()
        );
    }
    let both: Vec<_> = rows
        .iter()
        .filter(|r| r.validation.available && r.validation.grid_fit > 0 && r.validation.obj_fit > 0)
        .collect();
    let both_obj: Vec<_> = both
        .iter()
        .filter(|r| r.validation.obj_p > r.validation.grid_p + 1e-12)
        .collect();
    let both_grid = both
        .iter()
        .filter(|r| r.validation.grid_p > r.validation.obj_p + 1e-12)
        .count();
    println!("\nBoth languages fit the discovery examples on {} tasks. Within these, objects receive higher reserved-answer probability on {}, grid on {}, and {} tie. Among the {} shared-fit tasks preferring objects, object/core test selection is correct on {}/{}. This separates preference between viable families from choosing the only family that fits.\n", both.len(), both_obj.len(), both_grid, both.len()-both_obj.len()-both_grid, both_obj.len(), both_obj.iter().filter(|r|r.obj.correct).count(), both_obj.iter().filter(|r|r.grid.correct).count());
    println!("\n## Depth-3 audit of newly fitting tasks\n\nSelected by training fit and hash order, never test correctness; at most 16 tasks.\n\n| Task | Core d3 fits | Core d3 oracle | Object oracle | Core d3 selected correct | Object selected correct | Audit task seconds |\n|---|---:|---|---|---|---|---:|");
    for (id, s, secs) in &audit {
        let r = rows.iter().find(|r| &r.id == id).unwrap();
        println!(
            "| {id} | {} | {} | {} | {} | {} | {secs:.2} |",
            s.fit, s.oracle, r.obj.oracle, s.correct, r.obj.correct
        );
    }
    println!("\nThis is a targeted audit, not full-corpus depth-3 accuracy. {} audited tasks have an object oracle-correct solution absent from the enumerated core depth-3 pool.\n",audit.iter().filter(|(id,s,_)|!s.oracle&&rows.iter().find(|r|&r.id==id).unwrap().obj.oracle).count());
    println!("## Cost and prediction diagnostics\n\nCore enumerated syntax budget sums to {}; object candidates {} and actual training program/example checks {}. Core memoizes prefixes; these counts are not comparable primitive-operation costs. Summed per-task core/object full-fit times are {:.2}s/{:.2}s (overlap across workers; exclude validation/audit).\n",rows.iter().map(|r|r.grid_candidates).sum::<usize>(),rows.iter().map(|r|r.obj_candidates).sum::<usize>(),rows.iter().map(|r|r.obj_checks).sum::<usize>(),rows.iter().map(|r|r.grid_seconds).sum::<f64>(),rows.iter().map(|r|r.obj_seconds).sum::<f64>());
    println!("| Task | Fits grid/object | Joint predictions grid/object | Object program H | Answer H grid/object | Viable segmentations | Segmentation H | Validation p grid/object | Test correct grid/object | Oracle grid/object |\n|---|---:|---:|---:|---:|---:|---:|---:|---|---|");
    for r in &rows {
        println!(
            "| {} | {}/{} | {}/{} | {} | {}/{} | {} | {} | {}/{} | {}/{} | {}/{} |",
            r.id,
            r.grid.fit,
            r.obj.fit,
            r.grid.unique,
            r.obj.unique,
            num(r.obj.program_h),
            num(r.grid.answer_h),
            num(r.obj.answer_h),
            r.obj.segments,
            num(r.obj.segment_h),
            num(r.validation.grid_p),
            num(r.validation.obj_p),
            r.grid.correct,
            r.obj.correct,
            r.grid.oracle,
            r.obj.oracle
        );
    }
    println!("\nEntropies condition on full training fits; empty-pool entropy is n/a here. Segmentation entropy is over eight named interpretations, weighted by fitting AST counts, with initial entropy 3 bits. Interpretations can coincide on observed grids, so this is not semantic entropy. Joint prediction classes identify agreement only on observed test inputs, including undefined.\n\n## Witnesses and selection failures\n\nShown whenever objects fit and core does not, or object/core test accuracy differs. Correct witnesses were found by post-hoc scoring; they did not guide inference.\n\n| Task | Core selected | Object selected | Object correct witness |\n|---|---|---|---|");
    for r in &rows {
        if r.obj.fit > 0 && (r.grid.fit == 0 || r.grid.correct != r.obj.correct) {
            println!(
                "| {} | `{}` | `{}` | `{}` |",
                r.id, r.grid_name, r.obj_name, r.obj_witness
            );
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn labels_do_not_enter_inference() {
        let x = Grid::of_rows(&[vec![0, 1, 0], vec![0, 0, 0]]);
        let y = Grid::of_rows(&[vec![1]]);
        let mut t = Task {
            id: "fixture".into(),
            train: vec![(x.clone(), y.clone()), (x.clone(), y.clone())],
            test: vec![(x.clone(), y)],
        };
        let input = Inputs::from_task(&t, true);
        let a = object(&input);
        let g = core(&input, 2);
        t.train[1].1 = Grid::of_rows(&[vec![9]]);
        t.test[0].1 = Grid::of_rows(&[vec![8]]);
        let altered = Inputs::from_task(&t, true);
        assert_eq!(input.colours(), altered.colours());
        let b = object(&altered);
        assert_eq!(a.names, b.names);
        assert_eq!(a.preds, b.preds);
        assert_eq!(g.names, core(&altered, 2).names);
        assert!(!altered.colours().contains(&9));
        assert!(!altered.colours().contains(&8));
    }
    #[test]
    fn abstention_is_not_correct_certainty() {
        let p = Pool::default();
        let ds = distribution(&p, 0, false);
        assert_eq!(ds.get(&None), Some(&1.));
        assert_eq!(entropy(ds.into_values()), 0.);
    }
}
