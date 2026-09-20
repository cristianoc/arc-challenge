//! Stable solver CLI. Experiments depend on the library, never on this binary's source.
use std::path::Path;
use std::time::Instant;
use symarc::grid::{self, candidates, closure, greedy_gens, Gen, Grid};
use symarc::search::Config;
use symarc::task::load;
use symarc::{dsl, random, report, search, Task};

fn bench(t: &Task, cfg: &Config) -> String {
    let (cands, pool) = t.candidates_pool();
    let start = Instant::now();
    let (gens, cg) = greedy_gens(&cands, &t.train, cfg.cap_greedy);
    let d1 = start.elapsed().as_millis();
    let start = Instant::now();
    let c = closure(&gens, &t.train, cfg.cap);
    let d2 = start.elapsed().as_millis();
    let start = Instant::now();
    let progs = dsl::enumerate(&pool, &t.train, cfg.enum_len);
    let d3 = start.elapsed().as_millis();
    format!("{id} cands={} pool={} train={}\n{id} greedy    accepted={} closure={} {d1} ms\n{id} closure   size={} capped={} {d2} ms\n{id} enumerate fitting={} {d3} ms\n{id} TOTAL     {} ms\n",
        cands.len(),pool.len(),t.train.len(),gens.len(),cg.size(),c.size(),c.capped,progs.len(),d1+d2+d3,id=t.id)
}
#[derive(Default)]
struct Opts {
    cfg: Config,
    data: Option<String>,
    root: Option<String>,
    tasks_file: Option<String>,
    tasks: Vec<String>,
    limit: Option<usize>,
    demo: bool,
    bench: bool,
    verbose: bool,
    threads: usize,
}
const USAGE: &str = "symarc --demo | --data DIR [--task ID] | --root DIR --tasks-file FILE
Options: --threads N --limit N --show --bench --cap N --capgreedy N
         --samplecap N --fit N --restarts N --steps N --repair N
         --maxlen N --enumlen N --seed N --norealize";
fn parse_args() -> std::result::Result<Opts, String> {
    let mut o = Opts {
        threads: 1,
        ..Opts::default()
    };
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--demo" => o.demo = true,
            "--bench" => o.bench = true,
            "--show" => o.verbose = true,
            "--norealize" => o.cfg.realize_filter = false,
            "--help" | "-h" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            "--data" | "--root" | "--tasks-file" | "--task" => {
                let v = args
                    .next()
                    .ok_or_else(|| format!("missing value for {a}"))?;
                match a.as_str() {
                    "--data" => o.data = Some(v),
                    "--root" => o.root = Some(v),
                    "--tasks-file" => o.tasks_file = Some(v),
                    _ => o.tasks.push(v),
                }
            }
            "--seed" => {
                o.cfg.seed = args
                    .next()
                    .ok_or("missing seed")?
                    .parse()
                    .map_err(|_| "invalid seed")?
            }
            "--threads" | "--limit" | "--cap" | "--capgreedy" | "--samplecap" | "--fit"
            | "--restarts" | "--steps" | "--repair" | "--maxlen" | "--enumlen" => {
                let n: usize = args
                    .next()
                    .ok_or_else(|| format!("missing value for {a}"))?
                    .parse()
                    .map_err(|_| format!("invalid number for {a}"))?;
                match a.as_str() {
                    "--threads" => {
                        if n == 0 {
                            return Err("threads must be positive".into());
                        }
                        o.threads = n;
                    }
                    "--limit" => o.limit = Some(n),
                    "--cap" => o.cfg.cap = n.max(1),
                    "--capgreedy" => o.cfg.cap_greedy = n.max(1),
                    "--samplecap" => o.cfg.sample_cap = n.max(1),
                    "--fit" => o.cfg.fit_size = n,
                    "--restarts" => o.cfg.restarts = n,
                    "--steps" => o.cfg.steps = n,
                    "--repair" => o.cfg.repair_steps = n,
                    "--maxlen" => o.cfg.max_len = n,
                    _ => o.cfg.enum_len = n,
                }
            }
            _ => return Err(format!("unknown option: {a}")),
        }
    }
    if !o.demo && o.data.is_none() && !(o.root.is_some() && o.tasks_file.is_some()) {
        return Err(USAGE.into());
    }
    if o.tasks_file.is_some() && o.root.is_none() {
        return Err("--tasks-file requires --root".into());
    }
    Ok(o)
}
fn load_tasks(o: &Opts) -> std::result::Result<(Vec<Task>, Vec<String>), String> {
    let mut paths = vec![];
    if let Some(f) = &o.tasks_file {
        for line in std::fs::read_to_string(f)
            .map_err(|e| format!("{f}: {e}"))?
            .lines()
        {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let fields: Vec<_> = line.split_whitespace().collect();
            if fields.len() < 2 {
                return Err(format!("bad task list line: {line}"));
            }
            paths.push(
                Path::new(o.root.as_ref().unwrap())
                    .join(fields[1])
                    .join(format!("{}.json", fields[0])),
            );
        }
    } else {
        let dir = o.data.as_ref().unwrap();
        for e in std::fs::read_dir(dir).map_err(|e| format!("{dir}: {e}"))? {
            let path = e.map_err(|e| e.to_string())?.path();
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                paths.push(path);
            }
        }
        paths.sort();
    }
    paths.retain(|p| {
        o.tasks.is_empty()
            || o.tasks
                .iter()
                .any(|id| Some(id.as_str()) == p.file_stem().and_then(|s| s.to_str()))
    });
    paths.truncate(o.limit.unwrap_or(paths.len()));
    Ok(paths
        .iter()
        .map(|p| {
            (
                load(
                    p.to_str().unwrap(),
                    p.file_stem().unwrap().to_str().unwrap(),
                ),
                p.parent()
                    .and_then(|p| p.file_name())
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_string(),
            )
        })
        .unzip())
}
/// Ordered output, task-local random states, and a shared work queue for uneven tasks.
fn parallel<T: Send>(tasks: &[Task], threads: usize, f: impl Fn(&Task) -> T + Sync) -> Vec<T> {
    if threads <= 1 {
        return tasks.iter().map(f).collect();
    }
    use std::sync::atomic::{AtomicUsize, Ordering};
    let next = AtomicUsize::new(0);
    let mut out = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads.min(tasks.len()))
            .map(|_| {
                let f = &f;
                let next = &next;
                scope.spawn(move || {
                    let mut out = vec![];
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= tasks.len() {
                            break;
                        }
                        out.push((i, f(&tasks[i])));
                    }
                    out
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect::<Vec<_>>()
    });
    out.sort_by_key(|(i, _)| *i);
    out.into_iter().map(|(_, r)| r).collect()
}
fn main() {
    let start = Instant::now();
    let o = parse_args().unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(2);
    });
    if o.demo {
        demo();
        return;
    }
    let (tasks, splits) = load_tasks(&o).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(2);
    });
    if o.bench {
        for s in parallel(&tasks, o.threads, |t| bench(t, &o.cfg)) {
            print!("{s}");
        }
    } else {
        let results = parallel(&tasks, o.threads, |t| search::run_task(t, &o.cfg));
        let mut selection = o
            .tasks_file
            .as_ref()
            .map(|p| format!("task list {p}"))
            .unwrap_or_else(|| format!("directory {}", o.data.as_ref().unwrap()));
        if !o.tasks.is_empty() {
            selection.push_str(&format!("; task filter {}", o.tasks.join(", ")));
        }
        if let Some(limit) = o.limit {
            selection.push_str(&format!("; limit {limit}"));
        }
        print!(
            "{}",
            report::summary(
                &results,
                &splits,
                &o.cfg,
                &selection,
                o.threads,
                start.elapsed().as_secs_f64()
            )
        );
        println!("\n## Task details\n\n```text");
        for r in &results {
            print!("{}", report::result(r, o.verbose));
        }
        println!("```");
    }
    eprintln!(
        "{} tasks, {} thread(s): wall {} ms",
        tasks.len(),
        o.threads,
        start.elapsed().as_millis()
    );
}
fn demo() {
    use dsl::program_name;
    use report::{bool_str, families, gens_name};
    use search::{equivariant_at, eval, fit_sample, hill_climb_from};
    let x = Grid::of_rows(&[vec![0, 0, 0], vec![0, 7, 0], vec![0, 0, 0]]);
    let y = Grid::of_rows(&[vec![0, 0, 0], vec![7, 7, 7], vec![0, 0, 0]]);
    println!("Data: one example\n{x}\n  ->\n{y}\n");
    let d = vec![(x, y)];
    let pal: Vec<_> = (1..=9).collect();
    let gens = candidates(3, 3, &pal);
    let (acc, c) = greedy_gens(&gens, &d, 100000);
    println!(
        "candidates: {}; functional: {} [{}]",
        gens.len(),
        acc.len(),
        families(&acc)
    );
    println!(
        "rejected by functionality: {}",
        gens_name(
            &gens
                .iter()
                .filter(|g| !acc.contains(g))
                .copied()
                .collect::<Vec<_>>()
        )
    );
    let outputs = |c: &grid::Closure| {
        c.pairs
            .iter()
            .map(|(_, y)| (y, ()))
            .collect::<grid::FxMap<_, ()>>()
            .len()
    };
    println!(
        "|C(D)| = {}  functional = {}  outputs = {}  gain = {:.2} bits",
        c.size(),
        c.functional,
        outputs(&c),
        (c.size() as f64).log2()
    );
    let note: Vec<_> = gens
        .iter()
        .filter(|g| {
            matches!(
                g,
                Gen::SwapRows(..) | Gen::SwapCols(..) | Gen::SwapColours(..)
            )
        })
        .copied()
        .collect();
    let cn = closure(&note, &d, 100000);
    println!(
        "note's group (rows × cols × colours): |C(D)| = {}  outputs = {}  gain = {:.2} bits",
        cn.size(),
        outputs(&cn),
        (cn.size() as f64).log2()
    );
    let mut rng = random::Random::new(0);
    let s = fit_sample(&c, &acc, &d, 64, &mut rng);
    let pool = dsl::pool(&[0, 7]);
    let cl = hill_climb_from(&[], &pool, &s, 50, 200, 3, &mut rng);
    println!(
        "hill climb on {} closure pairs: {}  fitness={:.2}  evals={}",
        s.len(),
        program_name(&cl.prog),
        cl.fitness,
        cl.evals
    );
    let cl_d = hill_climb_from(&[], &pool, &d, 50, 200, 3, &mut rng);
    println!(
        "hill climb on the single example: {}  fitness={:.2}  evals={}",
        program_name(&cl_d.prog),
        cl_d.fitness,
        cl_d.evals
    );
    let cons = dsl::enumerate(&pool, &s, 2);
    println!(
        "programs of length ≤ 2 consistent with the closure: {}: [{}]",
        cons.len(),
        cons.iter()
            .take(8)
            .map(|p| program_name(p))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "programs of length ≤ 2 consistent with the single example: {}",
        dsl::enumerate(&pool, &d, 2).len()
    );
    let t = Grid::of_rows(&[vec![0, 0, 0], vec![0, 0, 0], vec![0, 3, 0]]);
    println!(
        "test input\n{t}\n  in closure: {}; prediction:\n{}",
        c.out_of.contains_key(&t),
        eval(&cl.prog, &t)
            .map(|g| g.to_string())
            .unwrap_or("undefined".into())
    );
    let t2 = Grid::of_rows(&[vec![0, 0, 0, 0], vec![0, 0, 3, 0], vec![0, 0, 0, 0]]);
    println!(
        "test input\n{t2}\n  in closure: {}; equivariant: {}; prediction:\n{}",
        c.out_of.contains_key(&t2),
        bool_str(equivariant_at(&acc, &cl.prog, &t2)),
        eval(&cl.prog, &t2)
            .map(|g| g.to_string())
            .unwrap_or("undefined".into())
    );
}
