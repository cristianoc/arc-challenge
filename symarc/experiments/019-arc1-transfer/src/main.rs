//! Thin export adapter. All search is delegated to the unchanged stable library.
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use symarc::{dsl, search, task};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(args.len(), 2, "usage: symarc-exp-019-export PROJECTED_DATA_ROOT");
    let root = Path::new(&args[1]);
    let mut tasks = Vec::new();
    for split in ["training", "evaluation"] {
        let mut paths: Vec<_> = std::fs::read_dir(root.join(split)).unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("json"))
            .collect();
        paths.sort();
        for p in paths {
            let id = p.file_stem().unwrap().to_str().unwrap();
            tasks.push((split, task::load(p.to_str().unwrap(), id)));
        }
    }
    assert!(!tasks.is_empty(), "no projected tasks");
    let next = AtomicUsize::new(0);
    let start = Instant::now();
    let mut results = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..12).map(|_| {
            let next = &next;
            let tasks = &tasks;
            scope.spawn(move || {
                let mut out = Vec::new();
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= tasks.len() { break; }
                    let (split, task) = &tasks[i];
                    let result = search::run_task(task, &search::Config::default());
                    let grids: Vec<_> = result.tests.iter().map(|t|
                        t.prediction.as_ref().map_or("null".to_string(), |g| format!("{:?}", g.rows()))
                    ).collect();
                    out.push((i, format!(
                        "{{\"id\":{:?},\"split\":{:?},\"fitted\":{},\"program\":{:?},\"predictions\":[{}]}}",
                        task.id, split, result.fitted(), dsl::program_name(&result.prog), grids.join(",")
                    )));
                }
                out
            })
        }).collect();
        handles.into_iter().flat_map(|h| h.join().unwrap()).collect::<Vec<_>>()
    });
    results.sort_by_key(|r| r.0);
    for (_, line) in results { println!("{line}"); }
    eprintln!("tasks={} workers=12 seed=0 seconds={:.6}", tasks.len(), start.elapsed().as_secs_f64());
}
