use std::{
    fs,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
};
use symarc::{dsl, search, task};
use symarc_exp_003_objects::{core, object, Inputs, Pool};
fn hash(s: &str) -> u64 {
    s.bytes().fold(14695981039346656037, |h, b| {
        (h ^ b as u64).wrapping_mul(1099511628211)
    })
}
fn pool_row(dataset: &str, split: &str, id: &str, arm: &str, p: &Pool, t: &symarc::Task) -> String {
    let correct = |ps: &Vec<Option<symarc::grid::Grid>>| {
        ps.iter()
            .zip(&t.test)
            .all(|(a, (_, y))| a.as_ref() == Some(y))
    };
    let grids = p.preds.first().map_or(0, |ps| {
        ps.iter()
            .zip(&t.test)
            .filter(|(a, (_, y))| a.as_ref() == Some(y))
            .count()
    });
    let witness = p
        .preds
        .iter()
        .position(correct)
        .map_or("".into(), |i| p.names[i].clone());
    format!(
        "{dataset}\t{split}\t{id}\t{arm}\t{}\t{}\t{grids}\t{}\t{}\t{}\t{}\n",
        !p.preds.is_empty(),
        p.preds.first().is_some_and(correct),
        t.test.len(),
        p.preds.iter().any(correct),
        p.names.first().cloned().unwrap_or_default(),
        witness
    )
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let mode = &args[1];
    assert!(mode == "pilot" || mode == "full");
    let mut items = vec![];
    for (dataset, root) in [("ARC1", &args[2]), ("ARC2", &args[3])] {
        let mut cohort = vec![];
        for split in ["training", "evaluation"] {
            for p in fs::read_dir(format!("{root}/{split}")).unwrap() {
                let p = p.unwrap().path();
                if p.extension().is_some_and(|e| e == "json") {
                    let id = p.file_stem().unwrap().to_str().unwrap().to_owned();
                    cohort.push((dataset.to_owned(), split.to_owned(), id, p));
                }
            }
        }
        assert_eq!(cohort.len(), if dataset == "ARC1" { 800 } else { 1120 });
        cohort.sort_by_key(|x| (hash(&x.2), x.2.clone()));
        if mode == "pilot" {
            cohort.truncate(24);
        }
        items.extend(cohort);
    }
    let next = AtomicUsize::new(0);
    let results = Mutex::new(vec![]);
    std::thread::scope(|scope| {
        for _ in 0..12 {
            let next = &next;
            let results = &results;
            let items = &items;
            scope.spawn(move || loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= items.len() {
                    break;
                }
                let (dataset, split, id, path) = &items[i];
                let t = task::load(path.to_str().unwrap(), id);
                let full = search::run_task(&t, &search::Config::default());
                let input = Inputs::from_task(&t, false);
                let g = core(&input, 2);
                let o = object(&input);
                let mut row = format!(
                    "{dataset}\t{split}\t{id}\tcomplete\t{}\t{}\t{}\t{}\tNA\t{}\t\n",
                    full.fitted(),
                    full.solved(),
                    full.tests.iter().filter(|q| q.correct).count(),
                    t.test.len(),
                    dsl::program_name(&full.prog)
                );
                for (name, p) in [
                    ("grid_d2", &g),
                    ("objects", &o),
                    ("grid_objects", if g.preds.is_empty() { &o } else { &g }),
                ] {
                    row += &pool_row(dataset, split, id, name, p, &t);
                }
                results.lock().unwrap().push((i, row));
            });
        }
    });
    let mut rows = results.into_inner().unwrap();
    rows.sort_by_key(|r| r.0);
    print!("dataset\tsplit\ttask\tarm\tfit\tcorrect\tcorrect_grids\ttest_grids\toracle_chosen_family\tselected_program\toracle_witness\n");
    for (_, r) in rows {
        print!("{r}");
    }
}
