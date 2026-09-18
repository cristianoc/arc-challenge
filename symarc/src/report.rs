//! Stable text output, compatible with subsets/estimate.py.
use crate::dsl::program_name;
use crate::grid::Gen;
use crate::search::{Config, Result};
use std::fmt::Write;
pub const FAMILIES: [&str; 5] = ["dihedral", "colours", "cyclic", "rows", "cols"];
/// Two nonnegative decimal places, with halfway values rounded upward.
pub fn number(f: f64) -> String {
    let n = (f * 100.0).round() as u64;
    format!("{}.{:02}", n / 100, n % 100)
}
pub fn families(gens: &[Gen]) -> String {
    FAMILIES
        .iter()
        .filter_map(|f| {
            let n = gens.iter().filter(|g| g.family() == *f).count();
            (n > 0).then(|| format!("{f}:{n}"))
        })
        .collect::<Vec<_>>()
        .join(" ")
}
pub fn gens_name(gens: &[Gen]) -> String {
    format!(
        "[{}]",
        gens.iter().map(Gen::name).collect::<Vec<_>>().join(", ")
    )
}
pub fn bool_str(b: Option<bool>) -> &'static str {
    match b {
        None => "-",
        Some(true) => "yes",
        Some(false) => "NO",
    }
}
pub fn result(r: &Result, verbose: bool) -> String {
    let sym = r.tests.iter().filter(|t| t.by_symmetry).count();
    let sym_ok = r
        .tests
        .iter()
        .filter(|t| t.sym_correct == Some(true))
        .count();
    let eqv = r
        .tests
        .iter()
        .map(|t| bool_str(t.equivariant))
        .collect::<Vec<_>>()
        .join(",");
    let det = r
        .tests
        .iter()
        .map(|t| bool_str(t.determined))
        .collect::<Vec<_>>()
        .join(",");
    let status = if r.solved() {
        "SOLVED"
    } else if r.fitted() {
        "fit-only"
    } else {
        "unfit"
    };
    let real = if r.realized {
        families(&r.gens)
    } else {
        "n/a".into()
    };
    let mut out = format!("{}  func[{}] real[{}] |C|={}{} gain={}b outs={}  progsD={} progs={} fitD={} fitC={} evals={} sym={}/{}(ok {}) equiv={} det={}  {}  :: {}\n",
        r.id, families(&r.func_gens), real, r.closure_size, if r.capped { "+" } else { "" },
        number(r.gain),r.outputs,r.consistent_d,r.consistent,number(r.fit_d),number(r.fit_c),r.evals,sym,r.tests.len(),sym_ok,eqv,det,status,program_name(&r.prog));
    if verbose {
        writeln!(out, "  accepted: {}", gens_name(&r.gens)).unwrap();
        writeln!(
            out,
            "  rejected by realizability: {}",
            gens_name(&r.rejected)
        )
        .unwrap();
        writeln!(out, "  repaired: {}", gens_name(&r.repaired)).unwrap();
        for t in &r.tests {
            if let Some(g) = &t.prediction {
                writeln!(
                    out,
                    "  prediction ({}):\n{g}",
                    if t.correct { "correct" } else { "wrong" }
                )
                .unwrap();
            } else {
                writeln!(out, "  prediction: undefined").unwrap();
            }
        }
    }
    out
}

/// Aggregate only values already computed by the solver, before formatting.
#[derive(Default)]
struct Metrics {
    tasks: usize,
    tests: usize,
    solved: usize,
    fitted: usize,
    both: usize,
    realized: usize,
    test_correct: usize,
    predicted: usize,
    rejected: usize,
    repaired: usize,
    capped: usize,
    by_symmetry: usize,
    sym_correct: usize,
    undefined: usize,
    equiv_yes: usize,
    equiv_no: usize,
    det_yes: usize,
    det_no: usize,
    func_families: [usize; 5],
    real_families: [usize; 5],
    median_gain: f64,
}
impl Metrics {
    fn from_results(rs: &[&Result]) -> Self {
        let mut m = Self::default();
        let mut gains = Vec::new();
        for r in rs {
            m.tasks += 1;
            m.solved += usize::from(r.solved());
            m.fitted += usize::from(r.fitted());
            m.both += usize::from(r.solved() && r.fitted());
            m.realized += usize::from(r.realized);
            m.rejected += usize::from(!r.rejected.is_empty());
            m.repaired += usize::from(!r.repaired.is_empty());
            m.capped += usize::from(r.capped);
            gains.push(r.gain);
            for (i, family) in FAMILIES.iter().enumerate() {
                m.func_families[i] +=
                    usize::from(r.func_gens.iter().any(|g| g.family() == *family));
                m.real_families[i] +=
                    usize::from(r.realized && r.gens.iter().any(|g| g.family() == *family));
            }
            for t in &r.tests {
                m.tests += 1;
                m.test_correct += usize::from(t.correct);
                m.predicted += usize::from(t.prediction.is_some());
                m.by_symmetry += usize::from(t.by_symmetry);
                m.sym_correct += usize::from(t.sym_correct == Some(true));
                m.undefined += usize::from(!t.defined);
                m.equiv_yes += usize::from(t.equivariant == Some(true));
                m.equiv_no += usize::from(t.equivariant == Some(false));
                m.det_yes += usize::from(t.determined == Some(true));
                m.det_no += usize::from(t.determined == Some(false));
            }
        }
        gains.sort_by(f64::total_cmp);
        m.median_gain = gains.get(gains.len() / 2).copied().unwrap_or(0.0);
        m
    }
}
fn rate(n: usize, d: usize) -> String {
    if d == 0 {
        "n/a (0/0)".into()
    } else {
        format!("{n}/{d} ({}%)", number(100.0 * n as f64 / d as f64))
    }
}
fn markdown_text(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('|', "&#124;")
        .replace('`', "&#96;")
        .replace(['\n', '\r'], " ")
}

/// A Markdown run report emitted directly from in-memory results. Split labels
/// come from task paths and never affect search. Timing excludes report I/O.
pub fn summary(
    rs: &[Result],
    splits: &[String],
    cfg: &Config,
    selection: &str,
    workers: usize,
    elapsed_seconds: f64,
) -> String {
    assert_eq!(rs.len(), splits.len(), "one split label per result");
    let mut groups: std::collections::BTreeMap<&str, Vec<&Result>> = Default::default();
    for (r, split) in rs.iter().zip(splits) {
        groups.entry(split).or_default().push(r);
    }
    let mut columns: Vec<(String, Metrics)> = groups
        .into_iter()
        .map(|(split, rows)| {
            let name = match split {
                "training" => "Public training tasks".into(),
                "evaluation" => "Public evaluation tasks".into(),
                "" => "Selected tasks".into(),
                other => format!("Split: {}", markdown_text(other)),
            };
            (name, Metrics::from_results(&rows))
        })
        .collect();
    // Conventional split order, with any custom splits following.
    columns.sort_by_key(|(name, _)| match name.as_str() {
        "Public training tasks" => 0,
        "Public evaluation tasks" => 1,
        _ => 2,
    });
    if columns.len() != 1 {
        columns.push((
            "All selected tasks".into(),
            Metrics::from_results(&rs.iter().collect::<Vec<_>>()),
        ));
    }
    let mut out = format!("# SymArc run report\n\nSelection: {}.\n\nWorkers: {workers}. Seed: {}. Loading + search wall time: {} s (excludes report I/O and compilation).\n\n",
        markdown_text(selection), cfg.seed, number(elapsed_seconds));
    out.push_str("## Predictive performance\n\nOne selected prediction per test input. **Task exact match** requires correct shape and every cell on every test output in a task; **test-grid exact match** scores individual test outputs. Counts and denominators are explicit.\n\n");
    let header = format!(
        "| Metric | {} |\n|---|{}\n",
        columns
            .iter()
            .map(|(n, _)| n.as_str())
            .collect::<Vec<_>>()
            .join(" | "),
        "---:|".repeat(columns.len())
    );
    out.push_str(&header);
    let mut row = |label: &str, f: fn(&Metrics) -> String| {
        writeln!(
            out,
            "| {label} | {} |",
            columns
                .iter()
                .map(|(_, m)| f(m))
                .collect::<Vec<_>>()
                .join(" | ")
        )
        .unwrap();
    };
    row("Tasks", |m| m.tasks.to_string());
    row("Test grids", |m| m.tests.to_string());
    row("**Task exact-match accuracy (primary)**", |m| {
        rate(m.solved, m.tasks)
    });
    row("Test-grid exact-match accuracy", |m| {
        rate(m.test_correct, m.tests)
    });
    row("Prediction coverage (test grids)", |m| {
        rate(m.predicted, m.tests)
    });
    row("Fits all within-task training examples", |m| {
        rate(m.fitted, m.tasks)
    });
    row("Fits training and solves every test", |m| {
        rate(m.both, m.tasks)
    });
    row("Task accuracy conditional on training fit", |m| {
        rate(m.both, m.fitted)
    });
    out.push_str("\nSplit labels describe task datasets, not a model trained across tasks. Each task is fitted independently. Test inputs participate in label-free constraints; test output labels are used only for scoring. Conditional accuracy applies only to the training-fitting subset.\n\nRates describe the selected tasks. A stratified quick set is not representative without reweighting. Public evaluation data used during development is not an untouched hidden test set.\n\n## Symmetry diagnostics\n\n");
    out.push_str(&header);
    let mut row = |label: &str, f: fn(&Metrics) -> String| {
        writeln!(
            out,
            "| {label} | {} |",
            columns
                .iter()
                .map(|(_, m)| f(m))
                .collect::<Vec<_>>()
                .join(" | ")
        )
        .unwrap();
    };
    row("Realizability rejected a generator (tasks)", |m| {
        rate(m.rejected, m.realized)
    });
    row("Successful repair (tasks)", |m| {
        rate(m.repaired, m.realized)
    });
    row("Capped final closure (tasks)", |m| rate(m.capped, m.tasks));
    row("Median reported coverage gain (bits)", |m| {
        number(m.median_gain)
    });
    row("Test inputs reached by symmetry", |m| {
        rate(m.by_symmetry, m.tests)
    });
    row("Symmetry-answer accuracy on reached inputs", |m| {
        rate(m.sym_correct, m.by_symmetry)
    });
    row("Program undefined at test input", |m| {
        m.undefined.to_string()
    });
    row("Test-input equivariance: yes / no / unassessed", |m| {
        format!(
            "{} / {} / {}",
            m.equiv_yes,
            m.equiv_no,
            m.tests - m.equiv_yes - m.equiv_no
        )
    });
    row("Survivor agreement: yes / no / unassessed", |m| {
        format!(
            "{} / {} / {}",
            m.det_yes,
            m.det_no,
            m.tests - m.det_yes - m.det_no
        )
    });
    for (i, family) in FAMILIES.iter().enumerate() {
        writeln!(
            out,
            "| Tasks retaining {family}: functional / realizable | {} |",
            columns
                .iter()
                .map(|(_, m)| format!("{} / {}", m.func_families[i], m.real_families[i]))
                .collect::<Vec<_>>()
                .join(" | ")
        )
        .unwrap();
    }
    out.push_str("\nRejection and repair denominators are tasks with a fitting program. Capped gains measure explored coverage; they are not exact closure entropies. Realizability uses sampling and may replace programs through repair. Program-count ratios are not automatically hypothesis-entropy reductions.\n\n## Configuration\n\n");
    writeln!(out,"Enumeration depth: {}. Mutation length threshold: {}. Realizability filter: {}.\n\nClosure caps: functionality {}, sampling {}, final {}. Extra sample attempts: {}.\n\nHill climbing: {} restarts × {} steps; repair: {} steps.\n",cfg.enum_len,cfg.max_len,cfg.realize_filter,cfg.cap_greedy,cfg.sample_cap,cfg.cap,cfg.fit_size,cfg.restarts,cfg.steps,cfg.repair_steps).unwrap();
    out.push_str("This is one run at one seed. Seed variation, cell accuracy, probability calibration, and a causal benefit from symmetry are not measured by this report. For harness runs, run.json records the exact command, revision, hashes, platform, and end-to-end subprocess timing.\n");
    out
}
