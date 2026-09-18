//! Stable text output, compatible with subsets/estimate.py.
use crate::dsl::program_name;
use crate::grid::Gen;
use crate::search::{Config, Result, TestReport};
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
pub fn summary(rs: &[Result], cfg: &Config) -> String {
    let count = |f: fn(&Result) -> bool| rs.iter().filter(|r| f(r)).count();
    let ct = |f: fn(&TestReport) -> bool| rs.iter().flat_map(|r| &r.tests).filter(|t| f(t)).count();
    let nt = ct(|_| true);
    let fam = |real: bool| {
        format!(
            "[{}]",
            FAMILIES
                .iter()
                .map(|f| {
                    let n = rs
                        .iter()
                        .filter(|r| {
                            if real {
                                r.realized && r.gens.iter().any(|g| g.family() == *f)
                            } else {
                                r.func_gens.iter().any(|g| g.family() == *f)
                            }
                        })
                        .count();
                    format!("({f}, {n})")
                })
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    let mut gains: Vec<_> = rs.iter().map(|r| r.gain).collect();
    gains.sort_by(f64::total_cmp);
    let median = gains.get(gains.len() / 2).copied().unwrap_or(0.0);
    format!("\ntasks: {}  test inputs: {nt}\n\
generators passing functionality, tasks by family: {}\n\
generators passing realizability, among the {} tasks with a fitting program, tasks by family: {}\n\
tasks where realizability rejected a functional generator: {}; repaired by hill climbing: {}\n\
closure capped at {}: {}; median gain: {} bits\n\
programs consistent with data: {}; fits data: {}; solves all tests: {}; fits data and solves all tests: {}; fits data but wrong on a test: {}\n\
test inputs determined by symmetry alone: {}/{nt}; of which the symmetry answer is correct: {}\n\
test inputs where the program is undefined: {}\n\
equivariance at test input: yes {}, NO {}\n\
surviving programs disagree at a test input: {}; agree: {}\n",
        rs.len(),fam(false),count(|r|r.realized),fam(true),count(|r|!r.rejected.is_empty()),count(|r|!r.repaired.is_empty()),
        cfg.cap,count(|r|r.capped),number(median),count(|r|r.consistent_d>0),count(Result::fitted),count(Result::solved),
        count(|r|r.fitted()&&r.solved()),count(|r|r.fitted()&&!r.solved()),ct(|t|t.by_symmetry),ct(|t|t.sym_correct==Some(true)),
        ct(|t|!t.defined),ct(|t|t.equivariant==Some(true)),ct(|t|t.equivariant==Some(false)),ct(|t|t.determined==Some(false)),ct(|t|t.determined==Some(true)))
}
