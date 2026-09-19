//! Full experimental pipeline. Caps, traversal, sampling and tie-breaking
//! preserve the original solver's behavior; capped closure checks are partial.
use crate::dsl::{self, Prim};
use crate::grid::{closure, greedy_gens, Closure, Example, FxMap, Gen, Grid};
use crate::random::Random;
use crate::Task;

pub type Prog = Vec<Prim>;
#[derive(Clone, Debug)]
pub struct Config {
    pub cap_greedy: usize,
    pub sample_cap: usize,
    pub cap: usize,
    pub fit_size: usize,
    pub restarts: usize,
    pub steps: usize,
    pub repair_steps: usize,
    pub max_len: usize,
    pub enum_len: usize,
    pub seed: u64,
    pub realize_filter: bool,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            cap_greedy: 1000,
            sample_cap: 300,
            cap: 20000,
            fit_size: 64,
            restarts: 16,
            steps: 200,
            repair_steps: 150,
            max_len: 3,
            enum_len: 2,
            seed: 0,
            realize_filter: true,
        }
    }
}
pub fn eval(p: &[Prim], x: &Grid) -> Option<Grid> {
    p.iter()
        .try_fold(x.clone(), |g, &pr| dsl::eval_bounded(pr, &g))
}
pub fn fitness(p: &[Prim], s: &[Example]) -> f64 {
    if s.is_empty() {
        return 0.0;
    }
    s.iter()
        .filter(|(x, y)| eval(p, x).as_ref() == Some(y))
        .count() as f64
        / s.len() as f64
}
pub fn fits_all(p: &[Prim], s: &[Example]) -> bool {
    s.iter().all(|(x, y)| eval(p, x).as_ref() == Some(y))
}
fn act_pair(g: Gen, p: &Example) -> Option<Example> {
    Some((g.act(&p.0)?, g.act(&p.1)?))
}
fn random_walk(gens: &[Gen], len: usize, p: &Example, rng: &mut Random) -> Example {
    let mut q = p.clone();
    if !gens.is_empty() {
        for _ in 0..len {
            if let Some(next) = act_pair(gens[rng.range(0, gens.len() - 1)], &q) {
                q = next;
            }
        }
    }
    q
}
pub fn fit_sample(
    c: &Closure,
    gens: &[Gen],
    d: &[Example],
    n: usize,
    rng: &mut Random,
) -> Vec<Example> {
    if !c.capped && c.size() <= n.saturating_add(d.len()) {
        return c.pairs.clone();
    }
    let mut s = d.to_vec();
    let mut seen: FxMap<Grid, ()> = d.iter().map(|(x, _)| (x.clone(), ())).collect();
    for &g in gens {
        for p in d {
            if let Some(q) = act_pair(g, p) {
                if seen.insert(q.0.clone(), ()).is_none() {
                    s.push(q);
                }
            }
        }
    }
    for i in 0..n {
        let q = if !c.capped {
            c.pairs[rng.range(0, c.size() - 1)].clone()
        } else {
            let len = rng.range(2, 12);
            random_walk(gens, len, &d[i % d.len()], rng)
        };
        if seen.insert(q.0.clone(), ()).is_none() {
            s.push(q);
        }
    }
    s
}
fn sample_for(gens: &[Gen], d: &[Example], cfg: &Config, rng: &mut Random) -> Vec<Example> {
    fit_sample(
        &closure(gens, d, cfg.sample_cap),
        gens,
        d,
        cfg.fit_size,
        rng,
    )
}
fn rand_prim(pool: &[Prim], rng: &mut Random) -> Prim {
    pool[rng.range(0, pool.len() - 1)]
}
fn mutate(p: &[Prim], pool: &[Prim], max_len: usize, rng: &mut Random) -> Prog {
    let mut q = p.to_vec();
    let n = q.len();
    let k = rng.range(0, 2);
    if n == 0 || (n < max_len && k == 0) {
        let i = rng.range(0, n);
        q.insert(i, rand_prim(pool, rng));
    } else if (n >= max_len && k == 0) || (n < max_len && k == 1) {
        q.remove(rng.range(0, n - 1));
    } else {
        let i = rng.range(0, n - 1);
        q[i] = rand_prim(pool, rng);
    }
    q
}
pub struct Climb {
    pub prog: Prog,
    pub fitness: f64,
    pub evals: usize,
}
fn shorten(p: Prog, s: &[Example], f: f64) -> (Prog, usize) {
    let mut cur = p;
    let mut evals = 0;
    let mut i = 0;
    while i < cur.len() {
        let mut cand = cur.clone();
        cand.remove(i);
        evals += 1;
        if fitness(&cand, s) >= f {
            cur = cand;
        } else {
            i += 1;
        }
    }
    (cur, evals)
}
pub fn hill_climb_from(
    start: &[Prim],
    pool: &[Prim],
    s: &[Example],
    restarts: usize,
    steps: usize,
    max_len: usize,
    rng: &mut Random,
) -> Climb {
    let mut best = start.to_vec();
    let mut best_f = fitness(&best, s);
    let mut evals = 1;
    for r in 0..restarts {
        if best_f == 1.0 {
            break;
        }
        let rp = rand_prim(pool, rng); // consumed even on the first restart
        let mut cur = if r == 0 { start.to_vec() } else { vec![rp] };
        let mut cur_f = fitness(&cur, s);
        evals += 1;
        for _ in 0..steps {
            if cur_f == 1.0 {
                break;
            }
            let cand = mutate(&cur, pool, max_len, rng);
            let f = fitness(&cand, s);
            evals += 1;
            if f > cur_f || (f == cur_f && cand.len() <= cur.len()) {
                cur = cand;
                cur_f = f;
            }
        }
        let (cur, e) = shorten(cur, s, cur_f);
        evals += e;
        if cur_f > best_f || (cur_f == best_f && cur.len() < best.len()) {
            best = cur;
            best_f = cur_f;
        }
    }
    Climb {
        prog: best,
        fitness: best_f,
        evals,
    }
}
pub fn commutes_at(g: Gen, p: &[Prim], x: &Grid) -> bool {
    if let (Some(y), Some(x2)) = (eval(p, x), g.act(x)) {
        if let Some(y2) = g.act(&y) {
            return eval(p, &x2) == Some(y2);
        }
    }
    true
}
pub struct Realize {
    pub gens: Vec<Gen>,
    pub progs: Vec<Prog>,
    pub rejected: Vec<Gen>,
    pub repaired: Vec<Gen>,
    pub evals: usize,
}
fn realize(
    func: &[Gen],
    p0: Vec<Prog>,
    pool: &[Prim],
    t: &Task,
    cfg: &Config,
    rng: &mut Random,
) -> Realize {
    let mut r = Realize {
        gens: vec![],
        progs: p0,
        rejected: vec![],
        repaired: vec![],
        evals: 0,
    };
    for &g in func {
        let step: Vec<_> = t.train.iter().filter_map(|p| act_pair(g, p)).collect();
        let quick: Vec<_> = r
            .progs
            .iter()
            .filter(|p| fits_all(p, &step) && t.test.iter().all(|(x, _)| commutes_at(g, p, x)))
            .cloned()
            .collect();
        r.evals += r.progs.len();
        let mut trial = r.gens.clone();
        trial.push(g);
        let sample = sample_for(&trial, &t.train, cfg, rng);
        r.evals += quick.len();
        let keep: Vec<_> = quick.into_iter().filter(|p| fits_all(p, &sample)).collect();
        if !keep.is_empty() {
            r.gens = trial;
            r.progs = keep;
        } else if r.progs.is_empty() {
            r.rejected.push(g);
        } else {
            let cl = hill_climb_from(
                &r.progs[0],
                pool,
                &sample,
                1,
                cfg.repair_steps,
                cfg.max_len,
                rng,
            );
            r.evals += cl.evals;
            if cl.fitness == 1.0 && t.test.iter().all(|(x, _)| commutes_at(g, &cl.prog, x)) {
                r.gens = trial;
                r.progs = vec![cl.prog];
                r.repaired.push(g);
            } else {
                r.rejected.push(g);
            }
        }
    }
    r
}
#[derive(Debug)]
pub struct TestReport {
    pub by_symmetry: bool,
    pub sym_correct: Option<bool>,
    pub defined: bool,
    pub equivariant: Option<bool>,
    pub prediction: Option<Grid>,
    pub correct: bool,
    pub determined: Option<bool>,
}
pub fn equivariant_at(gens: &[Gen], p: &[Prim], x: &Grid) -> Option<bool> {
    let y = eval(p, x)?;
    if gens.is_empty() {
        return None;
    }
    Some(gens.iter().all(|g| match (g.act(x), g.act(&y)) {
        (Some(x2), Some(y2)) => eval(p, &x2) == Some(y2),
        _ => true,
    }))
}
pub fn test_report(
    c: &Closure,
    gens: &[Gen],
    p: &[Prim],
    progs: &[Prog],
    x: &Grid,
    y: &Grid,
) -> TestReport {
    let sym = c.out_of.get(x);
    let answer = eval(p, x);
    let defined = answer.is_some();
    let prediction = answer.or_else(|| sym.cloned());
    let answers: Vec<_> = progs.iter().filter_map(|p| eval(p, x)).collect();
    let determined = if answers.len() < 2 {
        None
    } else {
        Some(answers.iter().all(|a| *a == answers[0]))
    };
    TestReport {
        by_symmetry: sym.is_some(),
        sym_correct: sym.map(|a| a == y),
        defined,
        equivariant: equivariant_at(gens, p, x),
        correct: prediction.as_ref() == Some(y),
        prediction,
        determined,
    }
}
pub struct Result {
    pub id: String,
    pub realized: bool,
    pub func_gens: Vec<Gen>,
    pub gens: Vec<Gen>,
    pub rejected: Vec<Gen>,
    pub repaired: Vec<Gen>,
    pub closure_size: usize,
    pub capped: bool,
    pub gain: f64,
    pub outputs: usize,
    pub consistent_d: usize,
    pub consistent: usize,
    pub prog: Prog,
    pub fit_d: f64,
    pub fit_c: f64,
    pub evals: usize,
    pub tests: Vec<TestReport>,
}
impl Result {
    pub fn solved(&self) -> bool {
        self.tests.iter().all(|t| t.correct)
    }
    pub fn fitted(&self) -> bool {
        self.fit_d == 1.0
    }
}
pub fn run_task(t: &Task, cfg: &Config) -> Result {
    let mut rng = Random::new(cfg.seed);
    let (cands, pool) = t.candidates_pool();
    let (func_gens, _) = greedy_gens(&cands, &t.train, cfg.cap_greedy);
    let mut evals = 0;
    let mut climbed = vec![];
    let mut p0 = if cfg.enum_len == 0 {
        vec![]
    } else {
        dsl::enumerate(&pool, &t.train, cfg.enum_len)
    };
    if p0.is_empty() {
        let cl = hill_climb_from(
            &[],
            &pool,
            &t.train,
            cfg.restarts,
            cfg.steps,
            cfg.max_len,
            &mut rng,
        );
        evals += cl.evals;
        climbed = cl.prog;
        if cl.fitness == 1.0 {
            p0.push(climbed.clone());
        }
    }
    let consistent_d = p0.len();
    let realized = !p0.is_empty();
    let r = if !realized {
        Realize {
            gens: func_gens.clone(),
            progs: vec![],
            rejected: vec![],
            repaired: vec![],
            evals: 0,
        }
    } else if cfg.realize_filter {
        realize(&func_gens, p0, &pool, t, cfg, &mut rng)
    } else {
        let sample = sample_for(&func_gens, &t.train, cfg, &mut rng);
        let mut keep: Vec<_> = p0
            .iter()
            .filter(|p| fits_all(p, &sample))
            .cloned()
            .collect();
        let mut e = p0.len();
        if keep.is_empty() {
            let cl = hill_climb_from(
                &p0[0],
                &pool,
                &sample,
                cfg.restarts,
                cfg.steps,
                cfg.max_len,
                &mut rng,
            );
            e += cl.evals;
            keep.push(cl.prog);
        }
        Realize {
            gens: func_gens.clone(),
            progs: keep,
            rejected: vec![],
            repaired: vec![],
            evals: e,
        }
    };
    evals += r.evals;
    let prog = r
        .progs
        .iter()
        .min_by_key(|p| p.len())
        .cloned()
        .unwrap_or(climbed);
    let c = closure(&r.gens, &t.train, cfg.cap);
    let sample = fit_sample(&c, &r.gens, &t.train, cfg.fit_size, &mut rng);
    let tests = t
        .test
        .iter()
        .map(|(x, y)| test_report(&c, &r.gens, &prog, &r.progs, x, y))
        .collect();
    let outputs: FxMap<_, ()> = c.pairs.iter().map(|(_, y)| (y, ())).collect();
    Result {
        id: t.id.clone(),
        realized,
        func_gens,
        gens: r.gens,
        rejected: r.rejected,
        repaired: r.repaired,
        closure_size: c.size(),
        capped: c.capped,
        gain: (c.size() as f64).log2() - (t.train.len() as f64).log2(),
        outputs: outputs.len(),
        consistent_d,
        consistent: r.progs.len(),
        fit_d: fitness(&prog, &t.train),
        fit_c: fitness(&prog, &sample),
        prog,
        evals,
        tests,
    }
}
