use crate::relations::{apply, Relation};
use symarc::{dsl::Prim, grid::Grid};
use symarc_exp_003_objects::{
    entropy,
    objects::{Scene, Segment, Select},
};
#[derive(Clone)]
pub struct Case {
    pub seed: usize,
    pub object_truth: bool,
    pub initial: Grid,
    pub menu: Vec<Grid>,
    pub test: Grid,
    pub source_colour: u8,
    pub initial_marker: u8,
    pub informative: usize,
}
fn scene(seed: usize, k: usize, marker: u8) -> Grid {
    let palette: Vec<_> = (0..9).map(|j| ((seed + j) % 9 + 1) as u8).collect();
    let source = palette[0];
    let distractor = palette[4];
    let h = 2 + (seed + k) % 2;
    let w = 2 + (seed / 2 + k) % 2;
    let row = 1 + (seed + k) % 3;
    let col = 1 + (seed / 3 + k) % 3;
    let mut g = Grid::of_fn(12, 12, |_, _| 0);
    for r in row..row + h {
        for c in col..col + w {
            g.cells[r * 12 + c] = source;
        }
    }
    g.cells[(8 + (seed + k) % 3) * 12 + 7 + (seed + 2 * k) % 4] = marker;
    if seed % 2 == 1 {
        g.cells[1 * 12 + 8] = distractor;
        g.cells[1 * 12 + 9] = distractor;
    }
    g
}
pub fn make(seed: usize, object_truth: bool) -> Case {
    let colour = |j: usize| ((seed + j) % 9 + 1) as u8;
    let source_colour = colour(0);
    let initial_marker = colour(1);
    let informative = seed % 4;
    let initial = scene(seed, 0, initial_marker);
    let menu = (0..4)
        .map(|i| {
            scene(
                seed,
                i + 1,
                if i == informative {
                    colour(2)
                } else {
                    initial_marker
                },
            )
        })
        .collect();
    let test = scene(seed, 5, colour(3));
    Case {
        seed,
        object_truth,
        initial,
        menu,
        test,
        source_colour,
        initial_marker,
        informative,
    }
}
// Learner predictions contain no truth label and no teacher output.
pub fn predict(g: &Grid, source_colour: u8, constant: u8, object_rule: bool) -> Option<Grid> {
    if !object_rule {
        return Prim::Recolour(source_colour, constant).eval(g);
    }
    let s = Scene::parse(
        g,
        Segment {
            modal_background: false,
            diagonal: false,
            monochrome: true,
        },
    );
    let sources = Select::Largest.indices(&s);
    let refs = Select::Smallest.indices(&s);
    if sources.len() != 1 || refs.len() != 1 {
        return None;
    }
    apply(&s, sources[0], refs[0], Relation::Colour)
}
// Independent teacher construction uses the declared source and marker cells,
// not object extraction or the relational interpreter.
pub fn teacher(case: &Case, g: &Grid, k: usize) -> Grid {
    let marker = g.get(8 + (case.seed + k) % 3, 7 + (case.seed + 2 * k) % 4);
    let colour = if case.object_truth {
        marker
    } else {
        case.initial_marker
    };
    g.map(|v| if v == case.source_colour { colour } else { v })
}
pub fn information(g: &Grid, source: u8, constant: u8) -> f64 {
    let a = predict(g, source, constant, false);
    let b = predict(g, source, constant, true);
    assert!(a.is_some() && b.is_some());
    if a == b {
        0.
    } else {
        entropy([0.5, 0.5].into_iter())
    }
}
pub fn choose(case: &Case) -> usize {
    let mut best = 0;
    let mut gain = -1.;
    for (i, q) in case.menu.iter().enumerate() {
        let ig = information(q, case.source_colour, case.initial_marker);
        if ig > gain + 1e-12 {
            best = i;
            gain = ig;
        }
    }
    best
}
#[derive(Clone, Copy, Default)]
pub struct Outcome {
    pub correct: f64,
    pub truth_mass: f64,
    pub brier: f64,
    pub h: f64,
    pub identified: f64,
}
pub fn evaluate(case: &Case, query: Option<usize>) -> Outcome {
    let mut live = vec![false, true];
    if let Some(i) = query {
        let answer = teacher(case, &case.menu[i], i + 1);
        live.retain(|&rule| {
            predict(&case.menu[i], case.source_colour, case.initial_marker, rule).as_ref()
                == Some(&answer)
        });
    }
    assert!(!live.is_empty());
    let mass = 1. / live.len() as f64;
    let expected = teacher(case, &case.test, 5);
    let selected = predict(&case.test, case.source_colour, case.initial_marker, live[0]);
    Outcome {
        correct: f64::from(selected.as_ref() == Some(&expected)),
        truth_mass: mass,
        brier: if live.len() == 1 { 0. } else { 0.5 },
        h: (live.len() as f64).log2(),
        identified: f64::from(live.len() == 1),
    }
}
pub fn cases() -> Vec<Case> {
    (0..32)
        .flat_map(|seed| [make(seed, false), make(seed, true)])
        .collect()
}
pub fn checked_cases() -> Vec<Case> {
    let cs = cases();
    for c in &cs {
        let initial = teacher(c, &c.initial, 0);
        for rule in [false, true] {
            assert_eq!(
                predict(&c.initial, c.source_colour, c.initial_marker, rule),
                Some(initial.clone())
            );
        }
        for (i, q) in c.menu.iter().enumerate() {
            assert_eq!(
                predict(q, c.source_colour, c.initial_marker, c.object_truth),
                Some(teacher(c, q, i + 1))
            );
            assert_eq!(
                information(q, c.source_colour, c.initial_marker),
                if i == c.informative { 1. } else { 0. }
            );
        }
        assert_eq!(
            predict(&c.test, c.source_colour, c.initial_marker, c.object_truth),
            Some(teacher(c, &c.test, 5))
        );
        assert_ne!(
            predict(&c.test, c.source_colour, c.initial_marker, false),
            predict(&c.test, c.source_colour, c.initial_marker, true)
        );
    }
    cs
}
pub fn report(cs: &[Case]) -> String {
    let names = [
        "no extra evidence",
        "uninformative example",
        "informative example (oracle query)",
        "first menu example",
        "random query (exact expectation)",
        "maximum expected information gain",
    ];
    let mut s=String::from("## Controlled abstraction identification\n\n64 constructed cases: 32 scene seeds × two equally represented rules. Both models fit the initial example and disagree at test. The hypothesis prior is 1/2 per rule (initial entropy 1 bit). One of four offered example inputs distinguishes them; three do not. Both models are defined on every input.\n\n| Query policy | Expected correct / 64 | Accuracy | Rule identified | Mean true-rule posterior mass | Mean posterior H (bits) | Mean Brier loss |\n|---|---:|---:|---:|---:|---:|---:|\n");
    for (policy, name) in names.iter().enumerate() {
        let mut sum = Outcome::default();
        for c in cs {
            let qs: Vec<(Option<usize>, f64)> = match policy {
                0 => vec![(None, 1.)],
                1 => vec![(Some((c.informative + 1) % 4), 1.)],
                2 => vec![(Some(c.informative), 1.)],
                3 => vec![(Some(0), 1.)],
                4 => (0..4).map(|i| (Some(i), 0.25)).collect(),
                _ => vec![(Some(choose(c)), 1.)],
            };
            for (q, w) in qs {
                let r = evaluate(c, q);
                sum.correct += w * r.correct;
                sum.truth_mass += w * r.truth_mass;
                sum.brier += w * r.brier;
                sum.h += w * r.h;
                sum.identified += w * r.identified;
            }
        }
        let n = cs.len() as f64;
        s += &format!(
            "| {name} | {:.1}/{} | {:.2}% | {:.1}/{} | {:.3} | {:.3} | {:.3} |\n",
            sum.correct,
            cs.len(),
            100. * sum.correct / n,
            sum.identified,
            cs.len(),
            sum.truth_mass / n,
            sum.h / n,
            sum.brier / n
        );
    }
    s+"\nEvery individual deterministic hypothesis has zero answer entropy, so choosing the most certain hypothesis cannot separate these rules. Information gain instead compares their disagreement on candidate examples, without seeing the answers. One bit of discriminating evidence resolves one bit of uncertainty.\n\nThese outcomes follow from the deliberately constructed two-model setting. They verify the mechanism and demonstrate identifiable versus unidentifiable evidence; they do NOT establish automatic representation discovery, a realistic prior, a generally optimal ARC strategy, or access to additional labelled examples in ARC. Random-query results are exact expectations, not sampled estimates.\n"
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn balanced_valid_generator() {
        let cs = checked_cases();
        assert_eq!(cs.len(), 64);
        assert_eq!(cs.iter().filter(|c| c.object_truth).count(), 32);
    }
    #[test]
    fn query_choice_does_not_use_labels() {
        for seed in 0..32 {
            let a = make(seed, false);
            let b = make(seed, true);
            assert_eq!(choose(&a), choose(&b));
            assert_eq!(choose(&a), seed % 4);
        }
    }
    #[test]
    fn evidence_identifiability_and_chance() {
        let cs = checked_cases();
        let mut none = 0.;
        let mut first = 0.;
        let mut gain = 0.;
        let mut random = 0.;
        for c in &cs {
            none += evaluate(c, None).correct;
            first += evaluate(c, Some(0)).correct;
            gain += evaluate(c, Some(choose(c))).correct;
            random += (0..4)
                .map(|i| evaluate(c, Some(i)).correct / 4.)
                .sum::<f64>();
            assert_eq!(evaluate(c, Some((c.informative + 1) % 4)).h, 1.);
            assert_eq!(evaluate(c, Some(c.informative)).h, 0.);
        }
        assert_eq!((none, first, gain, random), (32., 40., 64., 40.));
    }
}
