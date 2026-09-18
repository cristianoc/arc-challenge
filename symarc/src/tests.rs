use crate::{
    dsl::{self, Prim},
    grid::{self, Gen, Grid},
    random::Random,
    search, Task,
};
fn grid(rows: &[&[u8]]) -> Grid {
    Grid::of_rows(&rows.iter().map(|r| r.to_vec()).collect::<Vec<_>>())
}

#[test]
fn functionality_detects_stabilizer_conflict() {
    let x = grid(&[&[0, 0, 0], &[0, 7, 0], &[0, 0, 0]]);
    let y = grid(&[&[0, 0, 0], &[7, 7, 7], &[0, 0, 0]]);
    let d = vec![(x, y)];
    assert!(!grid::closure(&[Gen::Transpose], &d, 100).functional);
    let (gens, c) = grid::greedy_gens(&[Gen::Transpose, Gen::FlipH], &d, 100);
    assert_eq!(gens, vec![Gen::FlipH]);
    assert!(c.functional);
}
#[test]
fn closure_deduplicates_inputs_and_detects_conflicting_data() {
    let x = grid(&[&[1, 0]]);
    let y = grid(&[&[2, 0]]);
    let c = grid::closure(&[], &[(x.clone(), x.clone()), (x.clone(), y)], 100);
    assert_eq!(c.size(), 1);
    assert!(!c.functional);
    let c = grid::closure(&[Gen::FlipH], &[(x.clone(), x)], 100);
    assert_eq!(c.size(), 2);
    assert!(!c.capped);
    assert!(c.functional);
}
#[test]
fn capped_samples_keep_data_and_one_step_images() {
    let d = vec![(grid(&[&[1, 0, 0]]), grid(&[&[2, 0, 0]]))];
    let gens = [Gen::CycleCols];
    let c = grid::closure(&gens, &d, 1);
    assert!(c.capped);
    let s = search::fit_sample(&c, &gens, &d, 0, &mut Random::new(0));
    assert_eq!(s.len(), 2);
    assert_eq!(s[0], d[0]);
    assert_eq!(
        s[1],
        (gens[0].act(&d[0].0).unwrap(), gens[0].act(&d[0].1).unwrap())
    );
}
#[test]
fn complete_samples_do_not_depend_on_seed() {
    let d = vec![(grid(&[&[1, 0]]), grid(&[&[2, 0]]))];
    let gens = [Gen::FlipH];
    let c = grid::closure(&gens, &d, 100);
    assert_eq!(
        search::fit_sample(&c, &gens, &d, 64, &mut Random::new(42)),
        c.pairs
    );
}
#[test]
fn undefined_and_disagreement_are_separate_from_symmetry_prediction() {
    let x = grid(&[&[1]]);
    let y = grid(&[&[2]]);
    let c = grid::closure(&[], &[(x.clone(), y.clone())], 100);
    let p = vec![Prim::LeftHalf];
    let r = search::test_report(&c, &[], &p, &[vec![], vec![Prim::Recolour(1, 2)]], &x, &y);
    assert!(!r.defined);
    assert!(r.correct);
    assert!(r.by_symmetry);
    assert_eq!(r.equivariant, None);
    assert_eq!(r.determined, Some(false));
    assert_eq!(r.prediction, Some(y));
    assert!(search::commutes_at(Gen::SwapRows(0, 1), &[], &x));
}
#[test]
fn programs_obey_intermediate_size_bound() {
    let x = Grid::of_fn(20, 20, |_, _| 1);
    assert!(search::eval(&[Prim::Scale(2), Prim::CropBBox], &x).is_none());
    assert_eq!(search::eval(&[], &x), Some(x));
}
#[test]
fn enumeration_matches_direct_evaluation_and_order() {
    let x = grid(&[&[1, 0], &[0, 0]]);
    let y = x.flip_h();
    let d = vec![(x, y)];
    let pool = [Prim::FlipH, Prim::FlipV, Prim::Rot90];
    let mut expected = vec![];
    for p in std::iter::once(vec![])
        .chain(pool.iter().map(|&p| vec![p]))
        .chain(
            pool.iter()
                .flat_map(|&a| pool.iter().map(move |&b| vec![a, b])),
        )
    {
        if search::fits_all(&p, &d) {
            expected.push(p);
        }
    }
    assert_eq!(dsl::enumerate(&pool, &d, 2), expected);
}
#[test]
fn held_out_labels_do_not_affect_search() {
    let x = grid(&[&[0, 1], &[0, 0]]);
    let y = x.flip_h();
    let mut t = Task {
        id: "test".into(),
        train: vec![(x.clone(), y.clone())],
        test: vec![(x, y)],
    };
    let cfg = search::Config {
        cap: 100,
        cap_greedy: 100,
        repair_steps: 5,
        ..Default::default()
    };
    let a = search::run_task(&t, &cfg);
    t.test[0].1 = grid(&[&[9, 9, 9], &[9, 9, 9], &[9, 9, 9]]);
    let b = search::run_task(&t, &cfg);
    assert_eq!(a.gens, b.gens);
    assert_eq!(a.prog, b.prog);
    assert_eq!(a.evals, b.evals);
    assert_eq!(a.tests[0].prediction, b.tests[0].prediction);
}

#[test]
fn report_rounds_halfway_fitness_up() {
    assert_eq!(crate::report::number(0.125), "0.13");
    assert_eq!(crate::report::number(1.0), "1.00");
}

#[test]
fn report_uses_distinct_task_and_test_grid_denominators() {
    let x = grid(&[&[1]]);
    let make_task = |id: &str, outputs: Vec<Grid>| Task {
        id: id.into(),
        train: vec![(x.clone(), x.clone())],
        test: outputs.into_iter().map(|y| (x.clone(), y)).collect(),
    };
    let cfg = search::Config {
        enum_len: 0,
        restarts: 0,
        repair_steps: 0,
        cap: 20,
        ..Default::default()
    };
    let a = search::run_task(&make_task("a", vec![x.clone(), grid(&[&[2]])]), &cfg);
    let b = search::run_task(&make_task("b", vec![x.clone()]), &cfg);
    let s = crate::report::summary(
        &[a, b],
        &["training".into(), "evaluation".into()],
        &cfg,
        "test",
        12,
        1.0,
    );
    assert!(s.contains(
        "| **Task exact-match accuracy (primary)** | 0/1 (0.00%) | 1/1 (100.00%) | 1/2 (50.00%) |"
    ));
    assert!(s.contains(
        "| Test-grid exact-match accuracy | 1/2 (50.00%) | 1/1 (100.00%) | 2/3 (66.67%) |"
    ));
    assert!(s.contains(
        "| Prediction coverage (test grids) | 2/2 (100.00%) | 1/1 (100.00%) | 3/3 (100.00%) |"
    ));
}

#[test]
fn empty_report_has_no_division_by_zero_or_invented_accuracy() {
    let s = crate::report::summary(&[], &[], &search::Config::default(), "empty", 12, 0.0);
    assert!(s.contains("| **Task exact-match accuracy (primary)** | n/a (0/0) |"));
    assert!(!s.contains("NaN"));
}
