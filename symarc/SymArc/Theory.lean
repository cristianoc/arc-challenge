import Init
/-!
# The abstract theory, with proofs

This file is the mathematical model: generators `G` act partially on a carrier
`X`, with a functional grid model for the concrete involution lemmas. The sole
executable solver is Rust, in `src/`; its correspondence to this model is not proved.

Definitions, in order of appearance:

* `PAction`        – a partial action of generators on a carrier.
* `Reach`          – the closure of a set under a set of generators.
* `Functional`     – a relation between inputs and outputs is single-valued.
* `pairAct`        – the induced action on input–output pairs.
* `Fits`           – a partial program satisfies a specification.
* `EquivariantAt`  – a program commutes with a generator at one input.
* `Determined`     – all fitting programs agree at a test input.

Theorems:

* `Reach.mono_gens`, `Reach.mono_data` – closure is monotone.
* `Reach.trans`, `Reach.symm` – closure is transitive, and symmetric when
  generators are invertible, so it is an equivalence (orbit) relation.
* `functional_antitone` – admissibility is closed downward: if a generator set
  is admissible, so is every subset. This justifies greedy search.
* `fits_closure_of_equivariant`, `equivariant_of_fits_closure` – fitting the
  closure is the same as fitting the data and being equivariant along it.
* `determined_of_reach` – a test input inside the closure is determined.
-/

namespace SymArc.Theory

/-- A partial action of a set of generators `G` on a carrier `X`. `none` means
the generator does not apply (for example, a row swap on a grid with too few
rows). -/
def PAction (G X : Type) := G → X → Option X

variable {G X A B : Type}

/-- Everything reachable from `D` by finitely many generators satisfying `S`.
This is the concretisation of the abstract example: `γ(A(D))`. -/
inductive Reach (act : PAction G X) (S : G → Prop) (D : X → Prop) : X → Prop
  | base {x : X} : D x → Reach act S D x
  | step {g : G} {x y : X} : S g → Reach act S D x → act g x = some y → Reach act S D y

namespace Reach

variable {act : PAction G X} {S S' : G → Prop} {D D' : X → Prop}

theorem mono_gens (hS : ∀ g, S g → S' g) {x : X} (h : Reach act S D x) :
    Reach act S' D x := by
  induction h with
  | base hx => exact .base hx
  | step hg _ hact ih => exact .step (hS _ hg) ih hact

theorem mono_data (hD : ∀ x, D x → D' x) {x : X} (h : Reach act S D x) :
    Reach act S D' x := by
  induction h with
  | base hx => exact .base (hD _ hx)
  | step hg _ hact ih => exact .step hg ih hact

/-- Closing twice is closing once. -/
theorem trans {x y : X} (hx : Reach act S D x)
    (hy : Reach act S (fun z => z = x) y) : Reach act S D y := by
  induction hy with
  | base h => exact h ▸ hx
  | step hg _ hact ih => exact .step hg ih hact

/-- If every generator in `S` has an inverse in `S`, reachability between
single points is symmetric. Together with `trans` this makes
`fun x y => Reach act S (· = x) y` an equivalence relation whose classes are
the orbits. -/
theorem symm
    (hinv : ∀ g, S g → ∃ g', S g' ∧ ∀ x y, act g x = some y → act g' y = some x)
    {x y : X} (h : Reach act S (fun z => z = x) y) :
    Reach act S (fun z => z = y) x := by
  induction h with
  | base h => exact .base h.symm
  | @step g m y hg _ hact ih =>
    obtain ⟨g', hg', hinv'⟩ := hinv g hg
    have back : Reach act S (fun z => z = y) m :=
      .step hg' (.base rfl) (hinv' m y hact)
    exact trans back ih

end Reach

/-- A specification (a set of input–output pairs) is functional when no input
has two outputs. For a finite relation with full-support probability this is `H(Y | X) = 0`. -/
def Functional (R : A × B → Prop) : Prop :=
  ∀ a b b', R (a, b) → R (a, b') → b = b'

/-- Admissibility is downward closed: dropping generators cannot create a
conflict. Hence a greedy search that adds generators while the closure stays
functional never needs to backtrack. -/
theorem functional_antitone {act : PAction G (A × B)} {S S' : G → Prop}
    {D : A × B → Prop} (hS : ∀ g, S' g → S g)
    (h : Functional (Reach act S D)) : Functional (Reach act S' D) :=
  fun a b b' h1 h2 => h a b b' (h1.mono_gens hS) (h2.mono_gens hS)

/-- The induced action on pairs: apply the same generator to input and output.
The output may live in a different carrier (a different grid shape), so the two
component actions are separate representations of the same generator. -/
def pairAct (actA : PAction G A) (actB : PAction G B) : PAction G (A × B) :=
  fun g p =>
    match actA g p.1, actB g p.2 with
    | some a, some b => some (a, b)
    | _, _ => none

theorem pairAct_some {actA : PAction G A} {actB : PAction G B} {g : G}
    {a a' : A} {b b' : B} :
    pairAct actA actB g (a, b) = some (a', b') ↔
      actA g a = some a' ∧ actB g b = some b' := by
  unfold pairAct
  cases hA : actA g a <;> cases hB : actB g b <;> simp

/-- A partial program `p` fits a specification `R`. -/
def Fits (p : A → Option B) (R : A × B → Prop) : Prop :=
  ∀ a b, R (a, b) → p a = some b

/-- `p` is equivariant for `g` at `x`: if `g` moves `x` to `x'` and moves
`p x` to `y'`, then `p x' = y'`. -/
def EquivariantAt (actA : PAction G A) (actB : PAction G B)
    (p : A → Option B) (g : G) (x : A) : Prop :=
  ∀ x' y y', actA g x = some x' → p x = some y → actB g y = some y' →
    p x' = some y'

/-- Fitting the data and being equivariant along the closure implies fitting
the closure. -/
theorem fits_closure_of_equivariant {actA : PAction G A} {actB : PAction G B}
    {S : G → Prop} {D : A × B → Prop} {p : A → Option B}
    (hD : Fits p D)
    (heq : ∀ g, S g → ∀ x, (∃ y, Reach (pairAct actA actB) S D (x, y)) →
      EquivariantAt actA actB p g x) :
    Fits p (Reach (pairAct actA actB) S D) := by
  intro a b h
  suffices H : ∀ q, Reach (pairAct actA actB) S D q → p q.1 = some q.2 from
    H (a, b) h
  intro q hq
  induction hq with
  | @base q hx => exact hD q.1 q.2 hx
  | @step g x y hg hr hact ih =>
    obtain ⟨x1, x2⟩ := x
    obtain ⟨y1, y2⟩ := y
    rw [pairAct_some] at hact
    exact heq g hg x1 ⟨x2, hr⟩ y1 x2 y2 hact.1 ih hact.2

/-- Conversely, a program that fits the closure is equivariant at every input
in the closure. This is the precise form of "fitting `C(D)` is equivariance". -/
theorem equivariant_of_fits_closure {actA : PAction G A} {actB : PAction G B}
    {S : G → Prop} {D : A × B → Prop} {p : A → Option B}
    (hp : Fits p (Reach (pairAct actA actB) S D)) :
    ∀ g, S g → ∀ x y, Reach (pairAct actA actB) S D (x, y) →
      EquivariantAt actA actB p g x := by
  intro g hg x y hr x' y0 y' hx hpx hy
  have e : y0 = y := Option.some.inj (hpx.symm.trans (hp x y hr))
  subst e
  exact hp x' y' (.step hg hr (pairAct_some.mpr ⟨hx, hy⟩))

/-- The test input `x` is determined by specification `R` within the program
class `P`: all programs in `P` that fit `R` and are defined at `x` agree. -/
def Determined (P : (A → Option B) → Prop) (R : A × B → Prop) (x : A) : Prop :=
  ∀ p q, P p → P q → Fits p R → Fits q R →
    ∀ y y', p x = some y → q x = some y' → y = y'

/-- A test input that lies in the input projection of the specification is
determined, whatever the program class. This is generalisation by symmetry
alone. -/
theorem determined_of_reach {P : (A → Option B) → Prop} {R : A × B → Prop}
    {x : A} {y : B} (h : R (x, y)) : Determined P R x := by
  intro p q _ _ hp hq y1 y2 h1 h2
  have e1 : y = y1 := Option.some.inj ((hp x y h).symm.trans h1)
  have e2 : y = y2 := Option.some.inj ((hq x y h).symm.trans h2)
  exact e1.symm.trans e2

/-! ## Invertibility of the concrete generators

Proved on a functional model of grids (`Fin h → Fin w → Colour`) where the
index arithmetic is transparent. The Rust implementation is separate; its correspondence to this model is
not proved here. -/

/-- Functional grids, for proofs. -/
def FGrid (h w : Nat) := Fin h → Fin w → Nat

namespace FGrid

variable {h w : Nat}

def flipH (g : FGrid h w) : FGrid h w := fun i j => g i j.rev
def flipV (g : FGrid h w) : FGrid h w := fun i j => g i.rev j
def transpose (g : FGrid h w) : FGrid w h := fun i j => g j i
/-- Clockwise rotation: `new (i, j) = old (h - 1 - j, i)`. -/
def rot90 (g : FGrid h w) : FGrid w h := fun i j => g j.rev i
/-- Anticlockwise rotation: `new (i, j) = old (j, w - 1 - i)`. -/
def rot270 (g : FGrid h w) : FGrid w h := fun i j => g j i.rev

theorem flipH_flipH (g : FGrid h w) : flipH (flipH g) = g := by
  funext i j; simp [flipH, Fin.rev_rev]

theorem flipV_flipV (g : FGrid h w) : flipV (flipV g) = g := by
  funext i j; simp [flipV, Fin.rev_rev]

theorem transpose_transpose (g : FGrid h w) : transpose (transpose g) = g := rfl

theorem rot270_rot90 (g : FGrid h w) : rot270 (rot90 g) = g := by
  funext i j; simp [rot90, rot270, Fin.rev_rev]

theorem rot90_rot270 (g : FGrid h w) : rot90 (rot270 g) = g := by
  funext i j; simp [rot90, rot270, Fin.rev_rev]

end FGrid

/-- Transposition of two colours in the mathematical grid model. -/
def swapColourFn (a b c : Nat) : Nat :=
  if c == a then b else if c == b then a else c

/-- Swapping two colours is an involution. -/
theorem swapColourFn_involutive (a b c : Nat) :
    swapColourFn a b (swapColourFn a b c) = c := by
  unfold swapColourFn
  by_cases h1 : c = a
  · subst h1; by_cases h2 : b = c <;> simp [h2]
  · by_cases h2 : c = b
    · subst h2; simp [h1]
    · simp [h1, h2]

end SymArc.Theory
