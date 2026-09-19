import SymArc.Theory

/-!
# Operation-relative abstractions and concrete program families

Accepted structural results extracted from the experiments. No selection prior,
finite-search implementation, probability or generalisation guarantee is assumed.
Choice is used for existence of selectors; finite experiments enumerate them.
-/
namespace SymArc.Learning
open SymArc.Theory

variable {U Y Z W Op I : Type}

/-- The representation chooses a shared operation; the decoder may read `u`. -/
def run (E : Op → U → Option Y) (a : U → Z) (h : Z → Op) : U → Option Y :=
  fun u => E (h (a u)) u

/-- All labelled occurrences of key `z` permit this operation. -/
def Compatible (E : Op → U → Option Y) (a : U → Z)
    (D : U × Y → Prop) (z : Z) (o : Op) : Prop :=
  ∀ u y, D (u,y) → a u = z → E o u = some y

/-- Includes unused keys; applications have a nonempty operation alphabet. -/
def Coherent (E : Op → U → Option Y) (a : U → Z) (D : U × Y → Prop) : Prop :=
  ∀ z, ∃ o, Compatible E a D z o

/-- Per-class nonempty action intersections are exactly selector existence. -/
theorem coherent_iff_fitting (E : Op → U → Option Y) (a : U → Z)
    (D : U × Y → Prop) : Coherent E a D ↔ ∃ h, Fits (run E a h) D := by
  constructor
  · intro H
    let h : Z → Op := fun z => Classical.choose (H z)
    refine ⟨h, ?_⟩
    intro u y hy
    exact Classical.choose_spec (H (a u)) u y hy rfl
  · rintro ⟨h, H⟩ z
    refine ⟨h z, ?_⟩
    intro u y hy hz
    have hp := H u y hy
    simpa [run, hz] using hp

/-- Refinement is sufficient through an explicit recoding, not more labels. -/
theorem fitting_refinement (E : Op → U → Option Y) (a : U → Z) (b : U → W)
    (q : W → Z) (h : Z → Op) (D : U × Y → Prop)
    (hq : ∀ u, q (b u) = a u) (hp : Fits (run E a h) D) :
    Fits (run E b (fun w => h (q w))) D := by
  intro u y hy
  simpa [run, hq u] using hp u y hy

/-- An incompatible labelled subrelation cannot be placed in a single class. -/
theorem conflict_not_constant (E : Op → U → Option Y) (a : U → Z)
    (D K : U × Y → Prop) (h : Z → Op) (z : Z)
    (hKD : ∀ u y, K (u,y) → D (u,y))
    (hp : Fits (run E a h) D)
    (hc : ¬ ∃ o, ∀ u y, K (u,y) → E o u = some y) :
    ¬ (∀ u y, K (u,y) → a u = z) := by
  intro constant
  apply hc
  refine ⟨h z, ?_⟩
  intro u y hu
  simpa [run, constant u y hu] using hp u y (hKD u y hu)

/-- This is agreement about an output, with an explicit nonempty witness. -/
def Predicts (H : (U → Option Y) → Prop) (u : U) (y : Y) : Prop :=
  (∃ p, H p) ∧ ∀ p, H p → p u = some y

def TotalOn (p : U → Option Y) (Q : U → Prop) : Prop :=
  ∀ u, Q u → ∃ y, p u = some y

def Condition (H : (U → Option Y) → Prop) (Q : U → Prop) :
    (U → Option Y) → Prop := fun p => H p ∧ TotalOn p Q

theorem condition_preserves_member (H : (U → Option Y) → Prop)
    (Q : U → Prop) (p : U → Option Y) (hp : H p) (ht : TotalOn p Q) :
    Condition H Q p := ⟨hp, ht⟩

/-- Filtering preserves a determined answer only with a nonempty survivor set. -/
theorem predicts_of_subfamily (H K : (U → Option Y) → Prop)
    (u : U) (y : Y) (hk : ∀ p, K p → H p) (hne : ∃ p, K p)
    (hp : Predicts H u y) : Predicts K u y :=
  ⟨hne, fun p h => hp.2 p (hk p h)⟩

/-- Truth requires membership of the target; totality alone does not supply it. -/
theorem conditioned_prediction_correct (H : (U → Option Y) → Prop)
    (Q : U → Prop) (p : U → Option Y) (u : U) (y : Y)
    (hp : H p) (ht : TotalOn p Q) (h : Predicts (Condition H Q) u y) :
    p u = some y := h.2 p ⟨hp, ht⟩

/-- Conditioning distributes over a union of coupled program branches. -/
theorem condition_union (H : I → (U → Option Y) → Prop)
    (Q : U → Prop) (p : U → Option Y) :
    Condition (fun p => ∃ i, H i p) Q p ↔ ∃ i, Condition (H i) Q p := by
  constructor
  · rintro ⟨⟨i, hi⟩, ht⟩
    exact ⟨i, hi, ht⟩
  · rintro ⟨i, hi, ht⟩
    exact ⟨⟨i, hi⟩, ht⟩

/-- A concrete independent-key operation program. -/
def Allowed (V : Z → Op → Prop) (h : Z → Op) : Prop := ∀ z, V z (h z)

def KeyTotal (E : Op → U → Option Y) (a : U → Z)
    (Q : U → Prop) (z : Z) (o : Op) : Prop :=
  ∀ u, Q u → a u = z → ∃ y, E o u = some y

/-- Exact feasibility of independent choices under joint query definedness. -/
theorem independent_totality_iff (E : Op → U → Option Y) (a : U → Z)
    (Q : U → Prop) (V : Z → Op → Prop) :
    (∃ h, Allowed V h ∧ TotalOn (run E a h) Q) ↔
      ∀ z, ∃ o, V z o ∧ KeyTotal E a Q z o := by
  constructor
  · rintro ⟨h, hv, ht⟩ z
    refine ⟨h z, hv z, ?_⟩
    intro u hu hz
    simpa [run, hz] using ht u hu
  · intro H
    let h : Z → Op := fun z => Classical.choose (H z)
    refine ⟨h, ?_, ?_⟩
    · intro z
      exact (Classical.choose_spec (H z)).1
    · intro u hu
      exact (Classical.choose_spec (H (a u))).2 u hu rfl

/-- Exact per-key projection; the premise ensures the other keys can be filled. -/
theorem attainable_operation_iff (E : Op → U → Option Y) (a : U → Z)
    (Q : U → Prop) (V : Z → Op → Prop) (z : Z) (o : Op)
    (hne : ∃ h, Allowed V h ∧ TotalOn (run E a h) Q) :
    (∃ h, Allowed V h ∧ TotalOn (run E a h) Q ∧ h z = o) ↔
      V z o ∧ KeyTotal E a Q z o := by
  constructor
  · rintro ⟨h, hv, ht, hz⟩
    constructor
    · simpa [hz] using hv z
    · intro u hu ha
      simpa [run, ha, hz] using ht u hu
  · rintro ⟨hv, ht⟩
    classical
    obtain ⟨h₀, ha, hb⟩ := hne
    let h : Z → Op := fun w => if w = z then o else h₀ w
    refine ⟨h, ?_, ?_, ?_⟩
    · intro w
      by_cases hw : w = z
      · subst w
        simpa [h] using hv
      · simpa [h, hw] using ha w
    · intro u hu
      by_cases hz : a u = z
      · simpa [run, h, hz] using ht u hu hz
      · simpa [run, h, hz] using hb u hu
    · simp [h]

/-- A tiny counterexample to conditioning a coupled family after projection. -/
theorem projection_can_invent_program :
    (∀ z : Bool, ∃ h : Bool → Bool,
      ((∀ w, h w = false) ∨ (∀ w, h w = true)) ∧ h z = z) ∧
    ¬ (∃ h : Bool → Bool,
      ((∀ w, h w = false) ∨ (∀ w, h w = true)) ∧ ∀ z, h z = z) := by
  constructor
  · intro z
    cases z with
    | false => exact ⟨fun _ => false, Or.inl (fun _ => rfl), rfl⟩
    | true => exact ⟨fun _ => true, Or.inr (fun _ => rfl), rfl⟩
  · rintro ⟨h, hc, hi⟩
    cases hc with
    | inl hf => have : true = false := (hi true).symm.trans (hf true); cases this
    | inr ht => have : false = true := (hi false).symm.trans (ht false); cases this

/-- The same definition as a two-leaf domain-guarded program. -/
def orElse (p q : U → Option Y) : U → Option Y :=
  fun u => match p u with | some y => some y | none => q u

theorem fits_orElse (p q : U → Option Y) (D : U × Y → Prop)
    (hp : Fits p D) : Fits (orElse p q) D := by
  intro u y hy
  simp [orElse, hp u y hy]

/-- Unseen fallback labels remain arbitrary, even after totalizing the program. -/
theorem unseen_fallback (p : U → Option Y) (D : U × Y → Prop)
    (u : U) (hp : Fits p D) (hu : p u = none) (y : Y) :
    Fits (orElse p (fun _ => some y)) D ∧
    orElse p (fun _ => some y) u = some y := by
  exact ⟨fits_orElse p _ D hp, by simp [orElse, hu]⟩

/-! No canonical coarsest coherent quotient, even in a three-point problem. -/
namespace ThreePoint
inductive Point where | a | b | c deriving DecidableEq
open Point

def eval (o u : Point) : Option Bool := if o = u then none else some true
def data (uy : Point × Bool) : Prop := uy.2 = true

def first : Point → Bool | a => true | _ => false
def last : Point → Bool | c => true | _ => false

theorem first_fits : Fits (run eval first (fun b => if b then Point.b else Point.a)) data := by
  intro u y hy
  have : y = true := hy
  subst y
  cases u <;> simp [run, first, eval]

theorem last_fits : Fits (run eval last (fun b => if b then Point.a else Point.c)) data := by
  intro u y hy
  have : y = true := hy
  subst y
  cases u <;> simp [run, last, eval]

/-- Any quotient coarser than these two merges all points and is incoherent. -/
theorem no_common_coherent_coarsening {Z : Type} (c : Point → Z)
    (h₁ : c a = c b) (h₂ : c b = c Point.c) :
    ¬ ∃ h, Fits (run eval c h) data := by
  rintro ⟨h, hp⟩
  have all : ∀ u, c u = c a := by
    intro u
    cases u with
    | a => rfl
    | b => exact h₁.symm
    | c => exact (h₁.trans h₂).symm
  have failure := hp (h (c a)) true rfl
  simp [run, all, eval] at failure
end ThreePoint
end SymArc.Learning
