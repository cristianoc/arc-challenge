# Review of 013-minimal-sufficient-distinction

Status and disposition are recorded in the experiment ledger. This note corrects
the earlier proposed protocol; it is not a new empirical result.

The original runner hard-coded all three `positive_support_*` booleans to
`False`. Its call to `012.case` also computed test scores, despite the claim that
the check did not score tests. The runner was never executed as a registered
scientific measurement. It is retired rather than preserving a misleading
check. The separate `013-constant-lifting` experiment is unaffected.

There is a deeper design problem. Two fixed programs selected because both fit
all demonstrations cannot disagree on an observed demonstration label. Calling
that absence of disagreement “no positive evidence” is circular. It cannot show
that ARC generally lacks evidence for useful abstraction, or that no principled
preference between the programs is possible under an explicit prior.

What survives is the elementary factorisation criterion. For a representation
$a:X\to A$, a predictor $p:X\to Y$ factors through the image of $a$ exactly when

$$
a(x)=a(x')\quad\Longrightarrow\quad p(x)=p(x').
$$

A pair with equal statistics and different proposed repaired decisions refutes
that statistic's sufficiency for the proposed repair. An independently labelled
pair can refute sufficiency for the observed task data. An unlabelled synthetic
pair does not choose its own correct label. The rank case in 012 also remains a
change in use of retained information, not automatically a coarse-kernel error.

To measure prediction rather than restate fitting, 014 freezes a learner and
relearns it after withholding whole demonstrations. It also measures actual
local transports and their test contradictions across the pinned corpus. See
[014's findings](../014-cross-demonstration-transport/README.md). Neither study
provides a distribution-independent or prior-free definition of generalisation.
