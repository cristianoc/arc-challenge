# InPaintingARC: A Vertical Subset of ARC-AGI

**InPaintingARC** is a specialized variant of the ARC (Abstraction and Reasoning Corpus) challenge. It focuses on a narrower variety of puzzles while maintaining the complexity of individual tasks found in the broader ARC-AGI domain. InPaintingARC emphasizes deep understanding of spatial patterns, transformations, and frame invariants, making it an ideal platform for exploring advanced reasoning techniques such as **bi-abductive inference** and the application of the **frame rule**.

---

## InPainting Tasks

An inpainting task consists of grids where the input is the same as the output except for a number of cells covered with a specific color. The outputs contain a recognizable patterns so that based on that it is possible to reconstruct the original grid from just the input.




## Solution Approach

The solution to InPaintingARC leverages **bi-abductive inference** to derive transformation specifications from given examples. This involves formulating **spatial specifications** that are composable and capturing key puzzle elements through formal **predicates**. The approach focuses on both inferring the transformations (`Spec`) and identifying the invariant parts of the grid (`Frame` or `R`) that remain unaffected.

### Bi-Abductive Inference

**Bi-abduction** is a reasoning process that simultaneously infers both the missing specifications and the frame conditions required to explain observed behavior. In the context of grid transformations, bi-abduction aims to derive:

- **Specification (`Spec`)**: The minimal set of rules that explain the transformations in the examples.

#### General Shape of Bi-Abduction

The general form of bi-abduction can be expressed as:


$$
[\text{Spec}] \quad \text{Examples} \quad \vdash \quad \text{Input} \rightarrow [\text{Output}]
$$

- `[Spec]`: The specification inferred from the examples.
- `Examples`: Given input-output pairs demonstrating the transformation.
- `Input`: Gven grid to which the specification will be applied.
- `[Output]`: The inferred resulting grid after applying the specification to the input.


### Notation

#### Grid and Cells

- **Grid (`G`)**: A finite set of colored cells.
- **Position (`X`)**: Identified by its coordinates \( X = (x, y) \).

#### Color Predicate

- `G[X] = C`: Predicate indicating that cell `X` in grid `G` has color `C`.

#### Pattern Predicate

- `Pattern(X)`: A predicate defining a pattern for cell `X`. For example, in a checkerboard pattern:

$$
\text{Pattern}(x,y) \iff G[x,y] =
\begin{array}{ll}
black  & \text{if} \quad (x + y) \bmod 2 = 0 \\
red  & \text{otherwise}
\end{array}
$$

### Example

![bi-abduction-ex1-a](images/bi-abduction-ex1-a.png)



From the examples one can infer

$$
Spec \iff \text{in}.\text{rot90()} = \text{out}
$$

So the final derivation is

![bi-abduction-ex1-b](images/bi-abduction-ex1-b.png)


-----

TODO

#### The Frame Rule

The **frame rule** allows us to extend the specification to include parts of the grid that remain unaffected by the transformation. It is formulated as:

$$
\frac{[\text{Spec}] \quad \text{Examples} \quad \vdash \quad \text{Input} \rightarrow [\text{Output}]}{[\text{Spec} * R] \quad \text{Examples} * R \quad \vdash \quad \text{Input} * R \rightarrow [\text{Output} * R]}
$$

- `Spec * R`: The combined specification, where `Spec` and `R` are combined using the separating conjunction `*`, indicating they operate on disjoint parts of the grid.
- `Examples * R`, `Input * R`, `Output * R`: The extension of examples, input, and output by including the frame `R`.

The **separating conjunction** `*` asserts that the domains of `Spec` and `R` are disjoint, ensuring that the transformation and the frame do not interfere with each other.

