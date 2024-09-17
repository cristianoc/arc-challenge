# InPaintingARC: A Vertical Subset of ARC-AGI

**InPaintingARC** is a specialized variant of the ARC (Abstraction and Reasoning Corpus) challenge. It focuses on a narrower variety of puzzles while maintaining the complexity of individual tasks found in the broader ARC-AGI domain. InPaintingARC emphasizes deep understanding of spatial patterns, transformations, and frame invariants, making it an ideal platform for exploring advanced reasoning techniques such as **bi-abductive inference** and the application of the **frame rule**.

---

## Solution Approach

The solution to InPaintingARC leverages **bi-abductive inference** to derive transformation specifications from given examples. This involves formulating **spatial specifications** that are composable and capturing key puzzle elements through formal **predicates**. The approach focuses on both inferring the transformations (`Spec`) and identifying the invariant parts of the grid (`Frame` or `R`) that remain unaffected.

### Bi-Abductive Inference

**Bi-abduction** is a reasoning process that simultaneously infers both the missing specifications and the frame conditions required to explain observed behavior. In the context of grid transformations, bi-abduction aims to derive:

- **Specification (`Spec`)**: The minimal set of rules that explain the transformations in the examples.
- **Frame (`R`)**: The part of the grid that remains unchanged during the transformation.

#### General Shape of Bi-Abduction

The general form of bi-abduction can be expressed as:


$$
[\text{Spec}] \quad \text{Examples} \quad \vdash \quad \text{Input} \quad \rightarrow \quad [\text{Output}]
$$

- **`[Spec]`**: The specification inferred from the examples.
- **`Examples`**: Input-output pairs demonstrating the transformation.
- **`Input`**: The grid to which the specification will be applied.
- **`Output`**: The resulting grid after applying the specification to the input.

#### The Frame Rule

The **frame rule** allows us to extend the specification to include parts of the grid that remain unaffected by the transformation. It is formulated as:

$$
\frac{[\text{Spec}] \quad \text{Examples} \quad \vdash \quad \text{Input} \quad \rightarrow \quad [\text{Output}]}{[\text{Spec} * R] \quad \text{Examples} * R \quad \vdash \quad \text{Input} * R \quad \rightarrow \quad [\text{Output} * R]}
$$

- **`[Spec * R]`**: The combined specification, where `Spec` and `R` are combined using the separating conjunction `*`, indicating they operate on disjoint parts of the grid.
- **`Examples * R`**, **`Input * R`**, **`[Output * R]`**: The extension of examples, input, and output by including the frame `R`.

The **separating conjunction** `*` asserts that the domains of `Spec` and `R` are disjoint, ensuring that the transformation and the frame do not interfere with each other.

---

## Detailed Methodology

### 1. Formal Definitions

#### Grid and Cells

- **Grid (`G`)**: A finite two-dimensional array of cells.
- **Cell (`X`)**: Identified by its coordinates \( X = (x, y) \).

#### Color Predicate

- **`Color_G(C, X)`**: Predicate indicating that cell `X` in grid `G` has color `C`.

#### Pattern Predicate

- **`Pattern(X)`**: A predicate defining a pattern for cell `X`. For example, in a checkerboard pattern:

$$\text{Pattern}(x,y) \iff (x + y) \bmod 2 = 0$$

#### Frame Predicate

- **`Frame(X)`**: Indicates that cell `X` is part of the frame and should remain unchanged.
$$\text{Frame}(x,y) \iff y \in \{0, 1, 2\}$$

#### Specification (`Spec`)

- **`Spec(X)`**: Defines the transformation for cells not in the frame.

$`
\text{Spec}(X) \iff \neg \text{Frame}(X) \implies \left\{
\begin{array}{ll}
\text{Pattern}(X) & \implies \text{Color}_{G_{\text{out}}}(X) = C_1 \\
\neg \text{Pattern}(X) & \implies \text{Color}_{G_{\text{out}}}(X) = C_2
\end{array}
\right\}
`$

#### Frame (`R`)

- **`R(X)`**: Ensures that cells in the frame remain unchanged.

$$R(X) \iff \text{Frame}(X) \implies \text{Color}_{G_{\text{out}}}(X) = \text{Color}_{G_{\text{in}}}(X)$$

#### Separating Conjunction (`*`)

- Combines specifications or grids operating on disjoint domains.

---

### 2. Derivation Using Bi-Abductive Inference

TODO
---


## Example Workflow

### Problem Statement

- **Input Grid ( $`G_{\text{in}}`$ )**: A grid where the top three rows ( $` y = 0, 1, 2 `$ ) are solid yellow ($` C_{\text{yellow}} `$), and the remaining rows are unpatterned.
- **Desired Output Grid ($` G_{\text{out}} `$)**: The top three rows remain yellow, and the remaining rows form a checkerboard pattern with colors $` C_{\text{blue}} `$ and $` C_{\text{red}} `$.

### Step-by-Step Solution

#### Step 1: Define Predicates

- **Frame Predicate**:

$$\text{Frame}(X) \iff y \in \{0, 1, 2\}$$

- **Pattern Predicate**:
$$\text{Pattern}(X) \iff (x + y) \bmod 2 = 0$$

#### Step 2: Infer Specification (`Spec`)

From the examples (excluding the top three rows):


$`
\text{Spec}(X) \iff \neg \text{Frame}(X) \implies \left\{
  \begin{array}{ll}
    \text{Pattern}(X) & \implies \text{Color}_{G_{\text{out}}}(X) = C_{\text{blue}} \\
    \neg \text{Pattern}(X) & \implies \text{Color}_{G_{\text{out}}}(X) = C_{\text{red}}
  \end{array}
\right.
`$


#### Step 3: Identify Frame (`R`)

Frame ensuring top rows remain unchanged:

$`
R(X) \iff \text{Frame}(X) \implies \text{Color}_{G_{\text{out}}}(X) = C_{\text{yellow}}
`$

#### Step 4: Apply the Frame Rule

Using the frame rule:

$`
\frac{[\text{Spec}] \quad \text{Examples} \quad \vdash \quad G_{\text{in}} \quad \rightarrow \quad [G_{\text{out}}]}{[\text{Spec} * R] \quad \text{Examples} * R \quad \vdash \quad G_{\text{in}} * R \quad \rightarrow \quad [G_{\text{out}} * R]}
`$

#### Step 5: Formulate Combined Specification

$`
[\text{Spec} * R] \iff \forall X, \quad \text{Spec}(X) \land R(X)
`$

Expanding:

$`
\forall X, \quad \left\{
  \begin{array}{ll}
    y \in \{0, 1, 2\} & \implies \text{Color}_{G_{\text{out}}}(X) = C_{\text{yellow}} \\
    y \geq 3 & \implies \left\{
      \begin{array}{ll}
        (x + y) \bmod 2 = 0 & \implies \text{Color}_{G_{\text{out}}}(X) = C_{\text{blue}} \\
        (x + y) \bmod 2 = 1 & \implies \text{Color}_{G_{\text{out}}}(X) = C_{\text{red}}
      \end{array}
    \right.
  \end{array}
\right.
`$

#### Step 6: Apply to Input Grid

For all cells `X`:

- **If** $` y \in \{0, 1, 2\} `$:
  - $` \text{Color}_{G_{\text{out}}}(X) = C_{\text{yellow}} `$ (unchanged).
- **Else**:
  - **If** $` (x + y) \bmod 2 = 0 `$:
    - $` \text{Color}_{G_{\text{out}}}(X) = C_{\text{blue}} `$
  - **Else**:
    - $` \text{Color}_{G_{\text{out}}}(X) = C_{\text{red}} `$

---

## Conclusion

By utilizing **bi-abductive inference** and the **frame rule**, we derive precise and modular specifications for grid transformations in InPaintingARC puzzles. The process involves:

- **Defining formal predicates** to capture properties of cells.
- **Inferring the transformation specification (`Spec`)** from examples.
- **Identifying the frame (`R`)** that remains unchanged.
- **Applying the frame rule** to combine `Spec` and `R` into a combined specification `[Spec * R]`.
- **Applying the combined specification** to the input grid to obtain the desired output grid.

This methodology ensures that transformations are correctly applied to the intended parts of the grid while preserving the invariant regions, enabling effective and accurate solutions to complex grid-based puzzles.
