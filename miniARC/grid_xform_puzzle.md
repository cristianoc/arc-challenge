### MiniARC: Grid Transformation Puzzle Report

#### Problem Overview

In **MiniARC**, you are tasked with working on an $n \times n$ grid, where each cell is either black (represented by 0) or white (represented by 1). The goal is to apply a sequence of **primitives** to this grid, simplify these sequences into **transformations**, and solve puzzles based on provided examples.

#### Primitives

The grid in **MiniARC** can undergo the following **primitives**, which are the basic operations that can be combined to form transformations:

1. **Rotations**:
   - **R1**: 90 degrees
   - **R2**: 180 degrees
   - **R3**: 270 degrees

2. **Symmetric Primitives**:
   - **X**: Horizontal flip
   - **Y**: Vertical flip

3. **Inversion**:
   - **I**: Swapping 1 with 0 (black with white and vice versa)

#### Transformations

A **transformation** is a sequence of **primitives** applied in a specific order to modify the grid. These transformations can be simplified by applying certain commutation laws, which allow you to reorder the primitives without changing the final outcome:

- **Inversion (I)** can be moved before or after any other primitive.
- A **Symmetric primitive followed by a Rotation** can be reordered, possibly changing the type of symmetric primitive.

#### Normal Form

By applying these commutation laws in **MiniARC**, any sequence of primitives (i.e., any transformation) can be reduced to a **normal form**—the simplest possible transformation that cannot be further simplified. The normal form follows this order:

1. **Rotations**
2. **Horizontal flip (X)**
3. **Vertical flip (Y)**
4. **Inversion (I)**

Each step is optional, depending on the specific sequence of primitives. The **length** of the normal form is defined by the number of primitives involved, ranging from zero (no transformation, the identity) to four.

#### Counting Normal Forms

In **MiniARC**, there are 32 possible normal forms, categorized by their length:

- **Length 0**: 1 form (the identity transformation)
- **Length 1**: 6 forms
- **Length 2**: 12 forms
- **Length 3**: 10 forms
- **Length 4**: 3 forms

#### Visualization

To better understand these transformations, a visualization can be created by applying all 32 normal forms to a non-symmetric, black-and-white image in **MiniARC**. The results are displayed in rows, organized by the length of the transformation sequence. This visualization is generated using the `grid_xform_puzzle.py` script.

![transforms](https://github.com/user-attachments/assets/7e9ed873-1dda-4407-be68-5ebedfe70f19)

#### The Puzzle Game

The **MiniARC** game revolves around solving puzzles using transformations:

1. **Examples**: Each example consists of an input-output grid pair that excludes certain transformations.
2. **Tests**: The challenge is to identify the correct transformation based on the given examples.

##### Game Objective

The objective in **MiniARC** is to find the **simplest transformation** that explains all the provided examples. The process is as follows:

1. Begin with the grid image showing all 32 transformations.
2. For each example, eliminate the transformations that the example excludes.
3. After processing all examples, review the remaining transformations.
4. Identify the **lowest level** (i.e., shortest transformation) that still has uncrossed transformations.
5. If only one transformation remains at this level, it is the simplest and correct solution.
6. If more than one transformation remains, the problem is ambiguous, indicating multiple possible solutions.

This approach systematically narrows down the possible transformations to identify the simplest valid one.

### MiniARC Puzzles

#### Puzzle 1

![puzzle1](https://github.com/user-attachments/assets/970d2e43-b91e-45cb-b0ea-82f21ceb1bfb)

**Simplest solution** in **MiniARC**: Inversion $`I`$.

#### Puzzle 2

![puzzle2](https://github.com/user-attachments/assets/21ebe3ef-0144-42a7-a7ad-77ef1fd839ba)

**Simplest solution** in **MiniARC**: Inversion $`I`$.

#### Puzzle 3

![puzzle3](https://github.com/user-attachments/assets/4e37cff0-c32d-4138-ac15-60eb63384d4c)

**Simplest solution** in **MiniARC**: Horizontal flip $`X`$ followed by Inversion $`I`$.

**Non-minimal solution** in **MiniARC**: 180 rotation $`R2`$, followed by vertical flip $`Y`$, followed by Inversion $`I`$.

Number of solutions without applying normalization in **MiniARC**:

<img width="300" alt="Screenshot 2024-08-29 at 16 47 25" src="https://github.com/user-attachments/assets/da18a2a0-f458-49d2-8b66-084ab8810b02">


#### Puzzle 4: Ambiguity Detection

![puzzle4](https://github.com/user-attachments/assets/7a1296b4-9fc3-4d8a-8a95-46f972ffd4c9)

**Ambiguity detected:** Multiple transformations remain at level 1.

**Ambiguous Transformations at Level 1:**
- **R1**: Rotate 90 degrees
- **R3**: Rotate 270 degrees
- **X**: Horizontal flip
- **Y**: Vertical flip

This puzzle illustrates a case where the problem statement is ambiguous, as multiple valid transformations could produce the same output. Additional constraints or examples are necessary to uniquely identify the simplest solution.
