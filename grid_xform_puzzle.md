### Report: Grid Transformation Puzzle

#### Problem Overview

You are working with a grid of size $` n \times n `$, where each cell can be either black (represented by 0) or white (represented by 1). The task involves applying a sequence of transformations to this grid, simplifying these sequences, and eventually solving a puzzle based on given examples.

#### Transformations

The grid can undergo the following transformations:

1. **Rotations**:
   - R1: 90 degrees
   - R2: 180 degrees
   - R3: 270 degrees

2. **Symmetric Transformations**:
   - X: Horizontal flip (x-swap)
   - Y: Vertical flip (y-swap)

3. **Inversion**:
   - I: Swapping 1 with 0 (black with white and vice versa)

These transformations can be combined into sequences, and certain commutation laws allow the rearrangement of these sequences without changing the final outcome. Specifically:

- **Inversion** can be moved before or after any other transformation.
- **Symmetric transformation followed by a Rotation** can be reordered, possibly switching the type of symmetric transformation.

#### Normal Form

By systematically applying the commutation laws, any sequence of transformations can be reduced to a **normal form**, which is the simplest possible sequence that cannot be further simplified. The normal form follows this order:

1. **Rotations**
2. **x-swap (Horizontal flip)**
3. **y-swap (Vertical flip)**
4. **Inversion**

Each of these steps is optional, meaning some may be omitted depending on the specific transformation sequence.

The **length** of the normal form is defined by the number of transformations involved, ranging from zero (no transformation, the identity) to four.

#### Counting Normal Forms

There are 32 possible normal forms, categorized by length:

- **Length 0**: 1 form (the identity transformation).
- **Length 1**: 6 forms.
- **Length 2**: 12 forms.
- **Length 3**: 10 forms.
- **Length 4**: 3 forms.

#### Visualization

To better understand these transformations, a visualization can be created by applying all 32 normal forms to a black-and-white, non-symmetric image. The results will be displayed in rows, organized by the length of the transformation sequence. This is generated with `grid_xform_puzzle.py`:

<img width="1718" alt="Screenshot 2024-08-29 at 14 26 01" src="https://github.com/user-attachments/assets/25c94be0-d4e7-4d01-9d97-5bef8980122e">


#### The Puzzle Game

The game is structured around solving puzzles using transformation sequences:

1. **Examples**: Each example consists of an input-output grid pair that excludes certain transformations.
2. **Tests**: The challenge is to identify the correct transformation sequence based on the given examples.

##### Game Objective

The goal is to find the **simplest transformation** that explains all the examples provided. Here’s how the process works:

1. Begin with the grid image showing all 32 transformations.
2. For each example, cross out the transformations that the example excludes.
3. After processing all examples, examine the remaining transformations.
4. Identify the **lowest level** (i.e., shortest sequence length) that still has uncrossed transformations.
5. If this level has only one transformation left, that is the simplest and correct solution.
6. If more than one transformation remains, the problem statement is ambiguous, suggesting multiple possible solutions.

This puzzle-solving approach systematically narrows down the possible transformations until the simplest valid one is identified.
