# ARC Challenge Framework: InPaintingARC, BigARC, SizeARC, ColorsARC, and Foundations

This repository explores various simplified versions of the [Abstraction and Reasoning Corpus (ARC) challenge](https://www.kaggle.com/c/abstraction-and-reasoning-challenge), along with the theoretical framework and practical instantiations developed to solve them.

## Overview

The repository covers:

1. **InPaintingARC**: A vertical subset of ARC-AGI: the in-painting puzzles of any difficulty.
2. **BigARC**: A challenge focused on handling the largest grids with the largest number of colors.
3. **SizeARC**: A challenge focused on predicting the dimensions of output grids based on input grids.
4. **ColorsARC**: A related challenge where the task is to predict the color patterns in the output grid.
5. **Theory**: The underlying theoretical framework that guides the development of these challenges and solutions.
   - **MicroARC**: A foundational instantiation of the theory focused on the simplest grid transformations.
   - **MiniARC**: An extension of MicroARC that deals with more complex sequences of transformations.

## 1. InPaintingARC: A vertical subset of ARC-AGI

**InPaintingARC** is a variant of the ARC challenge intended to be smaller in variety, but not simpler on a per-puzzle basis, than full ARC-AGI.

### Solution Approach

The approach to InPaintingARC requires a re-thinking of the way solutions are represented and explored.
Solutions consists of spatial specifications that are composable and make use of predicates capturing certain aspects of the priors required (objects, color, symmetry, frames).
To drive the seach of solutions, a special form of bi-abducrive inference is developed, where given a set of examples and one input, one derives the spec for the transformation that the examples represent, and the corresponding final  output.

For more detailed information, please refer to the [InPaintingARC.md](./InPaintingARC.md) document.

## 2. BigARC: Handling the Largest Grids with the Largest Number of Colors

**BigARC** is a challenge focused on handling the largest grids with the largest number of colors, pushing the boundaries of grid-based reasoning and transformation.

For more detailed information, please refer to the [BigARC.md](./BigARC.md) document.

## 3. SizeARC: A Simplified ARC Challenge

**SizeARC** is a focused task derived from the ARC challenge, where the objective is to predict the dimensions of the output grid based on a given input grid, isolating this specific facet for exploration.

For more detailed information, please refer to the [SizeARC.md](./SizeARC.md) document.

## 4. ColorsARC: Predicting Color Patterns

**ColorsARC** is a variant of the ARC challenge where the focus is on predicting the color patterns in the output grid, rather than its size.

### Solution Approach

The approach to ColorsARC is similar to SizeARC but focuses on color transformations rather than size transformations. Predefined transformation rules and domain-specific knowledge are used to predict the color configurations in the output grid.

## 5. Theory: The Foundation of the Approach

The [ARC_Problem_Theory.md](./ARC_Problem_Theory.md) document provides an in-depth explanation of the theoretical framework guiding this project. The theory defines a general approach to solving ARC problems through minimal and well-defined transformation specifications.

### MicroARC: An Instantiation of the Theory

**MicroARC** is a foundational instantiation of the theory that focuses on the simplest grid transformations. It serves as the base level of complexity, where the goal is to identify minimal transformations that produce well-defined outputs.

### MiniARC: Extending MicroARC

**MiniARC** builds on the principles of MicroARC by handling more complex sequences of transformations. It explores how combinations of simple transformations can be used to solve more sophisticated grid manipulation tasks.
