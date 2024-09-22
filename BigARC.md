# BigARC: The Largest and Most Colorful Puzzles

**BigARC** is a new variant of the ARC challenge that focuses on the largest puzzles with the greatest number of colors. This challenge aims to push the boundaries of current ARC-solving techniques by introducing more complex and colorful grids.

## BigARC Tasks

BigARC tasks consist of grids with a large number of cells and a wide variety of colors. The goal is to identify and apply transformations that handle the increased complexity and color diversity.

## Solution Approach

The solution approach for BigARC is still under development. It will likely involve advanced techniques for managing large grids and multiple colors, building on methods used in other ARC challenges.

### Example 1

Here’s an example of a BigARC task with a 23x23 grid and 10 colors:
![bigarc-example1](images/bigarc-example1.png)

The solution is to identify the subgrid with only one color. We will now break this down into composable reasoning steps.

### Subgrid Extraction

Each input consists of NxN subgrids identified by a regular lattice structure. This process is handled by the visual cortex component and represented abstractly by a primitive `split: Grid -> List[Grid]`. Analogous primitives exist for other puzzle types, such as extracting all objects or all colored objects.

### List Transformations

Subgrid extraction changes the problem type: the original operates on grids, but now we must work with lists of grids. Transformations then apply from `List[Grid]` to `Grid`. Other transformation types also exist, like mapping lists to lists, or painting one grid onto another.

### Split List Transformation

In the example, we use a specific transformation called a "split transformation," parameterized by `split: Grid -> List[Grid]`, applied to the lattice split.

Here is the rule:

![split-rule](images/split-rule.png)

This rule packages the list of subgrids and delegates the task to rules that operate on lists.

In the specific example, each list contains NxN subgrids, though in general, the important point is that each subgrid has a unique index.

### Select Transformation

Given a list of grids, the **Select transformation** chooses one. In this example, it selects the grid with only one color.

Conceptually, grids are embedded into a 1-dimensional space with a single boolean, indicating whether a single color is present. Generally, we consider a space `Sp` and an embedding `emb: Grid -> Sp`. The role of the embedding is to identify the intended subgrid. Thus, the problem is one of classification, with the classes being elements of the space.

Several features, such as the number of colors, can be used to classify the correct subgrid. Here’s the Select rule:

![select-rule](images/select-rule.png)

This rule operates on lists of grids, denoted `L` and `Li`, and produces a single grid. It compares embeddings with a specific state `s`, used to select the correct subgrid. In the example, `s` determines which subgrid to project to get the answer. The inferred specification describes how `s` classifies the outputs in the example. The final output is computed by using `s` to classify the grids in the test input `L`.

### Solving the Example

To solve the example:

- **Level 1**: Apply the Split rule and move to the lists-to-grid space.
- **Level 2**: Apply the Select rule using the `single color` predicate to generate the embedding.


### Example 2

Here’s an example of a BigARC task with a 23x23 grid and 10 colors:
![bigarc-example2](images/bigarc-example2.png)

The example takes aspects from 2 subproblems:
1. The pattern from [InPaintingARC](InPaintingARC.md)
2. The size of the output from [SizeARC](SizeARC.md)

The pattern relies on Symmetry Masks, which means symmetries are satisfied only by a subset of the input (the mask).
The solution from InPaintingARC is not sufficient as it relies on at least one symmetry to hold without mask (to e.g. find the center of symmetry). While here some extension is required.
