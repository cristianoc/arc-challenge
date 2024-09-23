
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

Here’s another example of a BigARC task with a 23x23 grid and 10 colors:
![bigarc-example2](images/bigarc-example2.png)

This example combines aspects from two subproblems:
1. The pattern from [InPaintingARC](InPaintingARC.md)
2. The size of the output from [SizeARC](SizeARC.md)

The pattern relies on Symmetry Masks, which means symmetries are satisfied only by a subset of the input (the mask). The solution from InPaintingARC is not sufficient as it relies on at least one symmetry to hold without a mask (e.g., to find the center of symmetry). Here, some extension is required to simultaneously determine both the symmetry masks and the center of symmetry.

---

### Three Distinctions for Managing Patterns

After introducing Symmetry masks, the number of expressible patterns has increased. As a consequence, some of the previously found solutions were not being found anymore. This required a finer distinction between three specific cases that help address these new challenges.

1. **Shared Pattern**: In this case, the correct pattern is learned only after reviewing all the examples. The shared pattern is formed by combining insights from each example, and this unified solution is applied to the test input. This is crucial for puzzles like Sudoku, where looking at all examples together is necessary to identify the correct pattern.

2. **Pattern Relearn**: Here, each individual pattern is correct for its example, but the shared pattern is not. This implies that patterns differ across examples. When faced with a test input, the pattern must be inferred again from scratch based solely on that input, reinforcing the strategy of relearning the pattern each time.

3. **Rule Abandonment**: In this scenario, neither the individual patterns nor their intersection is correct. This is considered an unlikely case, and when it occurs, the rule is abandoned entirely.

This is a discrete description of the three cases, but one can also consider a more continuous description where a likelyhood score is assigned to each case.
Rule abandonment would translated to: there is low likelyhood that this rule applies.
Shared pattern would translated to: there is a high likelyhood that this rule applies, and depends on the likelihood of each individual pattern.
Pattern relearn would translated to: each pattern has some likelyhood of applying, and the final solution is some combination of the likelyhood of all patterns.
In this continuous description, the three cases are not mutually exclusive, and a likelyhood score can be assigned to each case.

The discrete case can be recovered from the continuout case in the following way:
- There are only a few predicates, and not likelyhood is associated to them. In other words, considering the set of all predicates, their likelyhood is either 0 or 1.
- Shared pattern always wins, as long as one can find predicates.
- Pattern relearn is the second choice when the first one fails.
- Rule abandonment is the last choice when suitable predicates cannot be found.
