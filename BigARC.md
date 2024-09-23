
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

### Three Approaches for Pattern Management

The introduction of Symmetry masks has expanded the range of expressible patterns, necessitating a more nuanced approach to pattern recognition and application. We can distinguish three main strategies:

1. **Shared Pattern**: 
   - Concept: The correct pattern emerges from analyzing all examples collectively.
   - Process: Combine insights from each example to form a unified solution.
   - Application: Apply this shared pattern to the test input.
   - Example: In puzzles like Sudoku, examining all examples together is crucial to identify the correct pattern.

2. **Pattern Relearn**: 
   - Concept: Individual patterns are correct for their respective examples, but no shared pattern exists.
   - Implication: Patterns vary across examples.
   - Approach: For each new input, including the test input, infer the pattern from scratch.
   - Strategy: Emphasizes adaptability and context-specific pattern recognition.

3. **Rule Abandonment**: 
   - Concept: Neither individual patterns nor their combination yields a correct solution.
   - Occurrence: Considered rare.
   - Action: Discard the current rule entirely and seek alternative approaches.

While these strategies are distinct, they can be viewed as part of a continuous spectrum rather than mutually exclusive categories. In a more nuanced approach, we can assign likelihood scores to each strategy:

- Rule Abandonment: Low likelihood of rule applicability.
- Shared Pattern: High likelihood of rule applicability, influenced by the likelihood of individual patterns.
- Pattern Relearn: Each pattern has a certain likelihood of applying, with the final solution being a combination of these likelihoods.

In this continuous model, multiple strategies can coexist with varying degrees of relevance.

To transition from the continuous model back to the discrete approach:

1. Consider only a limited set of predicates, each with a binary likelihood (0 or 1).
2. Prioritize strategies in this order:
   a. Shared Pattern (if predicates are found)
   b. Pattern Relearn (if Shared Pattern fails)
   c. Rule Abandonment (as a last resort when suitable predicates are not identified)
