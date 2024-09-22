# SizeARC: A Simplified ARC Challenge

**SizeARC** is a focused task derived from the ARC challenge, where the objective is to predict the dimensions of the output grid based on a given input grid, isolating this specific facet for exploration.

## Solution Approach: Symbolic, Specification-Driven Method

1. **Specification-Driven Selection**: Utilizes predefined, high-level transformation rules to identify applicable transformations based on grid properties.
2. **Domain-Specific Priors**: Incorporates domain-specific knowledge to guide transformation selection.
3. **Semantic Reasoning**: Focuses on understanding the required transformations conceptually rather than enumerating every possible step.
4. **Layered Complexity**: Applies progressively sophisticated techniques, starting with simpler transformations.

## Results Accuracy

The final configuration achieved 94-95% accuracy on the SizeARC training and evaluation datasets.

## Implementation

The method applies a series of predefined transformations to the input grids, designed to relate input properties to the output grid size. These transformations range from basic operations, like directly using the input size, to more complex analyses involving object properties and grid patterns.

1. **Predefined Transformations**: The transformations are applied sequentially, starting with simpler ones and progressing to more complex methods as needed, based on the characteristics of the grid.

2. **Object Matching and Feature Detection**: After applying transformations, the method attempts to match objects between the input and output grids. When objects can be matched, the method detects common features to identify which object in the input corresponds to the output.

3. **Regularized Regression**: In cases where object matching and feature detection are insufficient to fully determine the output size, regularized regression is applied. This involves solving a regression problem to find weights and biases that best fit the observed data, while also incorporating regularization constraints to ensure model simplicity and prevent overfitting.

## Example

![Example](https://github.com/user-attachments/assets/4112f991-a296-456b-838f-88574200a8d2)

An interesting category is where the output is a copy of an object in the input (or at least, its size is). These are handled in three phases:

- **Matching**: Objects in the input are found and matched against the one in the output.
- **Feature Detection**: If objects are successfully matched, the system detects common features that could determine which object to select.
- **Find a common decision rule**: A combination of features that uniquely identify which one is selected.

In this case, a single feature is detected as distinguishing the output:

```
Common decision rule (Symmetry): VERTICAL = False.
```

So the selected object is the only one that is not symmetric across the vertical axis.

## Ablation Study

### Overview

An ablation study was conducted to evaluate how different combinations of synthetic techniques
