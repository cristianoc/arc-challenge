# ARC Challenge Framework: SizeARC, ColorsARC, and Grid Transformation Instantiations

This repository explores various simplified versions of the [Abstraction and Reasoning Corpus (ARC) challenge](https://www.kaggle.com/c/abstraction-and-reasoning-challenge), along with the theoretical framework and practical instantiations developed to solve them.

## Overview

The repository covers:
- **SizeARC**: A challenge focused on predicting the dimensions of output grids based on input grids.
- **ColorsARC**: A related challenge where the task is to predict the color patterns in the output grid.
- **Theory**: The underlying theoretical framework that guides the development of these challenges and solutions.
- **MicroARC**: A foundational instantiation of the theory focused on the simplest grid transformations.
- **MiniARC**: An extension of MicroARC that deals with more complex sequences of transformations.

## 1. SizeARC: A Simplified ARC Challenge

**SizeARC** is a focused task derived from the ARC challenge, where the objective is to predict the dimensions of the output grid based on a given input grid, isolating this specific facet for exploration.

### Solution Approach: Symbolic, Specification-Driven Method

1. **Specification-Driven Selection**: Utilizes predefined, high-level transformation rules to identify applicable transformations based on grid properties.
2. **Domain-Specific Priors**: Incorporates domain-specific knowledge to guide transformation selection.
3. **Semantic Reasoning**: Focuses on understanding the required transformations conceptually rather than enumerating every possible step.
4. **Layered Complexity**: Applies progressively sophisticated techniques, starting with simpler transformations.

### Results Accuracy

The final configuration achieved 94-95% accuracy on the SizeARC training and evaluation datasets.

### Implementation

The method applies a series of predefined transformations to the input grids, designed to relate input properties to the output grid size. These transformations range from basic operations, like directly using the input size, to more complex analyses involving object properties and grid patterns.

1. **Predefined Transformations**: The transformations are applied sequentially, starting with simpler ones and progressing to more complex methods as needed, based on the characteristics of the grid.

2. **Object Matching and Feature Detection**: After applying transformations, the method attempts to match objects between the input and output grids. When objects can be matched, the method detects common features to identify which object in the input corresponds to the output.

3. **Regularized Regression**: In cases where object matching and feature detection are insufficient to fully determine the output size, regularized regression is applied. This involves solving a regression problem to find weights and biases that best fit the observed data, while also incorporating regularization constraints to ensure model simplicity and prevent overfitting.

### Example

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

An ablation study was conducted to evaluate how different combinations of synthetic techniques and feature sets impact the performance of the SizeARC solution. The study examined the model's accuracy across various levels of complexity, ranging from basic transformations to advanced feature combinations.

### Methodology

The study incrementally activated synthetic techniques and feature sets across difficulty levels from 1 to 12. Each level represents a progressively more complex configuration, allowing for an analysis of how each feature or transformation influences the model’s accuracy.

### Results

The graph below summarizes the performance of the model on both the training and evaluation datasets across the different difficulty levels.

<img width="866" alt="Screenshot 2024-08-26 at 06 38 49" src="https://github.com/user-attachments/assets/0a0b6e7c-6c79-433f-a8f0-193afafb592c">

### Key Insights

- **Consistent Generalization**: The mechanism used in the SizeARC solution consistently generalizes well across both training and evaluation datasets, with similar accuracy scores observed at each difficulty level.

- **Significant Early Gains**: A notable improvement in accuracy is observed early on, particularly between levels 1 and 2, showing how the problem is very easy initially but gets harder later. This insight led to the decision to prioritize simpler transformations in the initial stages of the model.

- **Robust Performance**: As more advanced techniques are introduced, the model maintains robust performance, with accuracy steadily improving or stabilizing as complexity increases. These results guided the integration of regularized regression at higher complexity levels to ensure stability and accuracy in difficult cases.

## 2. ColorsARC: Predicting Color Patterns

**ColorsARC** is a variant of the ARC challenge where the focus is on predicting the color patterns in the output grid, rather than its size.

### Solution Approach

The approach to ColorsARC is similar to SizeARC but focuses on color transformations rather than size transformations. Predefined transformation rules and domain-specific knowledge are used to predict the color configurations in the output grid.

## 3. Theory: The Foundation of the Approach

The [ARC_Problem_Theory.md](./ARC_Problem_Theory.md) document provides an in-depth explanation of the theoretical framework guiding this project. The theory defines a general approach to solving ARC problems through minimal and well-defined transformation specifications.

### MicroARC: An Instantiation of the Theory

**MicroARC** is a foundational instantiation of the theory that focuses on the simplest grid transformations. It serves as the base level of complexity, where the goal is to identify minimal transformations that produce well-defined outputs.

### MiniARC: Extending MicroARC

**MiniARC** builds on the principles of MicroARC by handling more complex sequences of transformations. It explores how combinations of simple transformations can be used to solve more sophisticated grid manipulation tasks.
