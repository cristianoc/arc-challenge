# A Theory of Well-Defined ARC Problems

## Definitions

### Spec
A **Spec** is a subset of GridxGrid.
- Example: "The output width and height are both equal to the input width."

### Example
An **Example** is a pair $`(I, O)`$ of input and output grid.

### Task
A **Task** is a set of examples and one test.
- Note: There can be more than one test, but for now, we'll focus on a single test.

### Spec as a Partial Function

A **Spec** determines a **partial function** from input to output:
- Given an input $`I`$, the partial function defined by the Spec returns an output $`O`$ only if there exists a unique output $`O`$ related to $`I`$. 
- If there is no such unique output (i.e., if multiple outputs are possible or no output exists), the function returns nothing.

### Spec as a Solution to a Task

A **Spec** defines a solution to a task if:
- The partial function it determines correctly maps the inputs to outputs in all provided examples. 
  - For each input $`I`$ in the examples, the spec's function should return the correct output $`O`$ as specified in the example.
- The partial function also correctly maps the input to the output in the test case(s) associated with the task.

### Expressible Specifications

**Expressible Specifications** are a subset of all possible specs, defined within a certain framework or system, that can be explicitly stated, described, or generated.

#### Partial Order: "Simpler"

- There exists a **partial order** on expressible specs, denoted as "simpler."
- This partial order must satisfy the following property:
  - If one spec $`S_1`$ is a superset of another spec $`S_2`$ (i.e., $`S_1`$ is more general or inclusive), then $`S_1`$ is considered "simpler" than $`S_2`$.

#### Mechanisms for Defining Expressible Specs

- **Mechanism for Creation:** The expressible specs may be defined using various mechanisms, such as a Domain-Specific Language (DSL) or other formal systems.
  - **Note:** The specific mechanism used to create the set of expressible specs is a detail and not a part of the problem definition itself.

### Defining an ARC Problem

To precisely define an ARC problem, the following parameters must be provided:
1. **Set of Expressible Specifications:** The subset of specs that can be explicitly stated or generated.
2. **Partial Order on Specs:** The "simpler" relation that orders specs by their complexity or generality.

A task in an ARC problem involves finding the simplest spec, according to this partial order, that correctly maps the inputs to outputs in the examples and test.

### Correct Solution

An **Expressible Specification** (or **Spec**) is considered the **correct solution** to a task if it satisfies the following condition:

- The spec is the **simplest solution** within the set of expressible specs that correctly maps the inputs to outputs in all provided examples and the test case(s).

#### Simplest Solution
- A spec is the **simplest** if there is no other spec within the set of expressible specs that:
  1. Correctly maps all the inputs to outputs in the examples and test.
  2. Is simpler according to the partial order on specs.

In other words, among all the expressible specs that solve the task, the correct solution is the one that is considered simplest by the partial order.

### Well-Defined Task

A **Well-Defined Task** is a task that is guaranteed to have a unique minimal solution within the set of expressible specs.

More precisely, given input $`I`$ for the test, we can consider all outputs $`O`$ and all correct solutions for the resulting task. The requirement is that such solution is unique. (Therefore, $`O`$ is also unique).

## Example Instantiation: MicroARC

### Grid Representation

- **Grid:** A 3x3 matrix where each cell contains a value from $`\{0, 1\}`$, representing two colors (e.g., 0 = black, 1 = white).

### Expressible Specifications

- **Set of Expressible Specs:**
  1. **Identity Spec:** The output grid is identical to the input grid.
  2. **Invert Colors Spec:** Each color in the grid is inverted (0 becomes 1, 1 becomes 0).
  3. **Vertical Flip Spec:** The grid is flipped vertically (top row becomes bottom row).
  4. **Horizontal Flip Spec:** The grid is flipped horizontally (left column becomes right column).

- **Simpler Relation:**
  - **Identity** is the simplest.
  - **Vertical Flip** and **Horizontal Flip** are equally simple and simpler than **Invert Colors**.
  - **Invert Colors** is the most complex.

### Example Task Definition

**Task:** Determine the correct transformation based on the following examples and apply it to a test case.

- **Examples:**
  1. **Input:** 
     ```
     1 0 1
     0 1 0
     1 0 1
     ```
     **Output:** 
     ```
     1 0 1
     0 1 0
     1 0 1
     ```
  
  2. **Input:** 
     ```
     1 1 0
     0 0 1
     1 1 0
     ```
     **Output:** 
     ```
     0 0 1
     1 1 0
     0 0 1
     ```

- **Test:**
  - **Input:** 
    ```
    1 0 1
    0 0 0
    1 0 1
    ```
  - **Expected Output:** 
    ```
    0 1 0
    1 1 1
    0 1 0
    ```

### Solution

- The correct expressible spec for this task is the **Invert Colors Spec**:
  - **Example 1** is symmetrical under color inversion.
  - **Example 2** clearly involves color inversion.
  
- **Test Case:** Applying the **Invert Colors Spec** to the test input yields the correct output, validating it as the simplest and correct solution.

Since the **Invert Colors Spec** is the only spec that can correctly solve both examples, the task is well-defined.

### Example of a Task That is Not Well-Defined

**Task Definition:**

- **Example:**
  - **Input:**
    ```
    0 1
    1 0
    ```
  - **Output:**
    ```
    1 0
    0 1
    ```

- **Test:**
  - **Input:**
    ```
    1 0
    0 0
    ```

  In this task, different specifications will yield different outputs for the test input:
    - **Vertical Flip Spec:** 
      ```
      0 0
      1 0
      ```
    - **Horizontal Flip Spec:** 
      ```
      0 1
      0 0
      ```
    - **Invert Colors Spec:** 
      ```
      0 1
      1 1
      ```

**Analysis:**

- **Vertical Flip Spec:** This spec flips the grid along its vertical axis and is consistent with the example.
- **Horizontal Flip Spec:** This spec flips the grid along its horizontal axis and is also consistent with the example.
- **Invert Colors Spec:** This spec inverts the colors of the grid, which is another consistent transformation based on the example.
- **Identity Spec:** The Identity Spec leaves the grid unchanged and does not match the example output.

**Why This Task is Not Well-Defined:**

This task is not well-defined because the provided example can be solved by multiple minimal specifications, each of which produces different outputs for the test input. Specifically, both the **Vertical Flip Spec** and **Horizontal Flip Spec** are minimal, but they produce different outputs when applied to the test input. Additionally, the **Invert Colors Spec** offers another solution, though it is more complex and not minimal compared to the flip specs. The existence of these multiple minimal solutions, combined with their differing outputs for the test case, introduces ambiguity and prevents the determination of a unique minimal solution. Therefore, the task is not well-defined.


## Additional Example: MiniARC

### Description of MiniARC

[MiniARC](miniARC) is a variant of the ARC problem where the task involves identifying a sequence of transformations from a set of operations, including flips, rotations, and inversions, to correctly map input grids to output grids. 

### Expressible Specifications in MiniARC

- **Core Specifications**:
  1. **Invert Colors:** Inverts all cell colors.
  2. **Vertical Flip:** Flips the grid vertically.
  3. **Horizontal Flip:** Flips the grid horizontally.
  4. **Rotate 90:** Rotates the grid 90 degrees clockwise.
  5. **Rotate 180:** Rotates the grid 180 degrees.
  6. **Rotate 270:** Rotates the grid 270 degrees clockwise.
- **Sequences**: Any valid sequence, including possibly empty sequences, composed of the Core Specifications.

### Why MiniARC is an Instance of the Theory

- **Expressible Specifications**: The MiniARC variant fits within the theory as it utilizes a defined set of expressible specifications (transformations like flips, rotations, and color inversions). Each transformation functions as a specific case of a relation between input and output grids, making it a concrete example of the relational concept used in the theory.

- **Partial Order**: MiniARC also incorporates a notion of simpler specifications. Since transformations can be reduced to normal forms with a defined size, the partial order is determined by the size of these forms: a specification is simpler than another if it has a strictly smaller size. Thus, the identity transformation is the simplest, and specifications of the same size are considered equivalent in complexity but not directly comparable to each other.
