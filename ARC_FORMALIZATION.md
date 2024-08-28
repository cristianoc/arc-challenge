# A Theory of ARC Problems

## Definitions

### Spec
A **Spec** is a subset of GridxGrid.
- Example: "The output width and height are both equal to the input width."

### Example
An **Example** is a pair (I, O) of input and output grid.

### Task
A **Task** is a set of examples and one test.
- Note: There can be more than one test, but for now, we'll focus on a single test.

### Spec as a Partial Function

A **Spec** determines a **partial function** from input to output:
- Given an input \( I \), the partial function defined by the Spec returns an output \( O \) only if there exists a unique output \( O \) related to \( I \). 
- If there is no such unique output (i.e., if multiple outputs are possible or no output exists), the function returns nothing.

### Spec as a Solution to a Task

A **Spec** defines a solution to a task if:
- The partial function it determines correctly maps the inputs to outputs in all provided examples. 
  - For each input \( I \) in the examples, the spec's function should return the correct output \( O \) as specified in the example.
- The partial function also correctly maps the input to the output in the test case(s) associated with the task.

### Expressible Specifications

**Expressible Specifications** are a subset of all possible specs, defined within a certain framework or system, that can be explicitly stated, described, or generated. 

#### Partial Order: "Simpler"

- There exists a **partial order** on expressible specs, denoted as "simpler."
- This partial order must satisfy the following property:
  - If one spec \( S_1 \) is a superset of another spec \( S_2 \) (i.e., \( S_1 \) is more general or inclusive), then \( S_1 \) is considered "simpler" than \( S_2 \).

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


## Example Instantiation of the Theory

#### Grid Representation

- **Grid:** A 3x3 matrix where each cell contains a value from {0, 1}, representing two colors (e.g., 0 = black, 1 = white).

#### Expressible Specifications

- **Set of Expressible Specs:**
  1. **Identity Spec:** The output grid is identical to the input grid.
  2. **Invert Colors Spec:** Each color in the grid is inverted (0 becomes 1, 1 becomes 0).
  3. **Vertical Flip Spec:** The grid is flipped vertically (top row becomes bottom row).
  4. **Horizontal Flip Spec:** The grid is flipped horizontally (left column becomes right column).

- **Simpler Relation:**
  - **Identity** is the simplest.
  - **Vertical Flip** and **Horizontal Flip** are equally simple and simpler than **Invert Colors**.
  - **Invert Colors** is the most complex.

#### Task Definition

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

#### Solution

- The correct expressible spec for this task is the **Invert Colors Spec**:
  - **Example 1** is symmetrical under color inversion.
  - **Example 2** clearly involves color inversion.
  
- **Test Case:** Applying the **Invert Colors Spec** to the test input yields the correct output, validating it as the simplest and correct solution.
