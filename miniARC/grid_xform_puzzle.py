# type: ignore
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List

from PIL import Image


# Define transformations
def rotate_90(grid):
    return np.rot90(grid, -1)


def rotate_180(grid):
    return np.rot90(grid, -2)


def rotate_270(grid):
    return np.rot90(grid, -3)


def flip_horizontal(grid):
    return np.fliplr(grid)


def flip_vertical(grid):
    return np.flipud(grid)


def invert(grid):
    return 1 - grid


# Apply a sequence of transformations
def apply_transformations(grid, transformations):
    for transform in transformations:
        grid = transform(grid)
    return grid


# Visualization of transformations with labels
def get_transformation_label(transformations):
    labels = []
    for transform in transformations:
        if transform == rotate_90:
            labels.append("R1")
        elif transform == rotate_180:
            labels.append("R2")
        elif transform == rotate_270:
            labels.append("R3")
        elif transform == flip_horizontal:
            labels.append("X")
        elif transform == flip_vertical:
            labels.append("Y")
        elif transform == invert:
            labels.append("I")
    return "".join(labels) if labels else "Identity"


def visualize_transformations_with_labels(grid, transformations_by_level):
    max_cols = max(len(level) for level in transformations_by_level)
    rows = len(transformations_by_level)

    fig, axes = plt.subplots(rows, max_cols, figsize=(max_cols * 1.5, rows * 1.5))

    if rows == 1:
        axes = np.array([axes])

    for i, level in enumerate(transformations_by_level):
        for j, (length, transformation_sequence) in enumerate(level):
            label = get_transformation_label(transformation_sequence)
            transformed_grid = apply_transformations(
                grid.copy(), transformation_sequence
            )
            ax = axes[i, j] if rows > 1 else axes[j]
            ax.imshow(
                transformed_grid, cmap="gray", vmin=0, vmax=1, interpolation="nearest"
            )
            ax.set_title(f"{label}")
            ax.axis("off")

        for k in range(len(level), max_cols):
            ax = axes[i, k] if rows > 1 else axes[k]
            ax.axis("off")

    plt.tight_layout()
    plt.show()


cat_image = Image.open("cat_image.png").convert("L")
cat_image = cat_image.resize((128, 128))
cat_array = np.array(cat_image)
binary_cat_array = np.where(cat_array > 128, 1, 0)

Rotations = [None, rotate_90, rotate_180, rotate_270]
X_Flip = [None, flip_horizontal]
Y_Flip = [None, flip_vertical]
Inversions = [None, invert]

Form = Tuple[
    Optional[callable], Optional[callable], Optional[callable], Optional[callable]
]


# Generate all possible normal forms
def generate_all_normal_forms() -> List[Form]:
    forms = []
    for r in Rotations:
        for x in X_Flip:
            for y in Y_Flip:
                for i in Inversions:
                    forms.append((r, x, y, i))
    return forms


# Define sorting order
def primitive_sort_key(primitive):
    if primitive == rotate_90:
        return 1
    elif primitive == rotate_180:
        return 2
    elif primitive == rotate_270:
        return 3
    elif primitive == flip_horizontal:
        return 4
    elif primitive == flip_vertical:
        return 5
    elif primitive == invert:
        return 6
    return 0  # Identity (None)


# Simplify the normal form and sort based on normal form order
def simplify_normal_form(form: Form) -> Tuple[int, List[callable]]:
    simplified_form = [t for t in form if t is not None]
    simplified_form.sort(key=primitive_sort_key)
    return len(simplified_form), simplified_form


# Generate all simplified normal forms grouped by their lengths
def generate_transformations_by_level():
    all_forms = generate_all_normal_forms()
    simplified_forms = [simplify_normal_form(form) for form in all_forms]

    transformations_by_level = [[] for _ in range(5)]  # 5 levels: 0 to 4
    for length, form in simplified_forms:
        transformations_by_level[length].append((length, form))

    for level in transformations_by_level:
        level.sort(key=lambda x: [primitive_sort_key(p) for p in x[1]])

    return transformations_by_level


transformations_by_level = generate_transformations_by_level()

visualize_transformations_with_labels(binary_cat_array, transformations_by_level)
