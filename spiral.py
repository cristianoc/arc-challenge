from typing_extensions import dataclass_transform
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass


def generate_spiral(N, M):
    # Create an NxM grid filled with zeros
    grid = np.zeros((N, M))

    # Starting point
    x, y = N // 2, M // 2
    grid[x, y] = 1

    # Directions: right, down, left, up (clockwise rotation)
    dir4 = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    step = 1  # Initial step size
    direction_index = 0  # Start by going right
    count = 2  # Start filling from 2 onwards

    while True:
        dx, dy = dir4[direction_index]
        for _ in range(step):
            x += dx
            y += dy
            if 0 <= x < N and 0 <= y < M:
                grid[x, y] = 1
            else:
                return grid
        direction_index = (direction_index + 1) % 4
        step += 1


# Set the grid size
N, M = 13, 13

# Generate the spiral pattern
spiral_grid = generate_spiral(N, M)

# Plot the spiral grid
plt.figure(figsize=(6, 6))
plt.imshow(spiral_grid, cmap="viridis", origin="upper")
plt.colorbar(label="Spiral Steps")
plt.title(f"{N}x{M} Spiral Grid")
plt.show()

@dataclass
class SubgridPattern:
    index: int
    frequency: int
    # map direction to indices of subgrids adjacent in that direction
    adjecency: dict[tuple[int, int], set[int]]

    def __str__(self):
        return f"Sub{self.index}: frequency={self.frequency} adjecency={self.adjecency}"


# Extract all 3x3 subgrids
subgrids = {}


# k defines the size of the subgrids (k x k) to be extracted from the spiral grid
k = 2
rows = N - k + 1
cols = M - k + 1
new_grid = np.zeros((rows, cols), dtype=int)


def sub_in_bound(r, c):
    top = r
    bottom = r + k - 1
    left = c
    right = c + k - 1
    return 0 <= top < N and 0 <= bottom < N and 0 <= left < M and 0 <= right < M


def get_normalized_subgrid(r, c):
    return tuple(map(tuple, spiral_grid[r : r + k, c : c + k]))


for i in range(rows):
    for j in range(cols):
        subgrid = get_normalized_subgrid(i, j)

        # Store the position of each subgrid occurrence
        if subgrid not in subgrids:
            index = len(subgrids)
            pattern = SubgridPattern(index, 1, {})
            subgrids[subgrid] = pattern
        else:
            pattern = subgrids[subgrid]
            pattern.frequency += 1
        new_grid[i, j] = pattern.index

dir8 = [(0, k), (k, k), (k, 0), (k, -k), (0, -k), (-k, -k), (-k, 0), (-k, k)]


# populate adjecency
for i in range(rows):
    for j in range(cols):
        subgrid = get_normalized_subgrid(i, j)
        pattern = subgrids[subgrid]
        for dx, dy in dir8:
            if sub_in_bound(i + dx, j + dy):
                adj_subgrid = get_normalized_subgrid(i + dx, j + dy)
                adj_pattern = subgrids[adj_subgrid]
                if (dx, dy) not in pattern.adjecency:
                    pattern.adjecency[(dx, dy)] = set()
                adj_set = pattern.adjecency[(dx, dy)]
                adj_set.add(adj_pattern.index)


# Number of subgrids to plot
num_subgrids = len(subgrids)

print(f"Number of subgrids: {num_subgrids}")

# Create a figure to hold all subgrid plots
nrows = num_subgrids // 4 + 1
ncols = num_subgrids // nrows + 1
fig, axes = plt.subplots(
    nrows=nrows, ncols=ncols, figsize=(4, nrows)
)

# Flatten the axes array for easier iteration
axes = axes.flatten()

# Plot each subgrid
for subgrid, pattern in subgrids.items():
    index = pattern.index
    print(f"{pattern}")
    axes[index].imshow(subgrid, cmap="viridis", vmin=0, vmax=1)
    axes[index].axis("off")
    axes[index].set_title(f"Sub {index}")

# Turn off any unused subplots
for j in range(i + 1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()

# print(
#     f"{np.array2string(new_grid, separator=', ', threshold=np.inf, max_line_width=np.inf)}"
# )

# Create a custom colormap with more distinct colors
num_colors = len(np.unique(new_grid))
cmap = plt.cm.tab20

# Plot new_grid
# plt.figure(figsize=(12, 12))
plt.imshow(new_grid, cmap=cmap, interpolation='nearest')
plt.colorbar(label="Subgrid Index", ticks=range(num_colors))
plt.title("New Grid - Subgrid Indices")
plt.xlabel("Column")
plt.ylabel("Row")

# Add text labels and borders for each cell
for i in range(new_grid.shape[0]):
    for j in range(new_grid.shape[1]):
        plt.text(j, i, str(new_grid[i, j]), ha='center', va='center', color='white', fontweight='bold')
        plt.gca().add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor='black', linewidth=0.5))

plt.tight_layout()
plt.show()

# plot the adjacencies for each subgrid
def plot_subgrid_adjacencies(subgrids):
    for subgrid, pattern in subgrids.items():
        fig, ax = plt.subplots(3, 3, figsize=(10, 10))
        fig.suptitle(f"Subgrid {pattern.index} Adjacencies")

        # Plot the central subgrid
        ax[1, 1].imshow(subgrid, cmap="viridis", vmin=0, vmax=1)
        ax[1, 1].set_title(f"Sub {pattern.index}")
        ax[1, 1].axis('off')

        # Plot adjacent subgrids
        for (dx, dy) in dir8:
            row, col = 1 + dx // k, 1 + dy // k
            adj_indices = pattern.adjecency.get((dx, dy), set())
            if adj_indices:
                rows = 3
                cols = 3
                for i, adj_index in enumerate(adj_indices):
                    adj_subgrid = next(sg for sg, pt in subgrids.items() if pt.index == adj_index)
                    r, c = i // cols, i % cols
                    vertical_spacing = 0.3  # amount of vertical spacing between subgrids
                    sub_ax = ax[row, col].inset_axes([c/cols, ((rows-1-r) - vertical_spacing*r)/rows, 1/cols, 1/rows])
                    sub_ax.imshow(adj_subgrid, cmap="viridis", vmin=0, vmax=1)
                    sub_ax.set_title(f"Sub {adj_index}", fontsize=8)
                    sub_ax.axis('off')
                
                ax[row, col].axis('off')
            else:
                ax[row, col].set_title("None")
                ax[row, col].axis('off')

        plt.tight_layout()
        plt.show()

plot_subgrid_adjacencies(subgrids)
