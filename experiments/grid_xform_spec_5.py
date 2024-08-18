# type: ignore

"""
Describe this transformation from grids to grids (ARC challenge).
Notice that transformations are described as predicates `Pre --> Post` where `Pre` describes the grid before the transformation, and `Post` the grid after the transformation.
To connect different predicates, `&&` is used for conjunction.
For example ` Obj1.getColor() == Obj2.getColor && Obj1.getSize() = (3,3)`.

Note that NestedGrid(origin, size) return a new nested grid, where there is a sub-grid of size `size` at each coordinate `(x,y)`. At coordinate `(0,0)`, the nested grid contains the grid of size `size` at position `origin` in the initial grid.
All operations are immutable. In particular setXXX returns a copy of the receives with some property set as in the argument.
"""


class GridTransformSpec:
    def Pre(self):
        Objs = self.find_objects()
        Obj = Objs.max()
        Origin = Obj.getOrigin()
        Size = Obj.getSize()
        NestedGrid = self.grid_to_subgrids(subgrid_size=Size, origin=Origin)
        return [NestedGrid, Obj]

    def Post():
        NG0, Obj0 = self.Pre()

        def clamp(p):
            return [max(min(p[0], 1), -1), max(min(p[1], 1), -1)]

        def is_cardinal_or_diagonal(p):
            return p[0] == 0 or p[1] == 0 or abs(p[0]) == abs(p[1])

        def new_color(p):
            Obj = NG0[clamp(p)]
            return Obj.getColor() if is_cardinal_or_diagonal(p): else 0
        NG1 = create_grid_of_objects(subgrid_size=Size,
                                     subgrid_func=lambda p: Obj.copy().setColor(get_color(p)))
        Grid1 = NG1.flatten()
        Grid1 = Grid1.shift(Origin)
        return Grid1
