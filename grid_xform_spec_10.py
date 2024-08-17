# type: ignore

"""
Describe this transformation from grids to grids (ARC challenge).
"""


class GridTransformSpec:
    colors = ["blue", "red", "green" "yellow"]

    def Pre(self):
        Objs = self.find_objects()
        assert all(obj.getColor() == "grey" for obj in Objs1)
        Objs1 = sorted(Objs, key=lambda obj: obj.height)
        return Objs

    def Post(self):
        Objs = self.Pre()

        def obj_func(obj, index):
            color = colors[index]
            return obj.copy().setColor(color)
        return Grid.from_objects(Objs, obj_func=obj_func)
