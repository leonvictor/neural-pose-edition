from panda3d.core import LineSegs
from .geom_primitives import GeomPrimitives

class Ground:
    default_color = (233, 228, 199)
    
    def __init__(self, render):
        """ make a ground with a grid. Size is 100x100 on XY plane """
        self.render = render
        self.node = render.attach_new_node("ground")

        # mesh of 40x40, for lighting, since lighting is computed per vertex
        self.ground_plane = GeomPrimitives.Plane(40)
        self.ground_plane.set_pos(0, 0, -0.01)                 # a little bit under 0
        self.ground_plane.set_scale(100, 100, 1)
        self.ground_plane.set_color(*Ground.default_color)
        self.ground_plane.reparent_to(self.node)

        # TODO: Add a parameter to toggle shadows
        self.ground_plane.set_shader_auto()

        self.grid_fine = GeomPrimitives.Grid(100, 100, 1,  1,  1, color=(0.2, 0.2, 0.2, 1.0))
        self.grid_large = GeomPrimitives.Grid(100, 100, 10, 10, 3, color=(0.2, 0.2, 0.2, 1.0))
        self.grid_large.reparent_to(self.node)
        self.grid_fine.reparent_to(self.node)

        # since the makePlane is 0..1; then scale 100, 100; and then translate -50, -50
        self.node.set_pos(-50, -50, 0)
    
    @property
    def enabled(self):
        return self.node.has_parent()

    @enabled.setter
    def enabled(self, enabled):
        self.show(enabled)

    @property 
    def grid_display(self):
        return self.grid_large.has_parent() and self.grid_fine.has_parent()

    @grid_display.setter
    def grid_display(self, on):
        self.show_grid(on)

    @property 
    def plane_display(self):
        return self.ground_plane.has_parent()

    @plane_display.setter
    def plane_display(self, on):
        self.show_plane(on)
    def toggle(self):
        self.show(not self.enabled)

    def show_grid(self, on: bool):
        if on:
            self.grid_large.reparent_to(self.node)
            self.grid_fine.reparent_to(self.node)
        else:
            self.grid_fine.detach_node()
            self.grid_large.detach_node()

    def show_plane(self, on: bool):
        if on:
            self.ground_plane.reparent_to(self.node)
        else:
            self.ground_plane.detach_node()

    # TODO: Show is a duplicate of the enabled setter
    def show(self, on: bool):
        if on:
            self.node.reparent_to(self.render)
        else:
            self.node.detach_node()