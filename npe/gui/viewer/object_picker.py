from panda3d.core import CollisionHandlerQueue, CollisionNode, CollisionRay, CollisionTraverser
from panda3d.core import GeomNode
from .PandaSkeleton import JointNode

class ObjectPicker:
    def __init__(self, base, render):
        # Setup base collision traverser https://docs.panda3d.org/1.10/python/programming/collision-detection/collision-traversers?highlight=traverser
        
        self.base = base
        self.render = render

        self.traverser = CollisionTraverser('base_traverser')
        self.base.cTrav = self.traverser
        self.ray = CollisionRay()
        
        # Setup the collision handler
        self.collision_handler = CollisionHandlerQueue()

        # setup picker https://docs.panda3d.org/1.10/python/programming/collision-detection/clicking-on-3d-objects
        node = CollisionNode('mouseRay')
        node_parent = base.cam.attach_new_node(node)
        node.set_from_collide_mask(GeomNode.get_default_collide_mask())
        node.add_solid(self.ray)
        self.traverser.add_collider(node_parent, self.collision_handler)
    
    def pick(self, x, y):
        # This makes the ray's origin the camera and makes the ray point
        # to the screen coordinates of the mouse.
        self.ray.set_from_lens(self.base.camNode, x, y)
        self.traverser.traverse(self.render)

        if self.collision_handler.get_num_entries() <= 0:
            return

        # This is so we get the closest object
        self.collision_handler.sort_entries()
        picked_object = self.collision_handler.get_entry(0).get_into_node_path()
        picked_object = picked_object.find_net_tag('selectable')
        if picked_object.has_python_tag("owner"):
            picked_object = picked_object.get_python_tag("owner")
        return picked_object