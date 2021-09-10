
from __future__ import annotations

import sys
import os
from enum import Enum
import imgui
from imgui.integrations.opengl import *
from .imgui_panda3d import *

from direct.showbase.ShowBase import ShowBase
from panda3d.core import *
from direct.showbase.InputStateGlobal import inputState
from .object_picker import ObjectPicker
from .geom_primitives import GeomPrimitives
from .ground import Ground


class GUI:
    app: PandaViewer

    def __init__(self):
        self.app = None

    def display(self):
        raise NotImplemented


class MouseButton(Enum):
    LEFT = 1
    RIGHT = 0
    MIDDLE = 2


class PandaViewer(ShowBase):
    default_background_color = (233/255, 228/255, 199/255)

    def __init__(self, win_size="1280 768", win_title="PandaViewer"):
        print("current dir: " + os.getcwd())

        load_prc_file_data('', 'win-size ' + win_size)
        load_prc_file_data('', 'window-title ' + win_title)

        # disable synchro with screen
        load_prc_file_data('', 'sync-video #f')
        load_prc_file_data('', 'default-antialias-enable 1')
        load_prc_file_data('', 'framebuffer-multisample 1')

        ShowBase.__init__(self)

        self.background_color = PandaViewer.default_background_color
        base.set_background_color(*self.background_color, 1.0)
        self.interactive_debug = False

        # The pressed buttons list will be populated by mouse event handlers
        self.pressed_buttons = []

        # add axis/ground to the scene graph
        self.axis3D = loader.load_model('misc/xyzAxis')
        self.axis3D.set_pos(0, 0, 0)
        # axis length=1 unit, so scale=0.1 since the 3Daxis mesh is 10 units
        self.axis3D.set_scale(0.1)
        self.axis3D.reparent_to(render)

        self.ground = Ground(render)

        # Imgui buttons toggles -
        self.ik_requested = False
        self.ik_toggle = False

        # create usefull geom/NodePath but do not show them
        # TODO: instanciate the primitives when we need them
        self.sphere = GeomPrimitives.Sphere(6)
        self.cubeRBG = loader.load_model('misc/rgbCube')
        self.axis_line = self.draw_axes()
        self.cylinder = GeomPrimitives.Cylinder(7)
        self.cube = GeomPrimitives.Cube()

        plight_data = PointLight('plight')
        plight_data.set_color((0.9, 0.9, 0.9, 1))  # LIGHT COLOR
        # plight_data.set_shadow_caster(False)
        self.plight = render.attach_new_node(plight_data)
        self.plight.set_pos(0, 0, 50)
        render.set_light(self.plight)
        # self.sphere.instance_to(self.plight)

        # Ambiant light
        # alight = AmbientLight('alight')
        # alight.set_color((0.05, 0.05, 0.05, 1))
        # self.alight = render.attach_new_node(alight)
        # render.set_light(self.alight)

        # Create "showcase lights"
        for i in (-1, 1):
            for j in (i, -i):
                plight = PointLight('plight')
                plight.set_color((0.9, 0.9, 0.9, 1))
                plight = render.attach_new_node(plight)
                plight.set_pos(i * 50, j * 50, 50)
                self.sphere.instance_to(plight)
                render.set_light(plight)

        render.setAntialias(AntialiasAttrib.MAuto)

        # CAMERA stuff
        # dummy node for camera, we will rotate the dummy node fro camera rotation
        self.camera_node = render.attach_new_node('camera_parent')
        self.camera_node.set_effect(
            CompassEffect.make(render))  # NOT inherit rotation

        # the camera
        base.cam.reparent_to(self.camera_node)
        base.cam.look_at(self.camera_node)
        base.cam.set_pos(0, -10, 2)  # camera distance from model

        self.object_picker = ObjectPicker(base, render)
        self.selected_object = None

        self.selected_original_color = None
        self.selected_color = (237/255, 180/255, 47/255)
        self.held_object = None

        # vars for camera rotation
        self.last_mouse_pos = (-1, -1)

        # camera control
        base.accept('wheel_up', lambda: base.cam.setY(
            base.cam.getY()+200 * globalClock.getDt()))
        base.accept('wheel_down', lambda: base.cam.setY(
            base.cam.getY()-200 * globalClock.getDt()))
        base.accept('arrow_left', lambda: base.cam.setX(
            base.cam.getX()-20 * globalClock.getDt()))
        base.accept('arrow_right', lambda: base.cam.setX(
            base.cam.getX()+20 * globalClock.getDt()))
        base.accept('page_up', lambda: base.cam.setZ(
            base.cam.getZ()+20 * globalClock.getDt()))
        base.accept('page_down', lambda: base.cam.setZ(
            base.cam.getZ()-20 * globalClock.getDt()))
        base.accept('control-arrow_up',
                    lambda: base.plight.setY(base.plight.getY()+20 * globalClock.getDt()))
        base.accept('control-arrow_down',
                    lambda: base.plight.setY(base.plight.getY()-20 * globalClock.getDt()))
        base.accept('control-arrow_left',
                    lambda: base.plight.setX(base.plight.getX()-20 * globalClock.getDt()))
        base.accept('control-arrow_right',
                    lambda: base.plight.setX(base.plight.getX()+20 * globalClock.getDt()))
        base.accept('control-page_up',
                    lambda: base.plight.setZ(base.plight.getZ()+20 * globalClock.getDt()))
        base.accept('control-page_down',
                    lambda: base.plight.setZ(base.plight.getZ()-20 * globalClock.getDt()))

        base.accept('control-arrow_up-repeat',
                    lambda: messenger.send('control-arrow_up'))
        base.accept('control-arrow_down-repeat',
                    lambda: messenger.send('control-arrow_down'))
        base.accept('control-arrow_right-repeat',
                    lambda: messenger.send('control-arrow_right'))
        base.accept('control-arrow_left-repeat',
                    lambda: messenger.send('control-arrow_left'))
        base.accept('control-page_up-repeat',
                    lambda: messenger.send('control-page_up'))
        base.accept('control-page_down-repeat',
                    lambda: messenger.send('control-page_down'))

        base.accept('arrow_up', lambda: messenger.send('wheel_up'))
        base.accept('arrow_down', lambda: messenger.send('wheel_down'))
        base.accept('page_up-repeat', lambda: messenger.send('page_up'))
        base.accept('page_down-repeat', lambda: messenger.send('page_down'))
        base.accept('arrow_left-repeat', lambda: messenger.send('arrow_left'))
        base.accept('arrow_right-repeat',
                    lambda: messenger.send('arrow_right'))
        base.accept('arrow_up-repeat', lambda: messenger.send('arrow_up'))
        base.accept('arrow_down-repeat', lambda: messenger.send('arrow_down'))
        # base.accept('wheel_up' or 'arrow_up', lambda : base.cam.setY(base.cam.getY()+200 * globalClock.getDt()))
        # base.accept('wheel_down', lambda : base.cam.setY(base.cam.getY()-200 * globalClock.getDt()))

        # Add the spinCameraTask procedure to the task manager.
        self.accept("mouse1", self.mouse_click, [MouseButton.RIGHT, True])
        self.accept("mouse1-up", self.mouse_click, [MouseButton.RIGHT, False])
        self.accept("mouse3", self.mouse_click, [MouseButton.LEFT, True])
        self.accept("mouse3-up", self.mouse_click, [MouseButton.LEFT, False])

        inputState.watchWithModifiers('mouse1', 'mouse1')
        inputState.watchWithModifiers('mouse2', 'mouse2')

        self.taskMgr.add(self.mouse_motion_task, "mouse_motion_task")
        self.animation_task = None
        # self.taskMgr.stop()

        # list of key
        self.add_key("q", "Quit")
        self.add_key("escape", "Quit")
        self.add_key("a", "Toggle drawing of axis")
        self.add_key("g", "Toggle drawing of the ground")
        self.add_key("w", "Toggle wireframe")
        self.add_key("f", "Toggle frame rate")
        self.add_key("h", "Print this help")
        self.add_key("z", "Animation: run/stop")
        self.add_key("n", "Animation: next frame")
        self.add_key("b", "Animation: back one frame")
        self.add_key("control-i", "Interactive Python interpreter")
        self.add_key("control-d", "Interactive Python debug")
        self.add_key("control-l", "reLoad the code")
        self.add_key("control-h", "hide UI")
        self.add_key("control-r", "record screen on/off")

        # Make sure we're running OpenGL.
        if base.pipe.getInterfaceName() != 'OpenGL':
            print("This program requires OpenGL.")
            sys.exit(1)

        self.hide_ui = False
        self.imgui_renderer = Panda3DRenderer(base, self.draw_imgui)
        self._external_guis = []

    def register_gui(self, gui: GUI) -> GUI:
        """Register a GUI that will be displayed on screen.
        @return The GUI object updated with a reference to this viewer. 
        """
        # TODO: assert that gui is a GUI...
        gui.app = self
        # TODO: populate a dict so we can access the guis easily if necessary
        self._external_guis.append(gui)
        return gui

    def main_gui(self):
        if self.selected_object:
            imgui.begin("Inspector")
            imgui.text("Object selected: " + str(self.selected_object))
            imgui.text("Position")
            _, x = imgui.drag_float("x", self.selected_object.get_x())
            _, y = imgui.drag_float("y", self.selected_object.get_y())
            _, z = imgui.drag_float("z", self.selected_object.get_z())
            self.selected_object.set_pos(x, y, z)
            if imgui.button("Hide"):
                self.selected_object.detach_node()

            imgui.end()

    def draw_imgui(self, cbdata):
        if self.hide_ui:
            return

        imgui.new_frame()
        self.main_gui()
        for gui in self._external_guis:
            gui.display()
        self.end_imgui(cbdata)

    def end_imgui(self, cbdata):
        imgui.render()
        imgui.end_frame()
        self.imgui_renderer.render(imgui.get_draw_data())
        cbdata.upcall()

    def set_title(self, title):
        win_props = base.get_properties()
        win_props.set_title(title)
        base.request_properties(win_props)

    def add_key(self, k, helptxt=""):
        base.accept(k, self.key_handler, [k])
        base.accept(k+"-repeat", self.key_handler, [k])
        if not hasattr(self, 'keyhelp'):
            self.keyhelp = {}
        self.keyhelp[k] = helptxt

    """Handle click"""

    def mouse_click(self, button, down):
        # TODO: Ignore when on top of UI elements
        if down:
            if imgui.core.is_any_item_active() or imgui.core.is_any_item_focused() or imgui.core.is_any_item_hovered():
                return
            self.pressed_buttons.append(button)
            md = base.win.get_pointer(0)
            self.last_mouse_pos = (md.get_x(), md.get_y())

            # First we check that the mouse is not outside the screen.
            if base.mouseWatcherNode.has_mouse():
                # This gives up the screen coordinates of the mouse.
                mpos = base.mouseWatcherNode.get_mouse()
                selected = self.object_picker.pick(mpos.x, mpos.y)
                self.held_object = None
                if selected:
                    if self.selected_object:
                        self.selected_object.set_color(
                            *self.selected_original_color)
                    self.selected_object = selected
                    self.selected_original_color = self.selected_object.get_color()
                    self.selected_object.set_color(*self.selected_color)
                    self.held_object = self.selected_object
                else:
                    if self.selected_object:
                        self.selected_object.set_color(
                            *self.selected_original_color)
                    self.selected_object = None
                    self.held_object = None

        else:
            if button in self.pressed_buttons:  # Button is not pressed if we click on the ui and release in the window
                self.pressed_buttons.remove(button)

        if not down and button == MouseButton.LEFT:
            self.last_mouse_pos = (-1, -1)  # ?

        # TODO: Move object as well

    def is_mouse_active(self):
        return len(self.pressed_buttons) > 0 or imgui.is_any_item_active()

    # camera rotation/translation task according to mouse movements
    def mouse_motion_task(self, task):
        if len(self.pressed_buttons) == 0:
            return task.cont

        xold, yold = self.last_mouse_pos
        md = base.win.get_pointer(0)
        x = md.get_x()
        y = md.get_y()
        # TODO: This might only need to happen if a button is pressed
        self.last_mouse_pos = (x, y)

        if MouseButton.LEFT in self.pressed_buttons:
            self.camera_node.set_hpr(
                self.camera_node.get_h() - (x - xold) * 0.5,
                self.camera_node.get_p() - (y - yold) * 0.5,
                0)  # No roll

        if MouseButton.RIGHT in self.pressed_buttons:
            if self.held_object:  # If an object is grabbed, move it instead of the camera
                # TODO: Find a good way to place the object on the mouse
                # ex: raycast from pointer to plane of object ?
                self.held_object.set_pos(self.camera_node,
                                         self.held_object.get_x(self.camera_node) + (x - xold) * 0.01,
                                         self.held_object.get_y(self.camera_node),
                                         self.held_object.get_z(self.camera_node) - (y - yold) * 0.01
                                         )

            else:  # Update the camera pos
                base.cam.set_pos(
                    base.cam.get_x() - (x - xold) * 0.01,
                    base.cam.get_y(),
                    base.cam.get_z() + (y - yold) * 0.01)

        return task.cont

    def draw_axes(self):
        length = 10
        lines = LineSegs()
        # x
        lines.set_color(1.0, 0.0, 0.0, 1.0)
        lines.move_to(0, 0, 0)
        lines.draw_to(length, 0, 0)
        # y
        lines.set_color(0.0, 1.0, 0.0, 1.0)
        lines.move_to(0, 0, 0)
        lines.draw_to(0, length, 0)
        # z
        lines.set_color(0.0, 0.0, 1.0, 1.0)
        lines.move_to(0, 0, 0)
        lines.draw_to(0, 0, length)

        lines.set_thickness(1)
        node = lines.create()
        return NodePath(node)

    # key handler
    def key_handler(self, key):
        if key == "q" or key == "escape":
            print("quit")
            # sys.exit(0)
            self.stop()
        elif key == 'a':
            if not self.axis3D.has_parent():
                self.axis3D.reparent_to(render)
            else:
                self.axis3D.detach_node()
        elif key == 'g':
            self.ground.toggle()
        elif key == 'w':
            if render.has_render_mode():
                render.clear_render_mode()
                # render.setRenderModeFilled()
            else:
                render.set_render_mode_wireframe()
        elif key == 'h':
            self.help()
        elif key == 'n':
            self.animate()
        elif key == 'f':
            if base.frameRateMeter is None:
                base.setFrameRateMeter(True)
                print("FrameRate: on")
            else:
                base.setFrameRateMeter(False)
                print("FrameRate: off")
        elif key == "z":
            if self.is_animation_on():
                self.animation_stop()
            else:
                self.animation_start()
        elif key == "control-i":
            # from imp import reload
            # code.interact(local=locals())
            # print("rerun")
            # IPython.start_ipython(argv=[])
            #import IPython
            # IPython.start_ipython(argv=[])
            # IPython.embed()
            #global interactive_debug
            self.interactive_debug = True
            print("stop for debug")
            self.stop()
        elif key == "control-l":
            #from IPython.lib.deepreload import reload as dreload
            #mod = sys.modules[__file__]
            # dreload(mod)
            print("change 19")
        elif key == "control-b":
            breakpoint()
        elif key == "control-h":
            self.hide_ui = not self.hide_ui
        elif key == "control-r":
            self.toggle_screen_recording()
        else:
            print("key unmanaged: "+key)

    def toggle_screen_recording(self):
        base.movie(duration=15.0)

    def add_sphere_object(self, position=(0, 0, 0), scale=(1, 1, 1), tag=None):
        # TODO: which node is the parent ?
        sphere = render.attach_new_node("sphere")
        sphere.set_pos(*position)
        sphere.set_scale(scale)
        self.sphere.instance_to(sphere)

    def is_animation_on(self):
        return self.animation_task and self.animation_task.is_alive()

    def animation_start(self):
        self.animation_task = self.taskMgr.add(self.animate_task, "animate")
        print("anim start")

    def animation_stop(self):
        self.taskMgr.remove(self.animation_task)
        print("anim stop")

    def stop(self):
        # Set a flag so we will stop before beginning next frame
        #self.running = 0
        self.taskMgr.stop()

    def help(self):
        print("Help")
        print("Animation On/Off: "+str(self.is_animation_on()))
        for key in self.keyhelp:
            print(key, ": ", self.keyhelp[key])

    def animate(self):
        print("animate")

    def animate_task(self, task):
        self.animate()
        return task.cont

    def step(self):
        self.taskMgr.running = True
        self.taskMgr.step()
        return self.taskMgr.running


if __name__ == "__main__":
    app = PandaViewer()
    app.run()
    #global interactive_debug
    if app.interactive_debug == True:
        app.interactive_debug = False
        from IPython.lib.deepreload import reload as dreload
        import IPython
        import PandaViewer
        IPython.embed()
        # interactive_debug
        # interactive_debug=False

    print("ending...")
