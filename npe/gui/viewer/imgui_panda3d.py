from imgui.integrations.opengl import ProgrammablePipelineRenderer, imgui
from direct.showbase.DirectObject import DirectObject
from panda3d.core import PythonCallbackObject

class Panda3DRenderer(ProgrammablePipelineRenderer, DirectObject):
    def __init__(self, base, callback):
        imgui.create_context()

        super(ProgrammablePipelineRenderer, self).__init__()
        super(DirectObject, self).__init__()

        self.panda_base = base
        self.io.display_size = (base.win.get_x_size(), base.win.get_y_size())
        
        base.cam2d.node().get_display_region(0).set_draw_callback(PythonCallbackObject(callback))
        
        for event in self._get_events():
            self.accept(event, self.event_handler, [event])

    def _get_events(self):
        return [
            'mouse1',
            'mouse2',
            'mouse3',
            'mouse1-up',
            'mouse2-up',
            'mouse3-up',
            'wheel_up',
            'wheel_down',
            'window-event'
        ]

    def event_handler(self, event, *args):
        io = self.io

        if event == 'mouse1':
            io.mouse_down[0] = 1
        if event == 'mouse1-up':
            io.mouse_down[0] = 0
        if event == 'mouse2':
            io.mouse_down[1] = 1
        if event == 'mouse2-up':
            io.mouse_down[1] = 0
        if event == 'mouse3':
            io.mouse_down[2] = 1
        if event == 'mouse3-up':
            io.mouse_down[2] = 0
        if event == 'wheel_up':
            io.mouse_wheel = .5
        if event == 'wheel_down':
            io.mouse_wheel = -.5
        if event == 'window-event':
            io.display_size = (base.win.get_x_size(), base.win.get_y_size())

    def _get_mouse_window_pos(self):
        x = base.mouseWatcherNode.get_mouse_x()
        x = ((base.win.get_x_size() / 2)) * (x+1)

        y = -base.mouseWatcherNode.getMouseY()
        y = ((base.win.get_y_size() / 2)) * (y+1)
        return (x, y)

    def render(self, draw_data):
        if base.mouseWatcherNode.has_mouse():
            self.io.mouse_pos = self._get_mouse_window_pos()
        super().render(draw_data)