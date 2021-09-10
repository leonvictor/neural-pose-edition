from npe.gui.viewer.PandaViewer import GUI
import imgui

class LimbLengthsGUI(GUI):
    def __init__(self, skeleton):
        super().__init__()
        self.starting_lengths = []
        self.current_lengths = []
        self._skeleton = skeleton

    def display(self):
        imgui.begin("Limb lengths")
        imgui.columns(3, "limbList")
        imgui.separator()
        imgui.text("Lengths")
        imgui.next_column()
        imgui.text("Starting")
        imgui.next_column()
        imgui.text("Current")
        imgui.next_column()
        imgui.separator()


        imgui.set_column_offset(1, 40)

        diff = 0
        for start, now in zip(self.starting_lengths, self.current_lengths):
            imgui.next_column()
            imgui.text("%.2f" % start)
            imgui.next_column()
            imgui.text("%.2f" % now)
            imgui.next_column()
            diff += abs(start-now)

        imgui.columns(1)
        
        imgui.text("Total difference")
        imgui.text("%.2f" % diff)
        
        imgui.end()
