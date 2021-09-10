from npe.gui.viewer.PandaViewer import GUI
import imgui
import time


class StatsComparisonGUI(GUI):
    def __init__(self, exp_names):
        super().__init__()
        self._names = exp_names
        self._compute_times = {}
        self._times = {}

        self.reset()

        self._max_iter_memory = 1000

    def start_timer(self, exp_name):
        self._times[exp_name] = time.perf_counter()

    def stop_timer(self, exp_name):
        if self._times[exp_name] < 0:
            raise RuntimeError("tried to stop a timer that was not started.")

        self._update_value(exp_name, time.perf_counter() - self._times[exp_name])
        self._times[exp_name] = -1

    def _update_value(self, exp_name, value):
        # avoid memory leaks by removing older values
        if len(self._compute_times[exp_name]) > self._max_iter_memory:
            self._compute_times[exp_name] = self._compute_times[exp_name][1:]
        self._compute_times[exp_name].append(value)

    def reset(self):
        for n in self._names:
            self._compute_times[n] = []
            self._times[n] = -1

    def display(self):
        imgui.begin("Running stats")
        imgui.columns(3, "stats_list")

        imgui.separator()
        imgui.text("Method")
        imgui.next_column()
        imgui.text("Last step")
        imgui.next_column()
        imgui.text("Average")
        imgui.next_column()
        imgui.separator()

        # imgui.set_column_offset(1, 40)

        for name in self._names:
            imgui.text(name)
            imgui.next_column()

            if len(self._compute_times[name]) == 0:
                imgui.next_column()
                imgui.next_column()
                continue

            # imgui.next_column()
            val = self._compute_times[name][-1] * 1000
            imgui.text("%.2fms" % val)
            imgui.next_column()

            avg_time = sum(self._compute_times[name]) / len(self._compute_times[name]) * 1000
            imgui.text("%.2fms" % avg_time)
            imgui.next_column()

        imgui.separator()
        imgui.columns(1)
        if imgui.button("Clear"):
            self.reset()
        imgui.end()
