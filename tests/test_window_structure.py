"""The window is assembled from page mixins; keep that assembly honest."""

import inspect
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class WindowStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from this_voice_thing.ui import main_window
        cls.window_class = main_window.ChatterboxApp

    def own_members(self, klass):
        return {name for name, value in vars(klass).items()
                if inspect.isfunction(value) or isinstance(value, staticmethod)}

    def test_no_method_is_defined_twice(self):
        """A name defined in two mixins would silently shadow one of them."""
        seen = {}
        for klass in self.window_class.__mro__:
            if not klass.__module__.startswith("this_voice_thing."):
                continue
            for name in self.own_members(klass):
                self.assertNotIn(name, seen, f"{name} is defined in both {seen.get(name)} and {klass.__name__}")
                seen[name] = klass.__name__

    def test_every_page_has_a_builder(self):
        for name in ("_build_generate_page", "_build_voice_page", "_build_transcribe_page",
                     "_build_model_page", "_build_advanced_page", "_build_log_page"):
            self.assertTrue(callable(getattr(self.window_class, name, None)), name)

    def test_mixins_stay_plain(self):
        """Mixins hold methods only; Qt base classes belong to the window itself."""
        for klass in self.window_class.__mro__[1:]:
            if klass.__module__.startswith("this_voice_thing.ui.pages."):
                self.assertEqual(klass.__bases__, (object,), klass.__name__)


if __name__ == "__main__":
    unittest.main()
