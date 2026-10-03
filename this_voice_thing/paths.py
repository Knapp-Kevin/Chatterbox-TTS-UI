"""Where things live on disk.

The code is in the this_voice_thing package, but the app's data stays in the
project folder it always used (models.json, app_settings.json, chatterbox_outputs/,
reference_recordings/, voice_library/, engines/<name>/.venv, logs/), so moving
the code doesn't strand anyone's files.
"""

import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PACKAGE_DIR)          # the project folder
ASSETS_DIR = os.path.join(ROOT, "assets")
ENGINES_DIR = os.path.join(ROOT, "engines")   # engine workers and their environments
LOGS_DIR = os.path.join(ROOT, "logs")
