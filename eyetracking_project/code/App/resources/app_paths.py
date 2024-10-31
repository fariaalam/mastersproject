# other imports
from pathlib import Path

# Path to the root of the app
APP_ROOT = Path(__file__).parent.parent

# Path to the root of the Project
PROJECT_ROOT = APP_ROOT.parent.parent

# Path to the data directory
DATA_PATH = PROJECT_ROOT / 'data'

# Path to the kv directory
KV_PATH = APP_ROOT / 'resources' / 'kv'