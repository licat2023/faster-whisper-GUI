"""源码位置与运行时数据目录。"""

from pathlib import Path
import sys
import os

PACKAGE_DIR = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE_DIR.parent
# 源码和 editable 安装保留仓库数据；wheel 和冻结安装使用用户数据目录。
if SOURCE_ROOT.name == 'src' and not getattr(sys, 'frozen', False):
    PROJECT_ROOT = SOURCE_ROOT.parent
else:
    user_data = Path(os.environ.get('LOCALAPPDATA') or os.environ.get('XDG_DATA_HOME') or str(Path.home() / '.local/share'))
    PROJECT_ROOT = user_data / 'faster-whisper-GUI'

CACHE_ROOT = PROJECT_ROOT / ".cache"
MODEL_CACHE_DIR = CACHE_ROOT / "models"
