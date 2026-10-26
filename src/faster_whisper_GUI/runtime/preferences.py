"""Application-owned YAML settings; no UI or inference imports."""

import os
from pathlib import Path
import re
import tempfile

import yaml

from .paths import PROJECT_ROOT

CONFIG_PATH = PROJECT_ROOT / 'config/config.yml'
SECTIONS = ('ui', 'model', 'transcription', 'vad', 'separation', 'settings', 'output')


class ConfigError(ValueError):
    """The configuration cannot be read, validated or saved."""


class _Loader(yaml.SafeLoader):
    yaml_implicit_resolvers = {
        key: [(tag, pattern) for tag, pattern in resolvers
              if tag not in {'tag:yaml.org,2002:bool', 'tag:yaml.org,2002:timestamp'}]
        for key, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
    }

    def construct_mapping(self, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or key in result:
                raise ConfigError('配置字段必须为不重复的字符串')
            result[key] = self.construct_object(value_node, deep=deep)
        return result


_Loader.add_implicit_resolver('tag:yaml.org,2002:bool', re.compile(r'^(?:true|false)$', re.I), list('tTfF'))


def _validate(data):
    if not isinstance(data, dict) or set(data) - set(SECTIONS):
        raise ConfigError('配置必须是包含已知配置节的映射')
    if any(not isinstance(section, dict) for section in data.values()):
        raise ConfigError('每个配置节必须是映射')
    ui = data.get('ui', {})
    if set(ui) - {'theme', 'theme_color', 'font_families'}:
        raise ConfigError('界面配置包含未知字段')
    if ui.get('theme', 'light') not in ('light', 'dark', 'auto'):
        raise ConfigError('主题必须是 light、dark 或 auto')
    color = ui.get('theme_color', '#009faa')
    if not isinstance(color, str) or not re.fullmatch(r'#[0-9a-fA-F]{6}(?:[0-9a-fA-F]{2})?', color):
        raise ConfigError('主题色必须是十六进制颜色')
    fonts = ui.get('font_families', [])
    if not isinstance(fonts, list) or any(not isinstance(font, str) or not font.strip() for font in fonts):
        raise ConfigError('字体必须是非空名称组成的列表')
    return data


def load_config(path=CONFIG_PATH):
    """Read YAML settings; errors contain no configuration values."""
    path = Path(path)
    if path.suffix.lower() not in ('.yml', '.yaml'):
        raise ConfigError('配置文件必须使用 .yml 或 .yaml 扩展名')
    try:
        data = yaml.load(path.read_text(encoding='utf-8-sig'), Loader=_Loader)
    except (OSError, UnicodeError, yaml.YAMLError):
        raise ConfigError(f'无法读取 YAML 配置：{path}') from None
    return _validate({} if data is None else data)


def validate_values(data, defaults):
    """Validate a partial configuration against the current controls' types."""
    for key, value in data.items():
        if key not in defaults:
            raise ConfigError('配置包含未知字段')
        expected = defaults[key]
        if isinstance(expected, dict):
            if not isinstance(value, dict):
                raise ConfigError('配置字段类型不匹配')
            validate_values(value, expected)
        elif isinstance(expected, float):
            if type(value) not in (int, float):
                raise ConfigError('配置字段类型不匹配')
        elif type(value) is not type(expected):
            raise ConfigError('配置字段类型不匹配')


def save_config(data, path=CONFIG_PATH):
    """Replace a YAML file only after its complete contents are written."""
    path = Path(path)
    if path.suffix.lower() not in ('.yml', '.yaml'):
        raise ConfigError('配置文件必须使用 .yml 或 .yaml 扩展名')
    _validate(data)
    temporary = None
    try:
        contents = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.config-', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except (OSError, yaml.YAMLError):
        raise ConfigError(f'无法保存 YAML 配置：{path}') from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
