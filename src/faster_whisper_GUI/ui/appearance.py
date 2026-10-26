"""Apply and collect Qt appearance without framework-owned persistence."""

from PySide6.QtGui import QColor
from qfluentwidgets import qconfig, setTheme, setThemeColor, Theme


def apply_appearance(settings):
    if 'font_families' in settings:
        qconfig.set(qconfig.fontFamilies, settings['font_families'], save=False)
    if 'theme' in settings:
        setTheme(Theme(settings['theme'].capitalize()), save=False, lazy=True)
    if 'theme_color' in settings:
        setThemeColor(settings['theme_color'], save=False, lazy=True)


def appearance_settings():
    return {
        'theme': qconfig.get(qconfig.themeMode).value.lower(),
        'theme_color': qconfig.get(qconfig.themeColor).name(QColor.NameFormat.HexArgb),
        'font_families': list(qconfig.get(qconfig.fontFamilies)),
    }
