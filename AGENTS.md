# 项目约定

- 修改模块边界或配置持久化前，阅读 `docs/ARCHITECTURE.md` 对应章节。
- 应用设置通过 `runtime/preferences.py` 读写；Qt 外观修改使用 `save=False`，由应用统一保存。
- 配置回归：`.venv\Scripts\python.exe -m unittest project_tools.check_preferences`。
- 界面回归：`.venv\Scripts\python.exe -m unittest project_tools.check_gui_interactions project_tools.check_startup`；测试使用隔离配置，不读取或保存个人设置。
