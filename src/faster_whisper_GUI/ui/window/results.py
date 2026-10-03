"""结果表格生命周期及简繁转换。"""

import logging
import os
from faster_whisper_GUI.ui.models.segments import TableModel
from faster_whisper_GUI.ui.widgets.result_table import CustomTableView
import opencc
from qfluentwidgets import MessageBox

log = logging.getLogger(__name__)

class ResultActions:
    """MainWindows 的内部职责分组，共用窗口状态，不创建额外 QObject。"""

    def changeTableData(self, results) -> None:
        updated = {result[1].replace("\\", "/"): result for result in results}
        tabs = self.page_output.tableTab
        for index in range(len(tabs.tabBar.items) - 1, -1, -1):
            item = tabs.tabBar.items[index]
            path = item.routeKey().removeprefix('tab_').replace("\\", "/")
            if path not in updated:
                widget = tabs.widgetForRouteKey(item.routeKey())
                tabs.stackedWidget.removeWidget(widget)
                tabs.tabBar.removeTab(index)
                widget.deleteLater()
                self.tableModel_list.pop(path, None)
            else:
                model = self.tableModel_list[path]
                if model._data is not updated[path][0]:
                    model.resetData(updated[path][0])
        for path, result in updated.items():
            if path not in self.tableModel_list:
                self.createResultInTable([result])

    def mergeNewResults(self, results):
        """Append task/import results, retaining edits for unrelated files."""
        merged = {entry[1].replace('\\', '/'): entry for entry in (self.current_result or [])}
        for entry in results:
            path = entry[1].replace('\\', '/')
            model = self.tableModel_list.get(path)
            if model is not None and model.isModified:
                if getattr(self, '_closingRequested', False):
                    continue
                dialog = MessageBox('替换字幕',
                    f'{os.path.basename(path)} 有未保存的编辑，是否用新结果替换？', self)
                if not dialog.exec():
                    continue
            merged[path] = entry
        return list(merged.values())

    def showResultInTable(self, results):

        # tabBarItems = self.page_output.tableTab.tabBar.items

        # # 遍历表格标签 
        # for tabBarItem in tabBarItems:
        #     #　清理掉已经过时的结果
        #     index = tabBarItems.index(tabBarItem)
        #     self.page_output.tableTab.tabBar.removeTab(index)
        
        # # 遍历stack下的表格控件
        # for i in range(self.page_output.tableTab.stackedWidget.count()):
        #     widget = self.page_output.tableTab.stackedWidget.widget(i)
        #     # 移除全部表格控件
        #     self.page_output.tableTab.stackedWidget.removeWidget(widget)
        
        # 创建数据表
        # self.createResultInTable(results=results)

        if len(self.tableModel_list) == 0:
            log.info("%s", "Create Tables")
            self.createResultInTable(results=results)
        else:
            log.info("%s", "UPdata DataModel")
            self.changeTableData(results)

    def createResultInTable(self, results):
        i = len(self.page_output.tableTab.tabBar.items)
        for result in results:
            segments, file, _ = result
            file = file.replace("\\", "/")
            
            table_view_widget = CustomTableView()
            table_model = TableModel(segments)

            self.tableModel_list[file] = table_model

            _,text = os.path.split(file)
            table_view_widget.setObjectName(f"tab_{file}")
            table_view_widget.setModel(self.tableModel_list[file])

            self.page_output.tableTab.addSubInterface(
                                                        table_view_widget
                                                        , f"tab_{file}" 
                                                        , text
                                                        , None
                                                    )
            
            i += 1
        
        log.info("%s", f"len_model: {len(self.tableModel_list)}")

    def simplifiedAndTraditionalChineseConvert(self, segments, language):
        # 設置轉換器
        #
        # 只有 "Auto" / "zhs" / "zht" 三种取值有对应的转换方向。旧实现在其它取值下
        # 会让 cc 保持未绑定，随后的 cc.convert() 抛 UnboundLocalError —— 而调用点
        # （transcribeOver）没有 try/except，于是「音频识别为中文、但用户在转写参数页
        # 选的是粤语等其它语言」这一组合会直接打断转写结果展示。这里显式跳过并记日志。
        if language == "Auto" or language == "zhs":
            log.info("%s", "convert to Simplified Chinese")
            log.info("%s", f"len:{len(segments)}")
            cc = opencc.OpenCC('t2s')

        elif language == "zht":
            log.info("%s", "convert to Traditional Chinese")
            log.info("%s", f"len:{len(segments)}")
            cc = opencc.OpenCC('s2t')

        else:
            log.warning("语言 %r 没有简繁转换方向，跳过简繁转换", language)
            return

        # 轉換簡繁
        for segment in segments:

            new_text = cc.convert(segment.text)
            segment.text = new_text

            if len(segment.words) > 0:

                for word in segment.words:
                    new_word = cc.convert(word.word)
                    # 原先写作 word = Word(word.start, word.end, new_word, ...)：
                    # 那只是把循环变量重绑到一个新对象上，既没有写回原对象，也不影响
                    # segment.words 里的元素 —— 词级文本实际上从未被转换（正确写法被
                    # 注释在下一行）。faster_whisper 的 Word 是可变对象，直接改字段。
                    word.word = new_word

    def deleteResultTableEvent(self, routeKey:str):

        file_key = routeKey.removeprefix('tab_').replace("\\", "/")
        self.tableModel_list.pop(file_key, None)
        for name in ['current_result', 'result_faster_whisper', 'result_whisperx_aligment', 'result_whisperx_speaker_diarize']:
            result = getattr(self, name, None)
            if result is not None:
                result[:] = [entry for entry in result if entry[1].replace("\\", "/") != file_key]
                if not result:
                    setattr(self, name, None)
