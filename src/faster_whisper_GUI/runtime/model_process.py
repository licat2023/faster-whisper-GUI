"""Persistent isolated model process; one serialized request at a time."""
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import uuid
from .paths import PROJECT_ROOT, CACHE_ROOT, SOURCE_ROOT


class ModelProcess:
    def __init__(self, interpreter, timeout=3600):
        if not Path(interpreter).is_file():
            raise ValueError(f'独立运行环境不存在：{interpreter}')
        logs = CACHE_ROOT / 'logs'
        logs.mkdir(parents=True, exist_ok=True)
        self.log_path = logs / f'model-{uuid.uuid4().hex}.log'
        self.log = self.log_path.open('w', encoding='utf-8')
        env = os.environ.copy()
        env['PYTHONPYCACHEPREFIX'] = str(CACHE_ROOT / 'pycache')
        env['PYTHONUNBUFFERED'] = '1'
        env['PYTHONIOENCODING'] = 'utf-8'
        try:
            self.process = subprocess.Popen([str(interpreter), '-u', str(SOURCE_ROOT / 'project_tools/model_server.py')],
                cwd=PROJECT_ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                encoding='utf-8', creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except Exception:
            self.log.close()
            raise
        self.responses = queue.Queue()
        self.timeout = timeout
        self.lock = threading.Lock()
        self.close_lock = threading.Lock()
        self.closed = False
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        try:
            for line in self.process.stdout:
                try:
                    self.responses.put(json.loads(line))
                except ValueError:
                    self.responses.put({'error': '模型进程返回了无效响应'})
        except (OSError, ValueError):
            pass  # Shutdown may close the pipe while the reader leaves readline().
        finally:
            self.responses.put({'error': f'模型进程已结束；日志：{self.log_path}'})

    def request(self, action, **payload):
        with self.lock:
            if self.process.poll() is not None:
                raise RuntimeError(f'模型进程已退出；请重新加载模型。日志：{self.log_path}')
            try:
                self.process.stdin.write(json.dumps({'action': action, **payload}, ensure_ascii=False) + '\n')
                self.process.stdin.flush()
            except (OSError, ValueError) as error:
                raise RuntimeError(f'模型进程连接已关闭；日志：{self.log_path}') from error
            try:
                response = self.responses.get(timeout=self.timeout)
            except queue.Empty:
                self.close()
                raise TimeoutError(f'模型处理超时；日志：{self.log_path}')
            if not isinstance(response, dict) or not ({'error', 'result'} & response.keys()):
                self.terminate()
                raise RuntimeError(f'模型进程返回了无效响应；日志：{self.log_path}')
            if 'error' in response:
                raise RuntimeError(f"{response['error']}；日志：{self.log_path}")
            return response['result']

    def close(self):
        with self.close_lock:
            if self.closed:
                return
            self.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
            self.reader.join(timeout=1)
            for stream in [self.process.stdin, self.process.stdout]:
                if stream:
                    stream.close()
            self.log.close()
            self.closed = True

    def terminate(self):
        """Non-blocking cancellation of a request currently running in the child."""
        if self.process.poll() is None:
            try:
                self.process.terminate()
            except ProcessLookupError:
                pass
