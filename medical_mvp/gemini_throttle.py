"""
Gemini API 全局限流：按配置的每分钟峰值请求数（RPM）在相邻两次 generate_content 之间插入等待。

环境变量 MEDICAL_MVP_GEMINI_MAX_RPM 可覆盖 config.GEMINI_MAX_RPM（整数；0 或负表示不限制）。
"""

from __future__ import annotations

import os
import threading
import time

from medical_mvp import config

_lock = threading.Lock()
_last_request_mono = 0.0


def _effective_max_rpm() -> int:
    raw = os.environ.get("MEDICAL_MVP_GEMINI_MAX_RPM")
    if raw is not None and str(raw).strip() != "":
        try:
            return max(0, int(str(raw).strip(), 10))
        except ValueError:
            pass
    return max(0, int(config.GEMINI_MAX_RPM))


def before_gemini_request() -> None:
    """在每次调用 Client.models.generate_content 之前执行，压低峰值 RPM。"""
    rpm = _effective_max_rpm()
    if rpm <= 0:
        return

    interval = 60.0 / float(rpm)
    global _last_request_mono
    with _lock:
        now = time.monotonic()
        wait = interval - (now - _last_request_mono)
        if wait > 0:
            time.sleep(wait)
        _last_request_mono = time.monotonic()
