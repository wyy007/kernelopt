"""optframe Web 控制台服务（只读视图 + 执行入口）。

零依赖：标准库 http.server。

    python -m optframe.webui.serve --root runs --port 8081
    # 打开 http://<host>:8081

接口：
    GET  /api/runs                 运行列表
    GET  /api/runs/<id>            单个运行详情
    GET  /api/diff?cur=&prev=      两次运行 SOL 对比
    GET  /api/actions              可用动作及参数
    POST /api/actions/<name>       提交任务（JSON body 为参数）
    GET  /api/jobs                 最近任务
    GET  /api/jobs/<id>            任务状态 + 日志尾部
    POST /api/jobs/<id>/cancel     中止
"""

from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, List, Optional
from urllib.parse import parse_qs, urlparse

from . import jobs, report

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
ROOT = Path("runs")


class Handler(BaseHTTPRequestHandler):
    server_version = "optframe-webui/0.2"

    def log_message(self, fmt: str, *args: Any) -> None:
        pass

    # ---- 基础 ----
    def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: Any, code: int = 200) -> None:
        self._send(json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8", code)

    def _static(self, rel: str) -> None:
        path = (STATIC / rel).resolve()
        if not str(path).startswith(str(STATIC.resolve())) or not path.is_file():
            return self._json({"error": "not found"}, 404)
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype == "application/javascript":
            ctype += "; charset=utf-8"
        self._send(path.read_bytes(), ctype)

    # ---- 路由 ----
    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        parts = [p for p in url.path.split("/") if p]
        qs = parse_qs(url.query)

        if not parts:
            return self._static("index.html")

        if parts[0] != "api":
            return self._static("/".join(parts))

        rest = parts[1:]
        if rest == ["runs"]:
            return self._json({"root": str(ROOT), "runs": report.scan_runs(ROOT)})
        if len(rest) == 2 and rest[0] == "runs":
            run = report.load_run(ROOT, rest[1])
            return self._json(run or {"error": "not found"}, 200 if run else 404)
        if rest == ["diff"]:
            cur = (qs.get("cur") or [""])[0]
            prev = (qs.get("prev") or [""])[0]
            return self._json(report.diff_runs(ROOT, cur, prev))
        if rest == ["actions"]:
            return self._json({"actions": [
                {"name": a.name, "title": a.title, "desc": a.desc,
                 "params": a.params, "needs_gpu": a.needs_gpu}
                for a in jobs.ACTIONS.values()]})
        if rest == ["jobs"]:
            return self._json({"jobs": jobs.recent()})
        if len(rest) == 2 and rest[0] == "jobs":
            job = jobs.get(rest[1])
            if job is None:
                return self._json({"error": "not found"}, 404)
            payload = job.to_dict()
            payload["log_tail"] = jobs.log_tail(job.id)
            return self._json(payload)

        return self._json({"error": "unknown api"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        parts = [p for p in url.path.split("/") if p]
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            return self._json({"error": "invalid json"}, 400)

        if len(parts) == 3 and parts[0] == "api" and parts[1] == "actions":
            try:
                job = jobs.submit(parts[2], body or {})
            except jobs.ParamError as exc:
                return self._json({"error": str(exc)}, 400)
            return self._json(job.to_dict(), 201)

        if len(parts) == 4 and parts[0] == "api" and parts[1] == "jobs" and parts[3] == "cancel":
            ok = jobs.cancel(parts[2])
            return self._json({"canceled": ok})

        return self._json({"error": "unknown api"}, 404)


def main(argv: Optional[List[str]] = None) -> int:
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8081)
    args = ap.parse_args(argv)

    ROOT = Path(args.root)
    if not ROOT.exists():
        print(f"[FAIL] runs root not found: {ROOT}")
        return 2
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"optframe webui -> http://{args.host}:{args.port} (root={ROOT})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
