"""把 runs/ 渲染成一个**静态**控制台页面（只读）。

为什么用静态生成而不是起服务：
    - 零依赖、零端口、拿到 HTML 就能看/发人；
    - 所有数字在生成时就固化，不会出现"页面和账本不一致"；
    - 需要重新看最新结果时重跑一次即可（几十毫秒）。

用法：
    python -m optframe.webui.build_static --root runs --out webui/out
    # 打开 webui/out/index.html

产物：
    out/index.html      单文件（内联 CSS/JS/数据，无外链）
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import report as rpt  # 数据读取逻辑复用（见同目录 report.py）

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"


def _inline(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build(root: Path, out: Path, *, limit: int = 40) -> Path:
    runs = rpt.scan_runs(root)
    snapshot: Dict[str, Any] = {"/api/runs": {"root": str(root), "runs": runs}}

    # 详情优先给"有瓶颈清单"的运行（清单页真正需要的），再补最近几个其他运行
    analysis = [r for r in runs if "analysis" in (r.get("kinds") or [])]
    others = [r for r in runs if "analysis" not in (r.get("kinds") or [])]
    picked = analysis[: max(1, limit // 2)] + others[: max(1, limit // 2)]
    for r in picked:
        detail = rpt.load_run(root, r["id"])
        if detail:
            snapshot[f"/api/runs/{r['id']}"] = detail

    # 相邻两次 analysis run 的 diff
    analyses = [r for r in runs if "analysis" in (r.get("kinds") or [])]
    for cur, prev in zip(analyses, analyses[1:]):
        key = f"/api/diff?cur={cur['id']}&prev={prev['id']}"
        snapshot[key] = rpt.diff_runs(root, cur["id"], prev["id"])

    html = _inline(STATIC / "index.html")
    html = html.replace('<link rel="stylesheet" href="/style.css">',
                        "<style>\n" + _inline(STATIC / "style.css") + "\n</style>")
    html = html.replace('<script src="/app.js"></script>',
                        "<script>window.SNAPSHOT="
                        + json.dumps(snapshot, ensure_ascii=False).replace("</", "<\\/")
                        + ";</script>\n<script>\n" + _inline(STATIC / "app.js") + "\n</script>")

    out.mkdir(parents=True, exist_ok=True)
    target = out / "index.html"
    target.write_text(html, encoding="utf-8")
    return target


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs")
    ap.add_argument("--out", default="webui/out")
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args(argv)

    target = build(Path(args.root), Path(args.out), limit=args.limit)
    size = target.stat().st_size
    print(f"[ok] {target}  ({size/1024:.0f} KB)")
    print(f"     打开方式：浏览器直接打开该文件（只读，无服务端）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
