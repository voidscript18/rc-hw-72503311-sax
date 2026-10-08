# -*- coding: utf-8 -*-
"""
T08 YOLO 推理 —— 作业（5 个函数）
"""
import os
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------- 1
def load_model(weight_path: str):
    """加载 YOLO 模型并返回模型对象。

    只给文件名时（如 "yolov8n.pt"），会在脚本目录、脚本目录的 assets/ 里找。
    """
    from ultralytics import YOLO

    if os.path.exists(weight_path):
        return YOLO(weight_path)
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (here, os.path.join(here, "assets"), "assets", "."):
        p = os.path.join(d, weight_path)
        if os.path.exists(p):
            return YOLO(p)
    return YOLO(weight_path)   # 都找不到就交给 ultralytics 报错


# ---------------------------------------------------------------- 2
def detect_objects(model, img, conf: float = 0.25) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
    """对一张 BGR 图跑推理，返回 [(类别名, 置信度, (x1,y1,x2,y2)), ...]，按置信度降序。"""
    results = model.predict(source=img, conf=conf, verbose=False)
    r = results[0]
    dets = []
    for b in r.boxes:
        cls_id = int(b.cls[0])
        name = r.names[cls_id]
        c = float(b.conf[0])
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        dets.append((name, c, (x1, y1, x2, y2)))
    dets.sort(key=lambda d: d[1], reverse=True)
    return dets


# ---------------------------------------------------------------- 3
def count_by_class(dets: List[Tuple[str, float, Tuple[int, int, int, int]]]) -> Dict[str, int]:
    """按类别名计数：{"person": 4, "bus": 1}"""
    counts = {}
    for name, _, _ in dets:
        counts[name] = counts.get(name, 0) + 1
    return counts


# ---------------------------------------------------------------- 4
def max_conf_target(dets, name: str):
    """返回指定类别中置信度最高的那一条；没有该类返回 None。"""
    best = None
    for d in dets:
        if d[0] == name and (best is None or d[1] > best[1]):
            best = d
    return best


# ---------------------------------------------------------------- 5
def draw_dets(img, dets, copy_img: bool = True) -> "np.ndarray":
    """把检测结果画到图上：绿框 + 顶部文字 "name conf"。"""
    import cv2
    out = img.copy() if copy_img else img
    for name, conf, (x1, y1, x2, y2) in dets:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = "%s %.2f" % (name, conf)
        cv2.putText(out, label, (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return out


if __name__ == "__main__":
    import sys
    import cv2

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    here = os.path.dirname(os.path.abspath(__file__))

    def _imread(p):
        return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)

    # 1 加载模型
    model = load_model("yolov8n.pt")

    # 2 读 bus.jpg 跑检测
    img = _imread(os.path.join(here, "assets", "bus.jpg"))
    dets = detect_objects(model, img, conf=0.25)
    print("检测到 %d 个目标（按置信度降序）：" % len(dets))
    for name, conf, box in dets:
        print("  %-10s %.3f  %s" % (name, conf, box))

    # 3 类别计数
    print("类别计数:", count_by_class(dets))

    # 4 最可信目标
    for name in ("person", "bus"):
        t = max_conf_target(dets, name)
        print("最可信 %s:" % name, t)

    # 5 画框保存结果图
    out = draw_dets(img, dets)
    out_dir = os.path.join(here, "..", "截图")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "T08_结果图.png")
    cv2.imencode(".png", out)[1].tofile(out_path)
    print("结果图 ->", out_path)
