# -*- coding: utf-8 -*-
"""
T05 OpenCV 基础 —— 作业（6 个必做函数 + 1 个选做）
"""
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------- 1
def load_image(path: str):
    """读入一张图片（BGR），失败返回 None。

    Windows 上 cv2.imread 遇到中文路径会返回 None（不报错），
    所以失败时用 imdecode 兜底，中文路径也能读。
    """
    if not path or not os.path.exists(path):
        return None
    img = cv2.imread(path)
    if img is None:
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return img


# ---------------------------------------------------------------- 2
def to_gray(img: "np.ndarray") -> "np.ndarray":
    """BGR 转灰度图，返回单通道图。"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------- 3
def crop_roi(img: "np.ndarray", x: int, y: int, w: int, h: int) -> "np.ndarray":
    """裁剪矩形区域。numpy 是 [y:y+h, x:x+w]，先高后宽，别写反。"""
    return img[y:y + h, x:x + w]


# ---------------------------------------------------------------- 4
def resize_keep(img: "np.ndarray", max_side: int = 640):
    """等比例缩放：长边等于 max_side，短边按比例。"""
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- 5
def draw_marker(img: "np.ndarray", cx: float, cy: float,
                text: Optional[str] = None) -> "np.ndarray":
    """在图上画质心：红色圆点(r=5) + 十字，可选在点右侧写文字。
    返回新图，不改动传入的原图。
    """
    out = img.copy()
    cx, cy = int(cx), int(cy)
    cv2.circle(out, (cx, cy), 5, (0, 0, 255), -1)                       # 红色圆点
    cv2.drawMarker(out, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)  # 十字
    if text is not None:
        cv2.putText(out, str(text), (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return out


# ---------------------------------------------------------------- 6
def read_frames(source, n: int = 10) -> List["np.ndarray"]:
    """读取帧序列的前 n 帧。source 可以是摄像头编号(int)或视频文件路径(str)。
    读不到就返回已读到的部分。记得 release。
    """
    cap = cv2.VideoCapture(source)
    frames = []
    for _ in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


# ---------------------------------------------------------------- 7（选做）
def mask_centroid(mask: "np.ndarray") -> Optional[Tuple[float, float]]:
    """求二值掩膜中白色区域的质心 (cx, cy)；没有白色像素返回 None。"""
    M = cv2.moments(mask)
    if M["m00"] == 0:
        return None
    return (M["m10"] / M["m00"], M["m01"] / M["m00"])


if __name__ == "__main__":
    import sys
    import tempfile

    # 让控制台中文不乱码（Windows 默认 GBK 会显示乱码）
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    # 造一张测试图：300x200，左边蓝、右边绿
    img = np.zeros((200, 300, 3), np.uint8)
    img[:, :150] = (255, 0, 0)
    img[:, 150:] = (0, 255, 0)

    # 1 load_image：写临时图再读回来
    tmp_png = os.path.join(tempfile.gettempdir(), "_t05_load.png")
    cv2.imwrite(tmp_png, img)
    loaded = load_image(tmp_png)
    print("1 load_image  shape =", loaded.shape)

    # 2 to_gray
    print("2 to_gray     shape =", to_gray(img).shape)

    # 3 crop_roi：x=50, y=30, w=100, h=80 -> (80, 100, 3)
    print("3 crop_roi    shape =", crop_roi(img, 50, 30, 100, 80).shape)

    # 4 resize_keep：长边 300 -> 150 -> (100, 150, 3)
    print("4 resize_keep shape =", resize_keep(img, 150).shape)

    # 5 draw_marker：画质心 + 十字 + 文字，存成结果图（中文路径用 tofile）
    marked = draw_marker(img, 150, 100, "center(150,100)")
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "截图")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "T05_结果图.png")
    cv2.imencode(".png", marked)[1].tofile(out_path)
    print("5 draw_marker 结果图 ->", out_path)

    # 6 read_frames：造 5 帧临时视频读回来
    tmp_avi = os.path.join(tempfile.gettempdir(), "_t05_frames.avi")
    vw = cv2.VideoWriter(tmp_avi, cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 64))
    for i in range(5):
        vw.write(np.full((64, 64, 3), i * 50, np.uint8))
    vw.release()
    print("6 read_frames 读到", len(read_frames(tmp_avi, 10)), "帧")

    # 7 mask_centroid：白圆质心应在圆心 (40, 60)
    mask = np.zeros((100, 100), np.uint8)
    cv2.circle(mask, (40, 60), 20, 255, -1)
    print("7 mask_centroid =", mask_centroid(mask))

    # 清理临时文件
    for f in (tmp_png, tmp_avi):
        if os.path.exists(f):
            os.remove(f)
