# -*- coding: utf-8 -*-
"""
T06 传统视觉三板斧 —— 作业（4 个函数）

三板斧：HSV 颜色阈值 → 形态学去噪 → 轮廓筛选 + 质心。
"""
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------- 工具（已给）
def _morph(mask, k=5):
    """开运算去噪 + 闭运算补洞"""
    kernel = np.ones((k, k), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _centroid(cnt) -> Optional[Tuple[float, float]]:
    """算轮廓质心，退化轮廓返回 None"""
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None
    return (M["m10"] / M["m00"], M["m01"] / M["m00"])


def _vertices(cnt) -> int:
    """多边形逼近后的顶点数（3=三角 4=四边 >=8≈圆）"""
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
    return len(approx)


def _in_range(img, lo, hi):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))


# ---------------------------------------------------------------- 1
def detect_red_circle(img) -> Optional[Tuple[float, float]]:
    """quiz_01：找出图中唯一的红色圆，返回质心 (cx, cy)；找不到返回 None。

    红色 H 跨 0 点，要 [0,10] 和 [170,180] 两段合并。
    """
    mask = _in_range(img, (0, 100, 50), (10, 255, 255)) | \
           _in_range(img, (170, 100, 50), (180, 255, 255))
    mask = _morph(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    return _centroid(c)


# ---------------------------------------------------------------- 2
def detect_blue_rect(img) -> Optional[Tuple[float, float]]:
    """quiz_02：找出蓝色正方形，返回质心。蓝色 H 约 100~130。"""
    mask = _in_range(img, (100, 100, 50), (130, 255, 255))
    mask = _morph(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    return _centroid(c)


# ---------------------------------------------------------------- 3
def detect_green_triangle(img) -> Optional[Tuple[float, float]]:
    """quiz_03：图中有绿三角 + 红圆干扰 + 蓝块干扰，只返回绿色三角质心。
    绿色 H 约 35~85；用顶点数 == 3 过滤掉其它形状。
    """
    mask = _in_range(img, (35, 100, 50), (85, 255, 255))
    mask = _morph(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) > 100 and _vertices(c) == 3:
            return _centroid(c)
    return None


# ---------------------------------------------------------------- 4
def detect_red_targets(img) -> List[Tuple[float, float]]:
    """quiz_04：找出图中所有红色目标的质心，按 x 升序返回。

    橙色圆 H≈19 不是红色，把红色 H 上限卡在 10 就不会误检。
    """
    mask = _in_range(img, (0, 100, 50), (10, 255, 255)) | \
           _in_range(img, (170, 100, 50), (180, 255, 255))
    mask = _morph(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = []
    for c in contours:
        if cv2.contourArea(c) > 100:
            p = _centroid(c)
            if p is not None:
                pts.append(p)
    pts.sort(key=lambda p: p[0])
    return pts


if __name__ == "__main__":
    import os
    import sys

    # 让控制台中文不乱码
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    here = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(here, "images")
    out_dir = os.path.join(here, "..", "截图")
    os.makedirs(out_dir, exist_ok=True)

    # 中文路径下 cv2.imread 会返回 None，用 imdecode 读
    def _read(p):
        return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)

    def _draw(img, pt, color=(0, 0, 255)):
        out = img.copy()
        if pt is not None:
            cx, cy = int(pt[0]), int(pt[1])
            cv2.drawMarker(out, (cx, cy), color, cv2.MARKER_CROSS, 20, 2)
            cv2.circle(out, (cx, cy), 5, color, -1)
        return out

    def _save(path, img):
        cv2.imencode(".png", img)[1].tofile(path)

    # 三个单目标题，真值对照
    cases = [
        ("quiz_01", detect_red_circle, (320, 240)),
        ("quiz_02", detect_blue_rect, (320, 240)),
        ("quiz_03", detect_green_triangle, (320, 293)),
    ]
    for name, fn, gt in cases:
        img = _read(os.path.join(img_dir, name + ".png"))
        pt = fn(img)
        if pt is not None:
            err = ((pt[0] - gt[0]) ** 2 + (pt[1] - gt[1]) ** 2) ** 0.5
        else:
            err = -1
        if pt is not None:
            print("%s 检测=(%.1f, %.1f) 真值=%s 误差=%.1f px" %
                  (name, pt[0], pt[1], gt, err))
        else:
            print("%s 检测=None 真值=%s" % (name, gt))
        _save(os.path.join(out_dir, "T06_结果图_" + name + ".png"), _draw(img, pt))

    # quiz_04 多目标
    img4 = _read(os.path.join(img_dir, "quiz_04.png"))
    pts = detect_red_targets(img4)
    out4 = img4.copy()
    for p in pts:
        cx, cy = int(p[0]), int(p[1])
        cv2.drawMarker(out4, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.circle(out4, (cx, cy), 5, (0, 0, 255), -1)
    gt4 = [(120, 130), (453, 373)]
    pts_fmt = [(round(x, 1), round(y, 1)) for x, y in pts]
    print("quiz_04 检测=%s 真值=%s" % (pts_fmt, gt4))
    _save(os.path.join(out_dir, "T06_结果图_quiz04.png"), out4)
