# -*- coding: utf-8 -*-
"""
T07 实时追踪 —— 完整程序（自己写）

把 T06 的三板斧封装成 detect_color，再串成一个能实时跑的程序：
读帧 -> 按当前颜色检测 -> 画质心十字/HUD -> 按键切换 -> q 退出。
"""
import os
import sys
import time

import cv2
import numpy as np


def _setup():
    """让控制台中文不乱码"""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


def _imread(p):
    """中文路径安全的读图"""
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


_setup()

HERE = os.path.dirname(os.path.abspath(__file__))

# 数字键 -> 颜色名（画面文字全用英文，putText 不支持中文）
COLOR_KEYS = {"red": "1", "blue": "2", "green": "3", "yellow": "4"}

# 没有摄像头/视频时，用这张四色图当一帧练手
FALLBACK_IMAGE = os.path.join(HERE, "images", "patterns.png")

# 每种颜色的 HSV 阈值（红色跨 0 点，要两段）
HSV_RANGES = {
    "red":    [((0, 100, 60), (10, 255, 255)),
               ((170, 100, 60), (179, 255, 255))],
    "blue":   [((100, 100, 60), (130, 255, 255))],
    "green":  [((35, 100, 60), (85, 255, 255))],
    "yellow": [((20, 100, 60), (35, 255, 255))],
}


# ---------------------------------------------------------------- 检测
def detect_color(frame, color_name):
    """在 BGR 帧里找出指定颜色目标的质心 (cx, cy)；没找到返回 None。

    color_name 取 "red" / "blue" / "green" / "yellow"。
    """
    if color_name not in HSV_RANGES:
        return None
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = None
    for lo, hi in HSV_RANGES[color_name]:
        m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        mask = m if mask is None else cv2.bitwise_or(mask, m)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # 去白点
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 补黑洞

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 100:     # 过滤小噪点
        return None
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    return (M["m10"] / M["m00"], M["m01"] / M["m00"])


# ---------------------------------------------------------------- 视频源
def open_source(src):
    """打开视频源并压低缓冲（延迟从几百 ms 降到几十 ms）。"""
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def _is_stream(src):
    """摄像头(数字)或网络流(http)算流，读完要重连；本地文件读完就退出。"""
    return isinstance(src, int) or (isinstance(src, str) and src.startswith("http"))


# ---------------------------------------------------------------- 主循环
def main(src=None):
    """实时追踪主循环。src 可以是摄像头编号 / http 流 / 视频文件 / 图片。"""
    if src is None:
        src = FALLBACK_IMAGE

    is_image = isinstance(src, str) and src.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    current = "red"

    cap = None
    static = None
    if is_image:
        static = _imread(src)
        if static is None:
            print("打不开图片:", src)
            return
    else:
        cap = open_source(src)
        if not cap.isOpened():
            print("打不开视频源:", src, "，退回用", FALLBACK_IMAGE)
            is_image = True
            static = _imread(FALLBACK_IMAGE)

    t0 = time.time()
    n = 0
    fps = 0.0

    while True:
        if is_image:
            frame = static.copy()
        else:
            ok, frame = cap.read()
            if not ok:
                if _is_stream(src):
                    cap.release()          # 断流重连
                    time.sleep(0.2)
                    cap = open_source(src)
                    if not cap.isOpened():
                        print("重连失败，退出")
                        break
                    continue
                else:
                    break                  # 本地视频读完

        # 检测当前颜色 + 画质心
        pt = detect_color(frame, current)
        if pt is not None:
            cx, cy = int(pt[0]), int(pt[1])
            cv2.drawMarker(frame, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
            cv2.putText(frame, "(%d, %d)" % (cx, cy), (cx + 12, cy - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 左上角 HUD：当前颜色 + FPS
        n += 1
        if n % 10 == 0:
            fps = n / (time.time() - t0)
        cv2.putText(frame, "Tracking: %s" % current, (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "FPS: %.1f" % fps, (10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("T07 Tracking", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key != 255:
            ch = chr(key)
            for name, k in COLOR_KEYS.items():
                if ch == k:
                    current = name
                    break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------- 自检
def self_test():
    """用 patterns.png 逐色定位 + 统计单帧 FPS。"""
    img = _imread(FALLBACK_IMAGE)
    if img is None:
        print("找不到", FALLBACK_IMAGE)
        return

    print("=== 四色定位（patterns.png）===")
    for name in ("red", "blue", "green", "yellow"):
        pt = detect_color(img, name)
        if pt is not None:
            print("  %-6s -> (%.1f, %.1f)" % (name, pt[0], pt[1]))
        else:
            print("  %-6s -> None" % name)

    print("=== 单帧 FPS（连续处理 60 帧）===")
    n = 60
    t0 = time.time()
    for _ in range(n):
        detect_color(img, "red")
    dt = time.time() - t0
    print("  处理 %d 帧耗时 %.2f s，平均 %.1f FPS" % (n, dt, n / dt))

    print("=== 基础：demo 视频逐帧统计 ===")
    demo = os.path.join(HERE, "demo_tracking.avi")
    if os.path.exists(demo):
        cap = cv2.VideoCapture(demo)
        frames = hits = 0
        t0 = time.time()
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames += 1
            if detect_color(f, "red") is not None:
                hits += 1
        dt = time.time() - t0
        cap.release()
        if frames:
            print("  帧数 %d，命中 %d，命中率 %.0f%%，%.1f FPS" %
                  (frames, hits, hits * 100.0 / frames, frames / dt))
    else:
        print("  没有 demo_tracking.avi，跳过")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        self_test()
    else:
        main()
