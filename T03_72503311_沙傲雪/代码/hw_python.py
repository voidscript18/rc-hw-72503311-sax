# -*- coding: utf-8 -*-
"""
T03 Python 速通 —— 作业（6 个函数）
"""
from typing import Dict, List, Tuple


# ---------------------------------------------------------------- 1
def count_colors(colors: List[str]) -> Dict[str, int]:
    """统计列表中每种颜色出现的次数。

    输入: ["red", "blue", "red", "green", "red"]
    输出: {"red": 3, "blue": 1, "green": 1}
    """
    counts = {}
    for c in colors:
        counts[c] = counts.get(c, 0) + 1
    return counts


# ---------------------------------------------------------------- 2
def largest_box(boxes: List[Tuple[float, float]]) -> int:
    """给定 [(宽, 高), ...]，返回面积最大的那个的下标。空列表返回 -1。

    输入: [(10, 20), (30, 5), (12, 12)]
    输出: 0            # 10*20=200 最大
    """
    if not boxes:
        return -1
    best = 0
    best_area = boxes[0][0] * boxes[0][1]
    for i, (w, h) in enumerate(boxes):
        area = w * h
        if area > best_area:
            best_area = area
            best = i
    return best


# ---------------------------------------------------------------- 3
def filter_by_conf(dets: List[Tuple[str, float]], thr: float = 0.5) -> List[Tuple[str, float]]:
    """过滤检测结果，只保留置信度 >= thr 的，保持原顺序。

    输入: [("person", 0.92), ("bus", 0.31), ("car", 0.55)], thr=0.5
    输出: [("person", 0.92), ("car", 0.55)]
    """
    return [d for d in dets if d[1] >= thr]


# ---------------------------------------------------------------- 4
def clamp(value: float, lo: float, hi: float) -> float:
    """把 value 限制在 [lo, hi] 区间内（控制量限幅，闭环里天天用）。

    clamp(1.5, -1, 1) -> 1.0
    clamp(-3,  -1, 1) -> -1.0
    clamp(0.2, -1, 1) -> 0.2
    """
    return max(lo, min(value, hi))


# ---------------------------------------------------------------- 5
def moving_average(values: List[float], k: int = 3) -> List[float]:
    """滑动平均（窗口 k），输出长度与输入相同；前 k-1 项用"到目前为止的均值"。

    输入: [1, 2, 3, 4, 5], k=3
    输出: [1.0, 1.5, 2.0, 3.0, 4.0]
    k <= 1 时原样返回。
    """
    if k <= 1:
        return values
    out = []
    for i in range(len(values)):
        start = max(0, i - k + 1)
        window = values[start:i + 1]
        out.append(sum(window) / len(window))
    return out


# ---------------------------------------------------------------- 6
def parse_detection_line(line: str) -> Tuple[str, float, Tuple[int, int, int, int]]:
    """解析一行检测结果文本，返回 (类别名, 置信度, (x1,y1,x2,y2))。

    输入: "person 0.92 100 120 300 400"
    输出: ("person", 0.92, (100, 120, 300, 400))
    格式不对时返回 ("", 0.0, (0, 0, 0, 0))
    """
    parts = line.split()
    if len(parts) != 6:
        return ("", 0.0, (0, 0, 0, 0))
    try:
        name = parts[0]
        conf = float(parts[1])
        x1, y1, x2, y2 = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
        return (name, conf, (x1, y1, x2, y2))
    except ValueError:
        return ("", 0.0, (0, 0, 0, 0))


if __name__ == "__main__":
    # 用题目给的例子自测
    print(count_colors(["red", "blue", "red", "green", "red"]))
    print(largest_box([(10, 20), (30, 5), (12, 12)]))
    print(filter_by_conf([("person", 0.92), ("bus", 0.31), ("car", 0.55)]))
    print(clamp(1.5, -1, 1), clamp(-3, -1, 1), clamp(0.2, -1, 1))
    print(moving_average([1, 2, 3, 4, 5]))
    print(parse_detection_line("person 0.92 100 120 300 400"))
    print(parse_detection_line("这是错误格式"))
