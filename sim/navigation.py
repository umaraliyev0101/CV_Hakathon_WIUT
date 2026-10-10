"""Visibility-graph navigation around inflated obstacle footprints + speed profiles (pure)."""
import heapq
import math

import numpy as np

from sim import layout

EPS = 0.05


def _inflate(poly, r):
    """Offset a convex CCW polygon outward by r (square corners: contains the Minkowski sum)."""
    P = np.asarray(poly, float)
    n = len(P)
    out = []
    for i in range(n):
        a, b, c = P[i - 1], P[i], P[(i + 1) % n]
        e1, e2 = b - a, c - b
        n1 = np.array([e1[1], -e1[0]]) / np.linalg.norm(e1)
        n2 = np.array([e2[1], -e2[0]]) / np.linalg.norm(e2)
        # intersection of the two offset edges
        bis = n1 + n2
        bis /= np.linalg.norm(bis)
        out.append(b + bis * r / float(np.dot(bis, n1)))
    return out


def _seg_hits_convex(a, b, poly):
    """True when segment a-b passes through the open interior of convex CCW polygon."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    t0, t1 = 0.0, 1.0
    P = np.asarray(poly, float)
    for i in range(len(P)):
        p, q = P[i], P[(i + 1) % len(P)]
        e = q - p
        nrm = np.array([e[1], -e[0]])  # outward normal for CCW
        num = float(np.dot(nrm, a - p))   # >0 outside this edge
        den = float(np.dot(nrm, d))
        if abs(den) < 1e-12:
            if num >= -1e-9:
                return False
            continue
        t = -num / den
        if den < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 >= t1 - 1e-9:
            return False
    return True


class NavGraph:
    def __init__(self, lay, nav_cfg):
        self.r = nav_cfg["body_radius_m"] + nav_cfg["clearance_m"]
        self.body_r = nav_cfg["body_radius_m"]
        self.W = lay["room"]["width_m"]
        self.D = lay["room"]["depth_m"]
        self.blocked = [_inflate(p, self.r) for p in lay["obstacles_m"]]
        self.nodes = []
        for p in lay["obstacles_m"]:
            for v in _inflate(p, self.r + EPS):
                if self.is_free(v):
                    self.nodes.append(np.asarray(v, float))

    def is_free(self, pt):
        x, y = float(pt[0]), float(pt[1])
        m = self.body_r
        if not (m <= x <= self.W - m and -1.0 <= y <= self.D - m):  # y < 0: outside the entrance
            return False
        return not any(layout.point_in_polygon(pt, b) for b in self.blocked)

    def visible(self, a, b):
        return not any(_seg_hits_convex(a, b, poly) for poly in self.blocked)

    def shortest_path(self, start, goal):
        start, goal = np.asarray(start, float), np.asarray(goal, float)
        if not self.is_free(start) or not self.is_free(goal):
            raise ValueError(f"start {start} or goal {goal} not free")
        pts = [start, goal] + self.nodes
        n = len(pts)
        dist = [math.inf] * n
        prev = [-1] * n
        dist[0] = 0.0
        pq = [(0.0, 0)]
        while pq:
            d, i = heapq.heappop(pq)
            if d > dist[i]:
                continue
            if i == 1:
                break
            for j in range(n):
                if j == i:
                    continue
                if not self.visible(pts[i], pts[j]):
                    continue
                nd = d + float(np.linalg.norm(pts[j] - pts[i]))
                if nd < dist[j] - 1e-12:
                    dist[j], prev[j] = nd, i
                    heapq.heappush(pq, (nd, j))
        if math.isinf(dist[1]):
            return None
        path, k = [], 1
        while k != -1:
            path.append(pts[k].tolist())
            k = prev[k]
        return path[::-1]


def path_length(path):
    return sum(math.dist(a, b) for a, b in zip(path[:-1], path[1:]))


def point_at(path, s):
    """Point at arc length s along polyline, plus the segment heading (deg)."""
    acc = 0.0
    for a, b in zip(path[:-1], path[1:]):
        L = math.dist(a, b)
        if s <= acc + L or b is path[-1]:
            u = 0.0 if L == 0 else min(1.0, max(0.0, (s - acc) / L))
            heading = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
            return [a[0] + u * (b[0] - a[0]), a[1] + u * (b[1] - a[1])], heading
        acc += L
    a, b = path[-2], path[-1]
    return list(b), math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def trapezoid_profile(length, v_max, accel, fps, decel=None):
    """Arc length per frame for accelerate-cruise-decelerate; returns list ending exactly at length."""
    decel = accel if decel is None else decel
    if length <= 1e-9:
        return [0.0]
    d_acc = v_max ** 2 / (2 * accel)
    d_dec = v_max ** 2 / (2 * decel)
    if d_acc + d_dec > length:  # triangle profile
        v_peak = (2 * length * accel * decel / (accel + decel)) ** 0.5
        d_acc, d_dec = v_peak ** 2 / (2 * accel), v_peak ** 2 / (2 * decel)
    else:
        v_peak = v_max
    t_acc, t_dec = v_peak / accel, v_peak / decel
    t_cruise = (length - d_acc - d_dec) / v_peak
    T = t_acc + t_cruise + t_dec
    n = int(math.ceil(T * fps))
    out = []
    for k in range(n + 1):
        t = min(k / fps, T)
        if t < t_acc:
            s = 0.5 * accel * t * t
        elif t < t_acc + t_cruise:
            s = d_acc + v_peak * (t - t_acc)
        else:
            td = T - t
            s = length - 0.5 * decel * td * td
        out.append(min(length, s))
    out[-1] = length
    return out
