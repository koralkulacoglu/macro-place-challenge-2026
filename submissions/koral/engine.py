import math
import time
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from macro_place.benchmark import Benchmark

class StaticDesignData:
    """Pickleable container for design-specific constants extracted from PlacementCost."""
    def __init__(
        self,
        h_alloc: float,
        v_alloc: float,
        smooth_range: int,
        wl_norm_n_nets: int,
        net_weight: np.ndarray,
        net_owner: List[np.ndarray],
        net_offx: List[np.ndarray],
        net_offy: List[np.ndarray],
    ):
        self.h_alloc = h_alloc
        self.v_alloc = v_alloc
        self.smooth_range = smooth_range
        self.wl_norm_n_nets = wl_norm_n_nets
        self.net_weight = net_weight
        self.net_owner = net_owner
        self.net_offx = net_offx
        self.net_offy = net_offy

    @staticmethod
    def extract(benchmark: Benchmark, plc) -> 'StaticDesignData':
        """One-time extraction from the C++ oracle."""
        h_alloc, v_alloc = 0.0, 0.0
        smooth_range = 2
        n_nets = int(benchmark.num_nets)
        if plc is not None:
            try:
                h_alloc, v_alloc = plc.get_macro_routing_allocation()
            except Exception:
                h_alloc = getattr(plc, "hrouting_alloc", 0.0)
                v_alloc = getattr(plc, "vrouting_alloc", 0.0)
            try:
                smooth_range = int(plc.get_congestion_smooth_range())
            except Exception:
                smooth_range = int(getattr(plc, "smooth_range", 2))
        
        wl_norm_n_nets = int(getattr(plc, "net_cnt", n_nets)) if plc is not None else n_nets
        if wl_norm_n_nets <= 0:
            wl_norm_n_nets = max(n_nets, 1)

        net_weight = np.ones(n_nets, dtype=np.float64)
        if plc is not None:
            try:
                driver_names = list(plc.nets.keys())
                for n in range(min(n_nets, len(driver_names))):
                    pi = plc.mod_name_to_indices[driver_names[n]]
                    net_weight[n] = float(plc.modules_w_pins[pi].get_weight())
            except Exception: pass

        n_hard = benchmark.num_hard_macros
        pin_offsets = benchmark.macro_pin_offsets
        npn = benchmark.net_pin_nodes
        net_owner, net_offx, net_offy = [], [], []
        
        if not npn:
            for n in range(n_nets):
                nodes = benchmark.net_nodes[n].cpu().numpy().astype(np.int64) if benchmark.net_nodes else np.zeros(0, dtype=np.int64)
                net_owner.append(nodes); net_offx.append(np.zeros(nodes.shape[0])); net_offy.append(np.zeros(nodes.shape[0]))
        else:
            for n in range(n_nets):
                pn = npn[n].cpu().numpy().astype(np.int64)
                if pn.size == 0:
                    net_owner.append(np.zeros(0, dtype=np.int64)); net_offx.append(np.zeros(0)); net_offy.append(np.zeros(0))
                    continue
                owners, slots = pn[:, 0], pn[:, 1]
                offx, offy = np.zeros(owners.shape[0]), np.zeros(owners.shape[0])
                for k in range(owners.shape[0]):
                    o, s = int(owners[k]), int(slots[k])
                    if o < n_hard and pin_offsets and o < len(pin_offsets):
                        po = pin_offsets[o]
                        if po is not None and po.shape[0] > s: offx[k], offy[k] = float(po[s, 0]), float(po[s, 1])
                net_owner.append(owners); net_offx.append(offx); net_offy.append(offy)

        return StaticDesignData(
            h_alloc=h_alloc, v_alloc=v_alloc, smooth_range=smooth_range,
            wl_norm_n_nets=wl_norm_n_nets, net_weight=net_weight,
            net_owner=net_owner, net_offx=net_offx, net_offy=net_offy
        )


class FastEvaluator:
    def __init__(self, benchmark: Benchmark, data: StaticDesignData):
        self.cw, self.ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
        self.grid_col, self.grid_row = int(benchmark.grid_cols), int(benchmark.grid_rows)
        self.gw, self.gh = self.cw / self.grid_col, self.ch / self.grid_row
        self.grid_area = self.gw * self.gh
        self.h_per_um, self.v_per_um = float(benchmark.hroutes_per_micron), float(benchmark.vroutes_per_micron)
        self.grid_v_routes, self.grid_h_routes = self.gw * self.v_per_um, self.gh * self.h_per_um
        self.h_alloc, self.v_alloc, self.smooth_range = data.h_alloc, data.v_alloc, data.smooth_range
        self.n_hard, self.n_macros = benchmark.num_hard_macros, benchmark.num_macros
        self.n_soft, self.n_nets = self.n_macros - self.n_hard, int(benchmark.num_nets)
        self.n_ports, self.wl_norm_n_nets = int(benchmark.port_positions.shape[0]), data.wl_norm_n_nets
        self.positions = benchmark.macro_positions.detach().cpu().numpy().astype(np.float64)
        self.sizes, self.port_pos = benchmark.macro_sizes.detach().cpu().numpy().astype(np.float64), benchmark.port_positions.detach().cpu().numpy().astype(np.float64) if benchmark.port_positions.shape[0] else np.zeros((0, 2))
        self.half, self.movable = self.sizes / 2.0, benchmark.get_movable_mask().detach().cpu().numpy().astype(bool)
        self.net_owner, self.net_offx, self.net_offy = data.net_owner, data.net_offx, data.net_offy
        self._net_xmin, self._net_ymin, self._net_xmax, self._net_ymax = np.zeros(self.n_nets), np.zeros(self.n_nets), np.zeros(self.n_nets), np.zeros(self.n_nets)
        self._net_weight = data.net_weight
        self._owner_to_nets = {}
        for n in range(self.n_nets):
            for o in self.net_owner[n]: self._owner_to_nets.setdefault(int(o), []).append(n)
        self.density_grid, self.h_pin_cong, self.v_pin_cong, self.h_macro_cong, self.v_macro_cong = np.zeros((self.grid_row, self.grid_col)), np.zeros((self.grid_row, self.grid_col)), np.zeros((self.grid_row, self.grid_col)), np.zeros((self.grid_row, self.grid_col)), np.zeros((self.grid_row, self.grid_col))
        self._init_caches()

    def _pin_x(self, owners, offx):
        out = np.empty(owners.shape[0], dtype=np.float64); m = owners < self.n_macros
        out[m] = self.positions[owners[m], 0] + offx[m]
        if (~m).any(): out[~m] = self.port_pos[owners[~m] - self.n_macros, 0] + offx[~m]
        return out
    def _pin_y(self, owners, offy):
        out = np.empty(owners.shape[0], dtype=np.float64); m = owners < self.n_macros
        out[m] = self.positions[owners[m], 1] + offy[m]
        if (~m).any(): out[~m] = self.port_pos[owners[~m] - self.n_macros, 1] + offy[~m]
        return out
    def _net_bbox(self, n):
        owners = self.net_owner[n]
        if owners.size == 0: return 0.0, 0.0, 0.0, 0.0
        xs, ys = self._pin_x(owners, self.net_offx[n]), self._pin_y(owners, self.net_offy[n])
        return xs.min(), ys.min(), xs.max(), ys.max()
    def _grid_cell(self, x, y):
        return max(0, min(self.grid_row-1, int(math.floor(y/self.gh)))), max(0, min(self.grid_col-1, int(math.floor(x/self.gw))))
    def _add_macro_density(self, i, sign=+1):
        x, y = self.positions[i]; w, h = self.sizes[i]; x_min, x_max, y_min, y_max = x-w/2, x+w/2, y-h/2, y+h/2
        br, bc = self._grid_cell(x_min, y_min); ur, uc = self._grid_cell(x_max, y_max)
        for r in range(br, ur+1):
            dy = min(y_max, (r+1)*self.gh) - max(y_min, r*self.gh)
            if dy>0:
                for c in range(bc, uc+1):
                    dx = min(x_max, (c+1)*self.gw) - max(x_min, c*self.gw)
                    if dx>0: self.density_grid[r,c] += sign*dx*dy
    def _add_macro_route(self, i, sign=+1):
        x, y = self.positions[i]; w, h = self.sizes[i]; x_min, x_max, y_min, y_max = x-w/2, x+w/2, y-h/2, y+h/2
        br, bc = self._grid_cell(x_min, y_min); ur, uc = self._grid_cell(x_max, y_max)
        eps = 1e-5; pv, ph = False, False
        for r in range(br, ur+1):
            dy = min(y_max, (r+1)*self.gh) - max(y_min, r*self.gh)
            if dy<=0: continue
            for c in range(bc, uc+1):
                dx = min(x_max, (c+1)*self.gw) - max(x_min, c*self.gw)
                if dx<=0: continue
                self.v_macro_cong[r,c] += sign*dx*self.v_alloc; self.h_macro_cong[r,c] += sign*dy*self.h_alloc
                if ur!=br and (r==br or r==ur) and abs(dy-self.gh)>eps: pv=True
                if uc!=bc and (c==bc or c==uc) and abs(dx-self.gw)>eps: ph=True
        if pv:
            for c in range(bc, uc+1):
                dx = min(x_max, (c+1)*self.gw) - max(x_min, c*self.gw)
                if dx>0: self.v_macro_cong[ur,c] -= sign*dx*self.v_alloc
        if ph:
            for r in range(br, ur+1):
                dy = min(y_max, (r+1)*self.gh) - max(y_min, r*self.gh)
                if dy>0: self.h_macro_cong[r,uc] -= sign*dy*self.h_alloc
    def _route_pin_cong(self, n, sign=+1):
        owners = self.net_owner[n]
        if owners.size==0: return
        xs, ys = self._pin_x(owners, self.net_offx[n]), self._pin_y(owners, self.net_offy[n])
        cells, c_set = [], set()
        for i in range(owners.shape[0]):
            rc = self._grid_cell(xs[i], ys[i]); cells.append(rc); c_set.add(rc)
        if len(c_set)<=1: return
        src, w = cells[0], self._net_weight[n]
        if len(c_set)==2: self._two_pin(src, list(c_set), w, sign)
        elif len(c_set)==3: self._three_pin(list(c_set), w, sign)
        else:
            for node in c_set:
                if node!=src: self._two_pin(src, [src, node], w, sign)
    def _two_pin(self, src, two, w, sign):
        sink = two[1] if two[0]==src else two[0]
        r0, r1, c0, c1 = min(src[0], sink[0]), max(src[0], sink[0]), min(src[1], sink[1]), max(src[1], sink[1])
        if c1>c0: self.h_pin_cong[src[0], c0:c1] += sign*w
        if r1>r0: self.v_pin_cong[r0:r1, sink[1]] += sign*w
    def _three_pin(self, cells, w, sign):
        cs = sorted(cells, key=lambda x: (x[1], x[0])); (y1, x1), (y2, x2), (y3, x3) = cs
        if x1<x2<x3 and min(y1,y3)<y2<max(y1,y3):
            if x2>x1: self.h_pin_cong[y1, x1:x2] += sign*w
            if x3>x2: self.h_pin_cong[y2, x2:x3] += sign*w
            if max(y1,y2)>min(y1,y2): self.v_pin_cong[min(y1,y2):max(y1,y2), x2] += sign*w
            if max(y2,y3)>min(y2,y3): self.v_pin_cong[min(y2,y3):max(y2,y3), x3] += sign*w
        elif x2==x3 and x1<x2 and y1<min(y2,y3):
            if x2>x1: self.h_pin_cong[y1, x1:x2] += sign*w
            if max(y2,y3)>y1: self.v_pin_cong[y1:max(y2,y3), x2] += sign*w
        elif y2==y3:
            if x2>x1: self.h_pin_cong[y1, x1:x2] += sign*w
            if x3>x2: self.h_pin_cong[y2, x2:x3] += sign*w
            if max(y1,y2)>min(y1,y2): self.v_pin_cong[min(y1,y2):max(y1,y2), x2] += sign*w
        else:
            cs2 = sorted(cs); (y1, x1), (y2, x2), (y3, x3) = cs2; xmin, xmax = min(x1,x2,x3), max(x1,x2,x3)
            if xmax>xmin: self.h_pin_cong[y2, xmin:xmax] += sign*w
            if max(y1,y2)>min(y1,y2): self.v_pin_cong[min(y1,y2):max(y1,y2), x1] += sign*w
            if max(y2,y3)>min(y2,y3): self.v_pin_cong[min(y2,y3):max(y2,y3), x3] += sign*w
    def _init_caches(self):
        self.density_grid[:], self.h_pin_cong[:], self.v_pin_cong[:], self.h_macro_cong[:], self.v_macro_cong[:] = 0, 0, 0, 0, 0
        for m in range(self.n_macros): self._add_macro_density(m, +1)
        for m in range(self.n_hard): self._add_macro_route(m, +1)
        for n in range(self.n_nets):
            x0, y0, x1, y1 = self._net_bbox(n); self._net_xmin[n], self._net_ymin[n], self._net_xmax[n], self._net_ymax[n] = x0, y0, x1, y1
            self._route_pin_cong(n, +1)
    def _density_cost(self):
        gc = (self.density_grid / self.grid_area).ravel(); nz = gc[gc>0]
        if nz.size==0: return 0.0
        cnt = math.floor(gc.size * 0.1)
        if cnt==0: return 0.5*float(nz.max())
        sd = np.sort(nz)[::-1]; return 0.5 * float(sd[:min(cnt, sd.size)].sum() / cnt)
    def _smooth(self, grid, axis):
        sr, (R, C) = self.smooth_range, grid.shape
        if axis == 0:
            cnt = (np.minimum(C - 1, np.arange(C) + sr) - np.maximum(0, np.arange(C) - sr) + 1).astype(np.float64)
            pad = np.pad(grid / cnt[np.newaxis, :], ((0, 0), (sr, sr)), mode="constant")
            cs = np.cumsum(pad, axis=1); return cs[:, 2*sr:] - np.concatenate([np.zeros((R, 1)), cs[:, :C-1+2*sr]], axis=1)[:, :C]
        else:
            cnt = (np.minimum(R - 1, np.arange(R) + sr) - np.maximum(0, np.arange(R) - sr) + 1).astype(np.float64)
            pad = np.pad(grid / cnt[:, np.newaxis], ((sr, sr), (0, 0)), mode="constant")
            cs = np.cumsum(pad, axis=0); return cs[2*sr:, :] - np.concatenate([np.zeros((1, C)), cs[:R-1+2*sr, :]], axis=0)[:R, :]
    def _congestion_cost(self):
        v_s, h_s = self._smooth(self.v_pin_cong / self.grid_v_routes, axis=0), self._smooth(self.h_pin_cong / self.grid_h_routes, axis=1)
        combined = np.concatenate([(v_s + self.v_macro_cong / self.grid_v_routes).ravel(), (h_s + self.h_macro_cong / self.grid_h_routes).ravel()])
        xs = np.sort(combined)[::-1]; cnt = math.floor(xs.size * 0.05)
        return float(xs[:cnt].mean()) if cnt>0 else (float(xs.max()) if xs.size else 0.0)
    def proxy_cost(self):
        wl = (float(np.sum(((self._net_xmax - self._net_xmin) + (self._net_ymax - self._net_ymin)) * self._net_weight)) / ((self.cw + self.ch) * self.wl_norm_n_nets))
        d, c = self._density_cost(), self._congestion_cost()
        return {"proxy_cost": wl + 0.5*d + 0.5*c, "wirelength_cost": wl, "density_cost": d, "congestion_cost": c}
    def move_macro(self, i, nx, ny, is_hard=True):
        if is_hard: self._add_macro_route(i, -1)
        self._add_macro_density(i, -1)
        nets = self._owner_to_nets.get(i, [])
        for n in nets: self._route_pin_cong(n, -1)
        self.positions[i] = [nx, ny]
        self._add_macro_density(i, +1)
        if is_hard: self._add_macro_route(i, +1)
        for n in nets:
            x0, y0, x1, y1 = self._net_bbox(n); self._net_xmin[n], self._net_ymin[n], self._net_xmax[n], self._net_ymax[n] = x0, y0, x1, y1
            self._route_pin_cong(n, +1)
    def swap_macros(self, i, j):
        xi, yi, xj, yj = self.positions[i][0], self.positions[i][1], self.positions[j][0], self.positions[j][1]
        self.move_macro(i, xj, yj, is_hard=(i < self.n_hard))
        self.move_macro(j, xi, yi, is_hard=(j < self.n_hard))
    def restore(self, pos):
        if not np.array_equal(pos, self.positions): self.positions[:], self._init_caches()

def _soft_centroid_target(ev: FastEvaluator, soft_idx: int):
    nets = ev._owner_to_nets.get(soft_idx, [])
    if not nets: return None
    tx, ty, tw = 0.0, 0.0, 0.0
    for n in nets:
        owners = ev.net_owner[n]
        if owners.size<2: continue
        xs, ys = ev._pin_x(owners, ev.net_offx[n]), ev._pin_y(owners, ev.net_offy[n])
        i = int(np.where(owners == soft_idx)[0][0]); k = owners.size
        w = ev._net_weight[n] / (k-1); tx += w * (xs.sum()-xs[i])/(k-1); ty += w * (ys.sum()-ys[i])/(k-1); tw += w
    return (float(tx/tw), float(ty/tw)) if tw>0 else None

def _slide_legal(ev, i, nx, ny):
    h = ev.half[i]
    if nx-h[0]<0 or nx+h[0]>ev.cw or ny-h[1]<0 or ny+h[1]>ev.ch: return False
    for k in range(ev.n_hard):
        if k==i: continue
        ox, oy = (ev.sizes[i,0]+ev.sizes[k,0])/2 - abs(nx-ev.positions[k,0]), (ev.sizes[i,1]+ev.sizes[k,1])/2 - abs(ny-ev.positions[k,1])
        if ox>0 and oy>0: return False
    return True

def _swap_legal(ev, i, j):
    pi, pj, hi, hj = ev.positions[i], ev.positions[j], ev.half[i], ev.half[j]
    if pj[0]-hi[0]<0 or pj[0]+hi[0]>ev.cw or pj[1]-hi[1]<0 or pj[1]+hi[1]>ev.ch or pi[0]-hj[0]<0 or pi[0]+hj[0]>ev.cw or pi[1]-hj[1]<0 or pi[1]+hj[1]>ev.ch: return False
    for k in range(ev.n_hard):
        if k==i or k==j: continue
        pk, sk = ev.positions[k], ev.sizes[k]
        if (ev.sizes[i,0]+sk[0])/2-abs(pj[0]-pk[0])>0 and (ev.sizes[i,1]+sk[1])/2-abs(pj[1]-pk[1])>0: return False
        if (ev.sizes[j,0]+sk[0])/2-abs(pi[0]-pk[0])>0 and (ev.sizes[j,1]+sk[1])/2-abs(pi[1]-pk[1])>0: return False
    return True

def lahc_polish(ev, list_len=100, time_budget_s=600.0, seed=0):
    rng, t0, it = np.random.default_rng(seed), time.time(), 0
    cur_cost = ev.proxy_cost()["proxy_cost"]; best_cost, best_pos, history = cur_cost, ev.positions.copy(), [cur_cost]*list_len
    while time.time()-t0 < time_budget_s:
        r = rng.random()
        if r < 0.3: # swap
            i = int(rng.integers(0, ev.n_hard))
            if not ev.movable[i]: it+=1; continue
            d = np.linalg.norm(ev.positions[:ev.n_hard]-ev.positions[i], axis=1); d[i]=np.inf
            j = int(np.argsort(d)[:12][int(rng.integers(0, 12))])
            if not ev.movable[j] or not _swap_legal(ev, i, j): it+=1; continue
            ev.swap_macros(i, j); cand = ev.proxy_cost()["proxy_cost"]
            if cand<cur_cost or cand<history[it%list_len]:
                cur_cost, history[it%list_len] = cand, cand
                if cand<best_cost: best_cost, best_pos = cand, ev.positions.copy()
            else: ev.swap_macros(i, j)
        elif r < 0.7 and ev.n_soft>0: # soft move
            i = ev.n_hard + int(rng.integers(0, ev.n_soft))
            if not ev.movable[i]: it+=1; continue
            ox, oy = ev.positions[i]; tgt = _soft_centroid_target(ev, i) if rng.random()<0.5 else None
            if tgt: f=float(rng.uniform(0.05, 0.5)); nx, ny = ox+f*(tgt[0]-ox), oy+f*(tgt[1]-oy)
            else: rx, ry = 0.03*ev.cw, 0.03*ev.ch; nx, ny = ox+float(rng.uniform(-rx,rx)), oy+float(rng.uniform(-ry,ry))
            nx, ny = max(ev.half[i,0], min(ev.cw-ev.half[i,0], nx)), max(ev.half[i,1], min(ev.ch-ev.half[i,1], ny))
            ev.move_macro(i, nx, ny, is_hard=False); cand = ev.proxy_cost()["proxy_cost"]
            if cand<cur_cost or cand<history[it%list_len]:
                cur_cost, history[it%list_len] = cand, cand
                if cand<best_cost: best_cost, best_pos = cand, ev.positions.copy()
            else: ev.move_macro(i, ox, oy, is_hard=False)
        else: # hard slide
            i = int(rng.integers(0, ev.n_hard))
            if not ev.movable[i]: it+=1; continue
            ox, oy, rx, ry = ev.positions[i][0], ev.positions[i][1], 0.06*ev.cw, 0.06*ev.ch
            nx, ny = max(ev.half[i,0], min(ev.cw-ev.half[i,0], ox+float(rng.uniform(-rx,rx)))), max(ev.half[i,1], min(ev.ch-ev.half[i,1], oy+float(rng.uniform(-ry,ry))))
            if not _slide_legal(ev, i, nx, ny): it+=1; continue
            ev.move_macro(i, nx, ny, is_hard=True); cand = ev.proxy_cost()["proxy_cost"]
            if cand<cur_cost or cand<history[it%list_len]:
                cur_cost, history[it%list_len] = cand, cand
                if cand<best_cost: best_cost, best_pos = cand, ev.positions.copy()
            else: ev.move_macro(i, ox, oy, is_hard=True)
        it += 1
    ev.restore(best_pos)
    return {"proxy_cost": best_cost, "iters": it}
