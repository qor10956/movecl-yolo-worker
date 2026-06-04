#!/usr/bin/env python3
"""
RTMPose Climbing Video Analyzer — Apache 2.0
- rtmlib Body (RTMDet-nano person detector + RTMPose-m 256×192, COCO 17-kp)
- person detector 내장 → 홀드/벽 오감지 방지
- One-Euro Filter 시계열 스무딩
- GPU T4 지원

Usage: python3 mediapipe_analyze.py <input> <output> <progress_json> [pose_json]
"""
import sys
import os
import json
import subprocess
import time
import math
import numpy as np
import cv2

from rtmlib import Wholebody
import onnxruntime as ort
from ultralytics import YOLO

# ── 메타데이터 기반 회전 감지 ────────────────────────────────────────────────
def get_metadata_rotation(input_path):
    """ffprobe로 영상 메타데이터 회전 읽기 — iPhone/Android 영상에 신뢰도 높음."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", input_path],
            capture_output=True, text=True, timeout=10,
        )
        data = json.loads(r.stdout)
        for s in data.get("streams", []):
            if s.get("codec_type") != "video": continue
            for sd in s.get("side_data_list", []):
                v = sd.get("rotation")
                if v is not None:
                    return (-int(v)) % 360
            tags = s.get("tags", {})
            v = tags.get("rotate")
            if v:
                return int(v) % 360
    except Exception: pass
    return None

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)

# ── 상수 ────────────────────────────────────────────────────────────────────
N_KPS             = 133
INFER_EVERY       = 2   # GPU 환경: 2 (더 자주 추론 → 더 부드러운 추적)
MAX_PROC_WIDTH    = 1280  # 960→1280: 원거리 소형 피사체 감지 개선 (A10G GPU에서 큰 부담 없음)
ONEEURO_MINCUTOFF = 0.30  # 낮을수록 정지/느린 동작에서 더 부드럽게 (기존 0.7)
ONEEURO_BETA      = 0.04  # 낮을수록 빠른 동작에도 스무딩 강하게 (기존 0.08)
ONEEURO_DCUTOFF   = 1.0
SAMPLE_EVERY_SEC  = 0.5
VIS_THRESH        = 0.35
MAX_GHOST_FRAMES  = 4

# ── Wholebody 133 키포인트 ───────────────────────────────────────────────────
# 0-16: COCO body / 17-19: L-foot / 20-22: R-foot
# 23-90: face(스킵) / 91-111: L-hand / 112-132: R-hand
TORSO_IDS   = frozenset({5, 6, 11, 12})
PARENT_KPS  = {7:5, 9:7, 8:6, 10:8, 13:11, 15:13, 14:12, 16:14}

# body 연결 (COCO 17)
_BODY_PAIRS = [
    (5,7),(7,9),(6,8),(8,10),(5,6),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]
# foot: ankle → big_toe, small_toe, heel
_FOOT_PAIRS = [(15,17),(15,18),(17,19),(18,19),(16,20),(16,21),(20,22),(21,22)]
# COCO-WholeBody hand: wrist + 5fingers × 4joints (offsets: thumb=1,index=5,mid=9,ring=13,pinky=17)
def _hand_pairs(w):
    pairs = []
    for off in [1, 5, 9, 13, 17]:
        b = w + off
        pairs.append((w, b))          # wrist → finger_base
        for j in range(3): pairs.append((b+j, b+j+1))   # chain within finger
    return pairs
_LHAND_PAIRS = [(9, 91)]  + _hand_pairs(91)   # max index: 91+17+3=111 ✓
_RHAND_PAIRS = [(10, 112)] + _hand_pairs(112)  # max index: 112+17+3=132 ✓

SKEL_PAIRS  = _BODY_PAIRS + _FOOT_PAIRS   # hand pairs 제거 — 손가락 연결선 숨김

SEG_COLORS  = {"arm":(0,200,255),"torso":(0,255,80),"leg":(255,80,200),"head":(200,200,50),
               "foot":(100,255,255),"hand":(255,180,50)}
ARM_P   = frozenset({(5,7),(7,9),(6,8),(8,10)})
TORSO_P = frozenset({(5,6),(5,11),(6,12),(11,12)})
LEG_P   = frozenset({(11,13),(13,15),(12,14),(14,16)})
HEAD_P  = frozenset({(0,5),(0,6)})
FOOT_P  = frozenset(map(tuple, _FOOT_PAIRS))
LHAND_P = frozenset(map(tuple, _LHAND_PAIRS))
RHAND_P = frozenset(map(tuple, _RHAND_PAIRS))
HL_KPS  = {9:"L-Hand",10:"R-Hand",15:"L-Foot",16:"R-Foot"}
HEAD_KPS = frozenset(range(0, 5)) | frozenset(range(23, 91))   # 코/눈/귀(0-4) + face(23-90) 스킵
HAND_KPS = frozenset(range(91, 133))  # hand 키포인트 그리기 스킵 (손목 9/10으로 대체)


# ── 유틸 ────────────────────────────────────────────────────────────────────
def write_progress(path, pct, msg, extra=None):
    try:
        d = {"pct": pct, "msg": msg, "done": pct >= 100}
        if extra: d.update(extra)
        with open(path, "w") as f: json.dump(d, f)
    except Exception: pass

def enhance_frame(frame):
    try:
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        m = g.mean()
        gv = 0.7 if m < 80 else 1.2 if m > 180 else 1.0
        if gv != 1.0:
            lut = np.array([((i/255.0)**(1/gv))*255 for i in range(256)], dtype=np.uint8)
            frame = cv2.LUT(frame, lut)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8)).apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    except Exception: return frame

def rot(f, a):
    if a == 0:   return f
    if a == 90:  return cv2.rotate(f, cv2.ROTATE_90_CLOCKWISE)
    if a == 180: return cv2.rotate(f, cv2.ROTATE_180)
    if a == 270: return cv2.rotate(f, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return f

def rot_back(f, a):
    if a == 0:   return f
    if a == 90:  return cv2.rotate(f, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if a == 180: return cv2.rotate(f, cv2.ROTATE_180)
    if a == 270: return cv2.rotate(f, cv2.ROTATE_90_CLOCKWISE)
    return f

def seg_color(s, e):
    p = (s, e)
    if p in HEAD_P:   return SEG_COLORS["head"]
    if p in ARM_P:    return SEG_COLORS["arm"]
    if p in TORSO_P:  return SEG_COLORS["torso"]
    if p in LEG_P:    return SEG_COLORS["leg"]
    if p in FOOT_P:   return SEG_COLORS["foot"]
    if p in LHAND_P:  return SEG_COLORS["hand"]
    if p in RHAND_P:  return SEG_COLORS["hand"]
    return (200, 200, 200)


# ── OneEuroFilter ────────────────────────────────────────────────────────────
class OEF:
    def __init__(self, mc=ONEEURO_MINCUTOFF, b=ONEEURO_BETA, dc=ONEEURO_DCUTOFF):
        self.mc=mc; self.b=b; self.dc=dc; self.x=None; self.dx=0.0; self.lt=None
    def sf(self, dt, c):
        return 1.0 / (1.0 + 1.0/(2.0*math.pi*c*dt))
    def update(self, x, t):
        if self.x is None: self.x=float(x); self.lt=t; return self.x
        dt = t - self.lt
        if dt <= 0: return self.x
        dx = (x - self.x) / dt
        self.dx = self.sf(dt, self.dc)*dx + (1-self.sf(dt, self.dc))*self.dx
        a = self.sf(dt, self.mc + self.b*abs(self.dx))
        self.x = a*x + (1-a)*self.x; self.lt = t
        return self.x

# 발/발목 키포인트 인덱스 (클라이밍 홀드 위에서 흔들림이 심한 부위)
_FOOT_KPS_ALL = frozenset({15, 16, 17, 18, 19, 20, 21, 22})

class Smoother:
    VF=15; VK=0.35; VD=0.97
    MAX_KP_JUMP_BS      = 0.55  # 일반 키포인트: body_size 배율 초과 이동 → 무시
    MAX_KP_JUMP_BS_FOOT = 0.25  # 발 키포인트: 절반 이하로 엄격히 제한 (홀드 위 튐 방지)

    def __init__(self):
        self.fx=None; self.fy=None; self.pv=None; self.va=None
        self.prev_smooth=None
    def reset(self):
        self.fx=None; self.fy=None; self.pv=None; self.va=None
        self.prev_smooth=None
    def update(self, kps, t):
        if self.fx is None:
            # 발 키포인트는 전용 OEF (mincutoff 훨씬 낮게 → 강한 정지 스무딩)
            self.fx=[OEF(mc=0.10, b=0.01) if i in _FOOT_KPS_ALL else OEF() for i in range(N_KPS)]
            self.fy=[OEF(mc=0.10, b=0.01) if i in _FOOT_KPS_ALL else OEF() for i in range(N_KPS)]
            self.pv=kps[:,2].copy()
            self.va=np.ones(N_KPS, dtype=np.int32)*self.VF
            self.prev_smooth=kps.copy()
            return kps.copy()
        # ── 키포인트 속도 제한: 이전 smooth 대비 급격한 이동 제거 (값 튐 방지) ──
        kps_f = kps.copy()
        if self.prev_smooth is not None:
            tp=[(self.prev_smooth[i,0],self.prev_smooth[i,1]) for i in TORSO_IDS if self.prev_smooth[i,2]>=0.3]
            if len(tp)>=2:
                bh=max(p[1] for p in tp)-min(p[1] for p in tp)
                bw=max(p[0] for p in tp)-min(p[0] for p in tp)
                bs=max(bh,bw,60.0)
                max_mv_body = bs * self.MAX_KP_JUMP_BS
                max_mv_foot = bs * self.MAX_KP_JUMP_BS_FOOT
                for i in range(N_KPS):
                    if kps_f[i,2]<0.1: continue
                    if self.prev_smooth[i,2]>=0.1:
                        mv=math.hypot(kps_f[i,0]-self.prev_smooth[i,0],kps_f[i,1]-self.prev_smooth[i,1])
                        limit = max_mv_foot if i in _FOOT_KPS_ALL else max_mv_body
                        if mv > limit: kps_f[i,2]=0.0
        s = np.zeros_like(kps)
        for i in range(N_KPS):
            s[i,0]=self.fx[i].update(kps_f[i,0], t)
            s[i,1]=self.fy[i].update(kps_f[i,1], t)
            s[i,2]=kps_f[i,2]
        rv=kps_f[:,2]; bv=0.18*rv+0.82*self.pv
        fd=(self.pv>=self.VK)&(bv<self.VK)
        self.va[fd]=np.maximum(0,self.va[fd]-1)
        keep=fd&(self.va>0)
        bv[keep]=np.maximum(bv[keep],self.pv[keep]*self.VD)
        self.va[bv>=self.VK]=self.VF
        s[:,2]=np.clip(bv,0.0,1.0); self.pv=bv.copy()
        self.prev_smooth=s.copy()
        return s


# ── outlier 제거 ─────────────────────────────────────────────────────────────
_LHAND_SET = frozenset(range(91, 112))
_RHAND_SET = frozenset(range(112, 133))
_LFOOT_SET = frozenset({17, 18, 19})
_RFOOT_SET = frozenset({20, 21, 22})

def reject_outliers(kps):
    tp=[(kps[i,0],kps[i,1]) for i in TORSO_IDS if kps[i,2]>=VIS_THRESH]
    if len(tp)<2:
        r=kps.copy()
        for i in range(N_KPS):
            if i not in TORSO_IDS: r[i,2]=0.0
        return r
    cx=float(np.mean([p[0] for p in tp])); cy=float(np.mean([p[1] for p in tp]))
    bh=max(p[1] for p in tp)-min(p[1] for p in tp)
    bw=max(p[0] for p in tp)-min(p[0] for p in tp)
    bs=max(bh,bw,60.0); md=bs*2.2; ms=max(bs*1.2,80.0)
    hand_r = bs*0.6   # wrist 기준 손 크기
    foot_r = bs*0.45  # ankle 기준 발 크기
    r=kps.copy()
    for i in range(N_KPS):
        if i in TORSO_IDS or r[i,2]<VIS_THRESH: continue
        # hand 키포인트: body wrist 기준 거리 제한
        if i in _LHAND_SET:
            lw=kps[9]
            if lw[2]>=VIS_THRESH and math.hypot(r[i,0]-lw[0],r[i,1]-lw[1])>hand_r:
                r[i,2]=0.0; continue
        elif i in _RHAND_SET:
            rw=kps[10]
            if rw[2]>=VIS_THRESH and math.hypot(r[i,0]-rw[0],r[i,1]-rw[1])>hand_r:
                r[i,2]=0.0; continue
        # foot 키포인트: ankle 기준 거리 제한
        elif i in _LFOOT_SET:
            la=kps[15]
            if la[2]>=VIS_THRESH and math.hypot(r[i,0]-la[0],r[i,1]-la[1])>foot_r:
                r[i,2]=0.0; continue
        elif i in _RFOOT_SET:
            ra=kps[16]
            if ra[2]>=VIS_THRESH and math.hypot(r[i,0]-ra[0],r[i,1]-ra[1])>foot_r:
                r[i,2]=0.0; continue
        # body 키포인트: torso 중심 기준
        else:
            if math.hypot(r[i,0]-cx,r[i,1]-cy)>md: r[i,2]=0.0; continue
            p=PARENT_KPS.get(i)
            if p and r[p,2]>=VIS_THRESH and math.hypot(r[i,0]-r[p,0],r[i,1]-r[p,1])>ms:
                r[i,2]=0.0
    return r

# ── 골격 그리기 ──────────────────────────────────────────────────────────────
def draw_skeleton(frame, kps, w, h):
    kps = reject_outliers(kps)
    # hp는 body(0-22)만 사용 — hand/foot 튐이 선 굵기에 영향 주지 않도록
    body_vis = kps[:23][kps[:23, 2] > VIS_THRESH]
    hp = body_vis[:,1].max()-body_vis[:,1].min() if len(body_vis)>2 else h*0.4
    dr  = 2; lw = 1
    mlp = max(40, hp*0.40)
    for s,e in SKEL_PAIRS:
        if kps[s,2]<VIS_THRESH or kps[e,2]<VIS_THRESH: continue
        x1,y1=int(kps[s,0]),int(kps[s,1]); x2,y2=int(kps[e,0]),int(kps[e,1])
        if not(0<x1<w and 0<y1<h and 0<x2<w and 0<y2<h): continue
        if math.hypot(x2-x1,y2-y1)>mlp: continue
        cv2.line(frame,(x1,y1),(x2,y2),seg_color(s,e),lw,cv2.LINE_AA)
    FINGER_TIPS = {95,99,103,107,111, 116,120,124,128,132}
    TOE_KPS     = {17,18,19,20,21,22}
    for i in range(N_KPS):
        if i in HEAD_KPS or i in HAND_KPS or kps[i,2]<VIS_THRESH: continue
        x,y=int(kps[i,0]),int(kps[i,1])
        if not(0<x<w and 0<y<h): continue
        if i in {9,10}:             c=(0,255,255)     # body wrist
        elif i in TOE_KPS:          c=(100,255,255)   # 발가락/발뒤꿈치
        elif i in {15,16}:          c=(100,100,255)   # ankle
        else:                       c=(0,220,255)     # 기본 body
        cv2.circle(frame,(x,y),dr,c,-1,cv2.LINE_AA)
        cv2.circle(frame,(x,y),dr+1,(0,0,0),1,cv2.LINE_AA)
    for i,_ in HL_KPS.items():
        if kps[i,2]<VIS_THRESH: continue
        x,y=int(kps[i,0]),int(kps[i,1])
        if not(0<x<w and 0<y<h): continue
        cv2.circle(frame,(x,y),dr+2,(0,0,0),1,cv2.LINE_AA)
        cv2.circle(frame,(x,y),dr,(0,255,255),-1,cv2.LINE_AA)


# ── 관절 각도 ──────────────────────────────────────────────────────────────────
def calc_angle(a, b, c):
    ax,ay=a[0]-b[0],a[1]-b[1]; cx,cy=c[0]-b[0],c[1]-b[1]
    dot=ax*cx+ay*cy; mag=math.sqrt(ax*ax+ay*ay)*math.sqrt(cx*cx+cy*cy)+1e-9
    return round(math.degrees(math.acos(max(-1.0,min(1.0,dot/mag)))),1)

# ── 포즈 특징값 ────────────────────────────────────────────────────────────────
def landmarks_to_features(kps, w, h, t):
    # 1차 threshold: 정상 신뢰도 (0.35)
    v=lambda i: kps[i,2]>VIS_THRESH
    # 2차 fallback threshold: RTMPose가 낮은 신뢰도라도 일단 사용 (원거리 피사체 대응)
    vf=lambda i: kps[i,2]>0.10
    # visible 여부: 1차 → 없으면 2차
    vis=lambda i: v(i) or vf(i)
    nx=lambda i: round(float(kps[i,0])/w,3)
    ny=lambda i: round(float(kps[i,1])/h,3)
    ft={"t":round(t,2)}
    if vis(5) and vis(7) and vis(9):  ft["le"]=calc_angle(kps[5,:2],kps[7,:2],kps[9,:2])
    if vis(6) and vis(8) and vis(10): ft["re"]=calc_angle(kps[6,:2],kps[8,:2],kps[10,:2])
    if vis(11) and vis(13) and vis(15): ft["lk"]=calc_angle(kps[11,:2],kps[13,:2],kps[15,:2])
    if vis(12) and vis(14) and vis(16): ft["rk"]=calc_angle(kps[12,:2],kps[14,:2],kps[16,:2])
    if vis(5) and vis(11) and vis(13): ft["lh"]=calc_angle(kps[5,:2],kps[11,:2],kps[13,:2])
    if vis(6) and vis(12) and vis(14): ft["rh"]=calc_angle(kps[6,:2],kps[12,:2],kps[14,:2])
    hy=None
    if vis(11) and vis(12): hy=(kps[11,1]+kps[12,1])/2; ft["hy"]=round(float(hy)/h,3)
    sy=None
    if vis(5) and vis(6): sy=(kps[5,1]+kps[6,1])/2
    if hy and vis(15): ft["lhr"]=round((hy-kps[15,1])/h,3)
    if hy and vis(16): ft["rhr"]=round((hy-kps[16,1])/h,3)
    if sy and vis(9):  ft["lwr"]=round((sy-kps[9,1])/h,3)
    if sy and vis(10): ft["rwr"]=round((sy-kps[10,1])/h,3)
    if vis(9):  ft["lwx"]=nx(9)
    if vis(10): ft["rwx"]=nx(10)
    if vis(13): ft["lkx"]=nx(13)
    if vis(15): ft["lax"]=nx(15)
    if vis(14): ft["rkx"]=nx(14)
    if vis(16): ft["rax"]=nx(16)
    if vis(5) and vis(6) and vis(11) and vis(12):
        scx=(kps[5,0]+kps[6,0])/2; scy=(kps[5,1]+kps[6,1])/2
        hcx=(kps[11,0]+kps[12,0])/2; hcy=(kps[11,1]+kps[12,1])/2
        ft["tt"]=round(math.degrees(math.atan2(hcx-scx,hcy-scy+1e-9)),1)
        ft["com"]=[round((scx+hcx)/2/w,3),round((scy+hcy)/2/h,3)]
    # fallback: poseData가 t만 있으면 visible 관절 raw 좌표 포함 (Gemini에 최소한 위치 정보 제공)
    if len(ft) == 1:
        coords = {}
        kp_names = ["nose","leye","reye","lear","rear","lsho","rsho","lelb","relb",
                    "lwri","rwri","lhip","rhip","lkne","rkne","lank","rank"]
        for i, name in enumerate(kp_names):
            if vf(i):
                coords[name] = [nx(i), ny(i), round(float(kps[i,2]),2)]
        if coords:
            ft["kp"] = coords
    return ft


# ── MOG2 ROI ──────────────────────────────────────────────────────────────────
def mog2_roi(fg, shape, min_area=1500):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    c = cv2.morphologyEx(fg,cv2.MORPH_OPEN,k)
    c = cv2.morphologyEx(c,cv2.MORPH_DILATE,k,iterations=8)
    ct,_=cv2.findContours(c,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not ct: return None
    # 가장 큰 움직임 영역 선택
    lg=max(ct,key=cv2.contourArea)
    if cv2.contourArea(lg)<min_area: return None
    x,y,bw,bh=cv2.boundingRect(lg)
    h,w=shape[:2]
    # padding 60% — 손발 뻗을 때도 전신 포함
    pd=int(max(bw,bh)*0.6)
    return (max(0,x-pd),max(0,y-pd),min(w,x+bw+pd),min(h,y+bh+pd))

def heuristic_rois(shape):
    h,w=shape[:2]
    return [(0,int(h*0.4),w,h),(0,int(h*0.6),w,h),(0,int(h*0.75),w,h)]


# ── YOLOv8-seg person detector ────────────────────────────────────────────────
# AGPL-3.0 — 사람 세그멘테이션, 다양한 각도/원거리 robust
# person(class 0) bbox만 사용 → RTMPose ROI

def init_yolo_seg():
    """YOLOv8m-seg 초기화 — medium 모델이 클라이밍 다각도에서 가장 안정적."""
    model = YOLO("yolov8m-seg.pt")
    print("[YOLOseg] yolov8m-seg ready ✓", flush=True)
    return model

def yolo_person_roi(frame, yolo_seg, last_kps=None, conf=0.45, pad=0.18):
    """YOLOv8-seg person(class 0) bbox → ROI (x1,y1,x2,y2).
    last_kps 있으면: 이전 클라이머 중심에 가장 가까운 bbox.
    없으면: 가장 넓은 면적의 bbox (클라이머가 보통 화면에서 가장 크게 찍힘)."""
    h, w = frame.shape[:2]
    results = yolo_seg(frame, classes=[0], conf=conf, verbose=False)
    if not results or len(results[0].boxes) == 0:
        return None

    boxes = results[0].boxes
    if last_kps is not None and len(boxes) > 1:
        # 이전 클라이머 중심에 가장 가까운 bbox 선택
        prev_cx = float(last_kps[:, 0].mean())
        prev_cy = float(last_kps[:, 1].mean())
        best_idx = 0; best_dist = float('inf')
        for i, box in enumerate(boxes.xyxy):
            bx, by = float((box[0]+box[2])/2), float((box[1]+box[3])/2)
            d = math.hypot(bx-prev_cx, by-prev_cy)
            if d < best_dist:
                best_dist = d; best_idx = i
    else:
        # last_kps 없으면 가장 큰 bbox 선택 (클라이머 = 화면에서 가장 큰 사람)
        best_idx = 0; best_area = -1.0
        for i, box in enumerate(boxes.xyxy):
            area = float((box[2]-box[0]) * (box[3]-box[1]))
            if area > best_area:
                best_area = area; best_idx = i

    x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)
    bw, bh = x2-x1, y2-y1
    px = int(max(bw, bh) * pad)
    return (max(0,x1-px), max(0,y1-px), min(w,x2+px), min(h,y2+px))


# ── Body (RTMDet + RTMPose) 초기화 ────────────────────────────────────────────
def init_rtmpose():
    providers = ort.get_available_providers()
    device = "cuda" if "CUDAExecutionProvider" in providers else "cpu"
    print(f"[RTMPose] providers={providers} → device={device}", flush=True)
    # Body: mode='performance' → RTMDet-nano(detector) + RTMPose-m(pose) 자동 선택
    # det/pose 파라미터 생략 — 문자열로 넘기면 URL로 처리되어 ValueError 발생
    # CUDA 사용 가능하면 performance(정확), 없으면 lightweight(CPU에서도 빠름)
    wmode = 'performance' if device == 'cuda' else 'lightweight'
    body = Wholebody(
        to_openpose=False,
        mode=wmode,
        backend='onnxruntime',
        device=device,
    )
    print(f"[RTMPose] Wholebody mode={wmode} device={device}", flush=True)
    # 클라이밍 홀드 오탐지 방지: YOLOX person detection threshold 높이기
    try:
        body.det_model.score_thr = 0.40
        print(f"[RTMPose] det score_thr → 0.40", flush=True)
    except Exception as e:
        print(f"[RTMPose] score_thr 설정 실패 (무시): {e}", flush=True)
    print(f"[RTMPose] Wholebody(133pt) ready ✓", flush=True)
    return body


MAX_JUMP_FRAC = 0.35   # 프레임 대각선 대비 최대 허용 이동 거리 비율

def infer_rtmpose(body, frame, prev_kps=None):
    """Body 파이프라인 추론. 이전 위치와 가장 가까운 사람 선택(홀드 오탐지 방지)."""
    kps_list, sc_list = body(frame)
    if kps_list is None or len(kps_list) == 0:
        return None

    if len(kps_list) == 1 or prev_kps is None:
        kps = np.zeros((N_KPS, 3), dtype=np.float32)
        kps[:, :2] = kps_list[0]
        kps[:, 2]  = sc_list[0]
        return kps

    # 이전 클라이머 중심 계산
    prev_center = prev_kps[:, :2].mean(axis=0)
    h, w = frame.shape[:2]
    max_jump = math.sqrt(w*w + h*h) * MAX_JUMP_FRAC

    best_idx = 0
    best_dist = float('inf')
    for i, kp in enumerate(kps_list):
        center = np.array(kp, dtype=np.float32).mean(axis=0)
        dist = float(np.linalg.norm(center - prev_center))
        if dist < best_dist:
            best_dist = dist
            best_idx = i

    # 가장 가까운 detection도 너무 멀면 (홀드 오탐지) 무시
    if best_dist > max_jump:
        return None

    kps = np.zeros((N_KPS, 3), dtype=np.float32)
    kps[:, :2] = kps_list[best_idx]
    kps[:, 2]  = sc_list[best_idx]
    return kps


def is_valid_pose(kps):
    """torso 키포인트(어깨 2 + 엉덩이 2) 중 2개 이상 VIS_THRESH 통과해야 유효.
    엉뚱한 홀드/배경 감지를 last_kps로 고정되는 것을 방지."""
    torso_visible = sum(1 for i in TORSO_IDS if kps[i, 2] >= VIS_THRESH)
    return torso_visible >= 2

def infer_on_crop(body, frame, roi):
    """ROI 크롭 후 Body 추론, 좌표 복원."""
    if roi is None: return None
    x1,y1,x2,y2 = roi
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0: return None
    kps_list, sc_list = body(crop)
    if kps_list is None or len(kps_list) == 0: return None
    kps = np.zeros((N_KPS, 3), dtype=np.float32)
    kps[:, :2] = kps_list[0]; kps[:, 2] = sc_list[0]
    kps[:, 0] += x1; kps[:, 1] += y1
    return kps


# ── 방향 감지 ──────────────────────────────────────────────────────────────────
MIN_ROT_MARGIN = 15.0   # 1위/2위 차이 이 미만이면 0도 폴백 (오탐지 방지)

def detect_rotation(cap, body, n=8):
    pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    scores = {0:0.0, 90:0.0, 180:0.0, 270:0.0}
    for _ in range(n):
        ret, frame = cap.read()
        if not ret: break
        h,w=frame.shape[:2]
        if w>480: frame=cv2.resize(frame,(480,int(h*480/w)))
        for a in [0,90,180,270]:
            r=rot(frame,a)
            kps_list, sc_list = body(r)
            if kps_list is not None and len(kps_list)>0:
                scores[a] += float(sc_list[0].sum())
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    best = max(scores, key=scores.get)
    sorted_vals = sorted(scores.values(), reverse=True)
    margin = sorted_vals[0] - sorted_vals[1]
    if margin < MIN_ROT_MARGIN:
        print(f"[RTMPose] rotation scores={scores} margin={margin:.1f} < {MIN_ROT_MARGIN} → forcing 0°", flush=True)
        return 0
    print(f"[RTMPose] rotation scores={scores} margin={margin:.1f} → {best}°", flush=True)
    return best


# ── 메인 ───────────────────────────────────────────────────────────────────────
def analyze_video(input_path, output_path, progress_path, pose_json_path=None):
    write_progress(progress_path, 2, "영상 로딩 중...")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        write_progress(progress_path, 100, "오류", {"error": "open failed"})
        sys.exit(1)

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ow    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    oh    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if ow > MAX_PROC_WIDTH:
        sc=MAX_PROC_WIDTH/ow; w=MAX_PROC_WIDTH; h=int(oh*sc)&~1
    else:
        w,h,sc=ow,oh,1.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    print(f"[RTMPose] {ow}x{oh}→{w}x{h} @{fps:.1f}fps total={total}", flush=True)

    write_progress(progress_path, 4, "YOLOv8-seg + RTMPose 초기화 중...")
    body = init_rtmpose()
    yolo_seg = init_yolo_seg()

    write_progress(progress_path, 8, "영상 방향 감지 중...")
    meta_ra = get_metadata_rotation(input_path)
    if meta_ra is not None and meta_ra != 0:
        ra = meta_ra
        print(f"[RTMPose] metadata rotation → {ra}°", flush=True)
    else:
        ra = detect_rotation(cap, body)
    write_progress(progress_path, 12, f"준비 ({w}x{h} rot={ra}°)")

    smoother = Smoother()
    last_kps = None
    fi = 0; ct = 0; ps = []
    si = max(1, int(fps * SAMPLE_EVERY_SEC))
    nd = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        if sc != 1.0:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

        pf  = rot(frame, ra)
        rh, rw = pf.shape[:2]
        t   = fi / fps

        if fi % INFER_EVERY == 0:
            enhanced = enhance_frame(pf)
            raw_kps  = None

            # ── 1단계: YOLOv8m-seg — 전신 bbox 추출 ──────────────────────────────
            yolo_roi = yolo_person_roi(pf, yolo_seg, last_kps=last_kps)

            if yolo_roi is not None:
                raw_kps = infer_on_crop(body, enhanced, yolo_roi)
                # YOLO ROI 결과가 torso 미달이면 → 전체 프레임 RTMPose 직접 실행
                # (YOLO가 홀드를 사람으로 오탐지해도 RTMDet이 실제 사람을 잡아줌)
                if raw_kps is None or not is_valid_pose(raw_kps):
                    print(f"[RTMPose] f={fi} YOLO ROI torso 미달 → full-frame fallback", flush=True)
                    raw_kps = infer_rtmpose(body, enhanced, prev_kps=last_kps)

            elif last_kps is not None:
                # ── 2단계: YOLO 실패 + last_kps 있음 → 이전 위치 기반 ROI ──────
                kx1 = int(last_kps[:, 0].min()); ky1 = int(last_kps[:, 1].min())
                kx2 = int(last_kps[:, 0].max()); ky2 = int(last_kps[:, 1].max())
                kpad = max(kx2-kx1, ky2-ky1, 80)
                k_roi = (max(0,kx1-kpad), max(0,ky1-kpad),
                         min(rw,kx2+kpad), min(rh,ky2+kpad))
                raw_kps = infer_on_crop(body, enhanced, k_roi)

            else:
                # ── 3단계: 초기 프레임 — 전체 프레임 RTMPose (RTMDet 내장 감지) ─
                # heuristic ROI 대신 RTMDet이 전체 프레임에서 사람을 직접 찾음
                raw_kps = infer_rtmpose(body, enhanced)
                if raw_kps is not None:
                    print(f"[RTMPose] f={fi} full-frame 초기 감지 성공", flush=True)

            # torso 품질 검증: 어깨+엉덩이 2개 이상 보여야 last_kps 채택
            if raw_kps is not None and is_valid_pose(raw_kps):
                last_kps = smoother.update(raw_kps, t); nd = 0
                wv=(raw_kps[9,2]+raw_kps[10,2])/2; av=(raw_kps[15,2]+raw_kps[16,2])/2
                if wv>0.4 or av>0.4: ct+=1
            else:
                if raw_kps is not None:
                    print(f"[RTMPose] f={fi} torso 불충분 — last_kps 유지", flush=True)
                nd += 1
                if nd > MAX_GHOST_FRAMES:
                    last_kps=None; smoother.reset()

        if fi % si == 0 and last_kps is not None:
            ps.append(landmarks_to_features(last_kps, rw, rh, t))

        if last_kps is not None:
            draw_skeleton(pf, last_kps, rw, rh)

        out.write(rot_back(pf, ra)); fi += 1

        if fi % 60 == 0:
            pct = 12 + int((fi/max(total,1))*75)
            write_progress(progress_path, min(pct,86), f"{fi}/{total}")
            print(f"[RTMPose] {fi} frames", flush=True)

    cap.release(); out.release()
    write_progress(progress_path, 88, "H.264 변환 중...")

    if pose_json_path and ps:
        try:
            with open(pose_json_path,"w") as f:
                json.dump({"samples":ps,"fps":round(fps,2),"total_frames":fi,
                           "sample_interval_sec":SAMPLE_EVERY_SEC,"rotation_applied":ra},
                          f, separators=(",",":"), cls=NumpyEncoder)
            print(f"[RTMPose] pose JSON: {len(ps)} samples", flush=True)
        except Exception as e:
            print(f"[RTMPose] pose JSON err: {e}", flush=True)

    tmp = output_path + ".raw.mp4"; os.rename(output_path, tmp)
    try:
        pr = subprocess.Popen(
            ["ffmpeg","-y","-i",tmp,"-c:v","libx264","-preset","ultrafast",
             "-crf","26","-movflags","+faststart","-an",output_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        tpct = 88
        while pr.poll() is None:
            time.sleep(3); tpct=min(97,tpct+1)
            write_progress(progress_path, tpct, "H.264 변환 중...")
        if pr.wait(timeout=10)==0: os.unlink(tmp)
        else: os.rename(tmp, output_path)
    except Exception as e:
        if os.path.exists(tmp): os.rename(tmp, output_path)
        print(f"[RTMPose] ffmpeg err: {e}", flush=True)

    inf=max(fi//INFER_EVERY,1); cr=ct/inf; ss=max(0,min(100,int(50+cr*20)))
    write_progress(progress_path, 100, f"Done! {fi} frames",
                   {"contacts":ct,"hold_detections":0,"stability_score":ss,
                    "total_frames":fi,"infer_frames":inf})
    print(f"[RTMPose] Done: {fi} frames | contacts={ct}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: mediapipe_analyze.py <input> <output> <progress_json> [pose_json]")
        sys.exit(1)
    analyze_video(sys.argv[1], sys.argv[2], sys.argv[3],
                  sys.argv[4] if len(sys.argv)>=5 else None)
