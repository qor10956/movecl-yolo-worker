#!/usr/bin/env python3
"""
YOLOv8m-pose — Climbing Video Analyzer (단일 모델)
라이선스: AGPL-3.0 (소스 공개)

파이프라인:
  YOLOv8m-pose → 17 COCO body 키포인트 + bbox
  INFER_EVERY=3 (3프레임마다 추론 → ~1분/영상)

키포인트 (COCO 17):
  0:nose 1:leye 2:reye 3:lear 4:rear
  5:lsho 6:rsho 7:lelb 8:relb 9:lwri 10:rwri
  11:lhip 12:rhip 13:lkne 14:rkne 15:lank 16:rank

Usage: python3 yolo_pose_analyze.py <input> <output> <progress_json> [pose_json]
"""
import sys, os, json, subprocess, time, math
import numpy as np
import cv2
from ultralytics import YOLO

# ── 상수 ─────────────────────────────────────────────────────────────────────
N_KPS             = 17
INFER_EVERY       = 3      # 3프레임마다 추론 (초반 감지 후)
INFER_WARMUP_SEC  = 15.0   # 처음 N초는 매 프레임 추론 — 초반 감지 보장
INFER_SIZE        = 640    # 일반 구간 imgsz
INFER_SIZE_WARMUP = 1024   # 초반 15초 고화질 추론 — 작고 납작한 자세 감지 강화
MAX_PROC_WIDTH    = 1280
ONEEURO_MINCUTOFF = 0.35   # 너무 높으면 노이즈에 민감 → 균형점
ONEEURO_BETA      = 0.06   # 빠른 움직임 반응 유지하되 jitter 감소
ONEEURO_DCUTOFF   = 1.0
SAMPLE_EVERY_SEC  = 0.5
VIS_THRESH        = 0.30   # 구 파이프라인과 동일
MAX_GHOST_FRAMES  = 75   # ~2.5초(30fps 기준) — 감지 실패해도 마지막 포즈 유지
CONF_YOLO         = 0.25   # 낮추되 오탐지 방지
MIN_BBOX_FRAC     = 0.01   # bbox 면적이 프레임의 1% 미만이면 무시 (홀드 오탐 차단)

# ── COCO 17 스켈레톤 연결 ────────────────────────────────────────────────────
BODY_PAIRS = [
    (5,7),(7,9),(6,8),(8,10),      # 팔
    (5,6),(5,11),(6,12),(11,12),   # 몸통
    (11,13),(13,15),(12,14),(14,16), # 다리
]
TORSO_IDS = frozenset({5, 6, 11, 12})
PARENT_KPS = {7:5, 9:7, 8:6, 10:8, 13:11, 15:13, 14:12, 16:14}

# 부위별 색상 (BGR)
_ARM_P   = frozenset({(5,7),(7,9),(6,8),(8,10)})
_TORSO_P = frozenset({(5,6),(5,11),(6,12),(11,12)})
_LEG_P   = frozenset({(11,13),(13,15),(12,14),(14,16)})

def seg_color(s, e):
    p = (s, e)
    if p in _ARM_P:   return (0, 200, 255)   # 주황 (BGR)
    if p in _TORSO_P: return (0, 255, 80)    # 초록
    if p in _LEG_P:   return (200, 80, 255)  # 보라
    return (200, 200, 200)

# 레이블 강조 키포인트
HL_LABELS = {9: "L-Hand", 10: "R-Hand", 15: "L-Foot", 16: "R-Foot"}
SKIP_HEAD = frozenset({0, 1, 2, 3, 4})

# ── 유틸 ──────────────────────────────────────────────────────────────────────
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)

def write_progress(path, pct, msg, extra=None):
    try:
        d = {"pct": pct, "msg": msg, "done": pct >= 100}
        if extra: d.update(extra)
        with open(path, "w") as f: json.dump(d, f)
    except Exception: pass

def get_metadata_rotation(input_path):
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
                if v is not None: return (-int(v)) % 360
            tags = s.get("tags", {})
            v = tags.get("rotate")
            if v: return int(v) % 360
    except Exception: pass
    return None

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


# ── OneEuroFilter ─────────────────────────────────────────────────────────────
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


class Smoother:
    VF=15; VK=0.30; VD=0.97
    MAX_JUMP_BS = 0.40   # 한 프레임 최대 이동 = bbox 40% — jitter 억제

    def __init__(self):
        self.fx=None; self.fy=None; self.pv=None; self.va=None
        self.prev_smooth=None

    def reset(self):
        self.fx=None; self.fy=None; self.pv=None; self.va=None
        self.prev_smooth=None

    def update(self, kps, t):
        if self.fx is None:
            self.fx=[OEF() for _ in range(N_KPS)]
            self.fy=[OEF() for _ in range(N_KPS)]
            self.pv=kps[:,2].copy()
            self.va=np.ones(N_KPS, dtype=np.int32)*self.VF
            self.prev_smooth=kps.copy()
            return kps.copy()

        kps_f = kps.copy()
        if self.prev_smooth is not None:
            tp=[(self.prev_smooth[i,0],self.prev_smooth[i,1])
                for i in TORSO_IDS if self.prev_smooth[i,2]>=0.3]
            if len(tp)>=2:
                bh=max(p[1] for p in tp)-min(p[1] for p in tp)
                bw=max(p[0] for p in tp)-min(p[0] for p in tp)
                bs=max(bh,bw,60.0)
                max_mv=bs*self.MAX_JUMP_BS
                for i in range(N_KPS):
                    if kps_f[i,2]<0.1: continue
                    if self.prev_smooth[i,2]>=0.1:
                        mv=math.hypot(kps_f[i,0]-self.prev_smooth[i,0],
                                      kps_f[i,1]-self.prev_smooth[i,1])
                        if mv>max_mv: kps_f[i,2]=0.0

        s=np.zeros_like(kps)
        for i in range(N_KPS):
            s[i,0]=self.fx[i].update(kps_f[i,0], t)
            s[i,1]=self.fy[i].update(kps_f[i,1], t)
            s[i,2]=kps_f[i,2]

        rv=kps_f[:,2]; bv=0.30*rv+0.70*self.pv   # 0.18→0.30: 현재 감지값 가중치 높여 반응성 개선
        fd=(self.pv>=self.VK)&(bv<self.VK)
        self.va[fd]=np.maximum(0,self.va[fd]-1)
        keep=fd&(self.va>0)
        bv[keep]=np.maximum(bv[keep],self.pv[keep]*self.VD)
        self.va[bv>=self.VK]=self.VF
        s[:,2]=np.clip(bv,0.0,1.0); self.pv=s[:,2].copy()
        self.prev_smooth=s.copy()
        return s


# ── outlier 제거 ─────────────────────────────────────────────────────────────
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
    bs=max(bh,bw,60.0); md=bs*2.5; ms=max(bs*1.3,80.0)
    r=kps.copy()
    for i in range(N_KPS):
        if i in TORSO_IDS or r[i,2]<VIS_THRESH: continue
        if math.hypot(r[i,0]-cx,r[i,1]-cy)>md: r[i,2]=0.0; continue
        p=PARENT_KPS.get(i)
        if p and r[p,2]>=VIS_THRESH and math.hypot(r[i,0]-r[p,0],r[i,1]-r[p,1])>ms:
            r[i,2]=0.0
    return r


# ── 골격 그리기 ────────────────────────────────────────────────────────────────
def draw_skeleton(frame, kps, fw, fh):
    kps = reject_outliers(kps)
    body_vis = kps[kps[:, 2] > VIS_THRESH]
    hp = body_vis[:,1].max()-body_vis[:,1].min() if len(body_vis)>2 else fh*0.4
    scale = max(fh, fw)
    dr  = max(2, int(scale * 0.004))
    lw  = max(1, int(scale * 0.002))
    mlp = max(50, hp*0.50)

    # 연결선
    for s, e in BODY_PAIRS:
        if kps[s,2]<VIS_THRESH or kps[e,2]<VIS_THRESH: continue
        x1,y1=int(kps[s,0]),int(kps[s,1]); x2,y2=int(kps[e,0]),int(kps[e,1])
        if not(0<x1<fw and 0<y1<fh and 0<x2<fw and 0<y2<fh): continue
        if math.hypot(x2-x1,y2-y1)>mlp: continue
        cv2.line(frame,(x1,y1),(x2,y2),seg_color(s,e),lw,cv2.LINE_AA)

    # 키포인트 점
    for i in range(N_KPS):
        if i in SKIP_HEAD or kps[i,2]<VIS_THRESH: continue
        x,y=int(kps[i,0]),int(kps[i,1])
        if not(0<x<fw and 0<y<fh): continue
        c = (0,255,255) if i in {9,10} else (0,220,255)  # wrist=노랑, 나머지=주황노랑
        cv2.circle(frame,(x,y),dr,c,-1,cv2.LINE_AA)
        cv2.circle(frame,(x,y),dr+1,(0,0,0),1,cv2.LINE_AA)

    # 관절 레이블 (겹침 방지 — 이미 그린 위치와 너무 가까우면 y 오프셋)
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    fs   = max(0.25, scale * 0.00022)
    MIN_LABEL_GAP = max(12, dr * 3)   # 레이블 간 최소 거리 (px)
    drawn_label_pos = []
    for i, label in HL_LABELS.items():
        if kps[i,2]<VIS_THRESH: continue
        x,y=int(kps[i,0]),int(kps[i,1])
        if not(0<x<fw and 0<y<fh): continue
        # 겹침 방지: 기존 레이블과 너무 가까우면 y 올림
        lx, ly = x + dr + 4, y - 5
        for (px, py) in drawn_label_pos:
            if abs(lx - px) < MIN_LABEL_GAP * 3 and abs(ly - py) < MIN_LABEL_GAP:
                ly = py - MIN_LABEL_GAP  # 위로 밀기
        drawn_label_pos.append((lx, ly))
        cv2.circle(frame,(x,y),dr+2,(0,0,0),1,cv2.LINE_AA)
        cv2.circle(frame,(x,y),dr,(0,255,255),-1,cv2.LINE_AA)
        cv2.putText(frame,label,(lx,ly),FONT,fs,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(frame,label,(lx,ly),FONT,fs,(255,255,255),1,cv2.LINE_AA)


# ── 관절 각도 / 포즈 피처 ────────────────────────────────────────────────────
def calc_angle(a, b, c):
    ax,ay=a[0]-b[0],a[1]-b[1]; cx_,cy_=c[0]-b[0],c[1]-b[1]
    dot=ax*cx_+ay*cy_; mag=math.sqrt(ax*ax+ay*ay)*math.sqrt(cx_*cx_+cy_*cy_)+1e-9
    return round(math.degrees(math.acos(max(-1.0,min(1.0,dot/mag)))),1)

def landmarks_to_features(kps, fw, fh, t):
    v  = lambda i: kps[i,2]>VIS_THRESH
    nx = lambda i: round(float(kps[i,0])/fw,3)
    ny = lambda i: round(float(kps[i,1])/fh,3)
    ft = {"t":round(t,2)}
    if v(5) and v(7) and v(9):   ft["le"]=calc_angle(kps[5,:2],kps[7,:2],kps[9,:2])
    if v(6) and v(8) and v(10):  ft["re"]=calc_angle(kps[6,:2],kps[8,:2],kps[10,:2])
    if v(11) and v(13) and v(15):ft["lk"]=calc_angle(kps[11,:2],kps[13,:2],kps[15,:2])
    if v(12) and v(14) and v(16):ft["rk"]=calc_angle(kps[12,:2],kps[14,:2],kps[16,:2])
    if v(5) and v(11) and v(13): ft["lh"]=calc_angle(kps[5,:2],kps[11,:2],kps[13,:2])
    if v(6) and v(12) and v(14): ft["rh"]=calc_angle(kps[6,:2],kps[12,:2],kps[14,:2])
    hy=None
    if v(11) and v(12): hy=(kps[11,1]+kps[12,1])/2; ft["hy"]=round(float(hy)/fh,3)
    sy=None
    if v(5) and v(6):   sy=(kps[5,1]+kps[6,1])/2
    if hy and v(15): ft["lhr"]=round((hy-kps[15,1])/fh,3)
    if hy and v(16): ft["rhr"]=round((hy-kps[16,1])/fh,3)
    if sy and v(9):  ft["lwr"]=round((sy-kps[9,1])/fh,3)
    if sy and v(10): ft["rwr"]=round((sy-kps[10,1])/fh,3)
    if v(9):  ft["lwx"]=nx(9)
    if v(10): ft["rwx"]=nx(10)
    if v(15): ft["lax"]=nx(15)
    if v(16): ft["rax"]=nx(16)
    if v(5) and v(6) and v(11) and v(12):
        scx=(kps[5,0]+kps[6,0])/2; scy=(kps[5,1]+kps[6,1])/2
        hcx=(kps[11,0]+kps[12,0])/2; hcy=(kps[11,1]+kps[12,1])/2
        ft["tt"]=round(math.degrees(math.atan2(hcx-scx,hcy-scy+1e-9)),1)
        ft["com"]=[round((scx+hcx)/2/fw,3),round((scy+hcy)/2/fh,3)]
    return ft


# ── torso 유효성 ───────────────────────────────────────────────────────────────
def is_valid_pose(kps):
    return sum(1 for i in TORSO_IDS if kps[i,2]>=VIS_THRESH) >= 1


# ── YOLOv8m-pose 추론 ─────────────────────────────────────────────────────────
def infer_yolo_pose(model, frame, prev_kps=None, imgsz=INFER_SIZE):
    """YOLOv8m-pose → (17,3) [x,y,conf]. bbox 크기 필터 + 이전 위치 기반 사람 선택."""
    h, w = frame.shape[:2]
    frame_area = float(h * w)
    min_area = frame_area * MIN_BBOX_FRAC

    results = model(frame, classes=[0], conf=CONF_YOLO,
                    imgsz=imgsz, verbose=False)
    if not results or results[0].keypoints is None:
        return None

    kps_data = results[0].keypoints.data    # (N, 17, 3)
    boxes    = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes else None
    if kps_data is None or len(kps_data) == 0:
        return None

    kps_np = kps_data.cpu().numpy().astype(np.float32)
    n = len(kps_np)

    # bbox 최소 면적 필터 — 홀드 오탐 차단
    valid_idx = []
    for i in range(n):
        if boxes is not None and i < len(boxes):
            bx = boxes[i]
            area = float((bx[2]-bx[0])*(bx[3]-bx[1]))
            if area < min_area:
                continue
        valid_idx.append(i)

    if not valid_idx:
        return None

    if len(valid_idx) == 1:
        chosen = valid_idx[0]
    elif prev_kps is not None:
        # 이전 위치에 가장 가까운 사람
        prev_cx = float(prev_kps[:,0].mean())
        prev_cy = float(prev_kps[:,1].mean())
        best_d = float('inf'); chosen = valid_idx[0]
        for i in valid_idx:
            cx = float(kps_np[i,:,0].mean())
            cy = float(kps_np[i,:,1].mean())
            d = math.hypot(cx-prev_cx, cy-prev_cy)
            if d < best_d: best_d=d; chosen=i
    else:
        # 첫 감지 — bbox 면적 최대 (클라이머 = 화면에서 가장 큰 사람)
        best_area = -1.0; chosen = valid_idx[0]
        for i in valid_idx:
            if boxes is not None and i < len(boxes):
                bx = boxes[i]
                area = float((bx[2]-bx[0])*(bx[3]-bx[1]))
                if area > best_area: best_area=area; chosen=i

    kps = kps_np[chosen]   # (17,3)

    # torso 1개 이상 가시 — 완전 오탐지 최후 방어
    if sum(1 for i in TORSO_IDS if kps[i,2] >= VIS_THRESH) < 1:
        return None

    return kps


# ── 방향 감지 ─────────────────────────────────────────────────────────────────
def detect_rotation(cap, model, n=8):
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
            results=model(r,classes=[0],conf=0.20,imgsz=INFER_SIZE,verbose=False)
            if results and len(results[0].boxes)>0:
                confs=results[0].boxes.conf.cpu().numpy()
                scores[a]+=float(confs.sum())
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    best=max(scores,key=scores.get)
    sorted_vals=sorted(scores.values(),reverse=True)
    margin=sorted_vals[0]-sorted_vals[1]
    if margin<0.3:
        print(f"[YOLOpose] rotation margin={margin:.2f} 작음 → 0°",flush=True)
        return 0
    print(f"[YOLOpose] rotation scores={scores} → {best}°",flush=True)
    return best


# ── 메인 ─────────────────────────────────────────────────────────────────────
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
    print(f"[YOLOpose] {ow}x{oh}→{w}x{h} @{fps:.1f}fps total={total} every={INFER_EVERY}",flush=True)

    write_progress(progress_path, 4, "YOLOv8m-pose 초기화 중...")
    model = YOLO("yolov8m-pose.pt")
    print("[YOLOpose] yolov8m-pose ready ✓",flush=True)

    write_progress(progress_path, 8, "영상 방향 확인 중...")
    ra = get_metadata_rotation(input_path) or 0
    print(f"[YOLOpose] rotation={ra}°",flush=True)
    write_progress(progress_path, 10, f"준비 ({w}x{h} rot={ra}°)")

    smoother = Smoother()
    last_kps = None
    fi=0; ct=0; ps=[]; nd=0
    si=max(1,int(fps*SAMPLE_EVERY_SEC))

    while True:
        ret, frame = cap.read()
        if not ret: break
        if sc != 1.0:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

        pf = rot(frame, ra)
        rh, rw = pf.shape[:2]
        t = fi / fps

        in_warmup = t < INFER_WARMUP_SEC
        every = 1 if in_warmup else INFER_EVERY
        if fi % every == 0:
            sz = INFER_SIZE_WARMUP if in_warmup else INFER_SIZE
            raw_kps = infer_yolo_pose(model, pf, last_kps, imgsz=sz)

            if raw_kps is not None:
                last_kps = smoother.update(raw_kps, t); nd = 0
                wv=(raw_kps[9,2]+raw_kps[10,2])/2; av=(raw_kps[15,2]+raw_kps[16,2])/2
                if wv>0.3 or av>0.3: ct+=1
            else:
                nd+=1
                if nd>MAX_GHOST_FRAMES:
                    last_kps=None; smoother.reset()

        if fi%si==0 and last_kps is not None:
            ps.append(landmarks_to_features(last_kps, rw, rh, t))

        if last_kps is not None:
            draw_skeleton(pf, last_kps, rw, rh)

        out.write(rot_back(pf, ra)); fi+=1

        if fi%60==0:
            pct=12+int((fi/max(total,1))*75)
            write_progress(progress_path,min(pct,86),f"{fi}/{total}")
            print(f"[YOLOpose] {fi} frames",flush=True)

    cap.release(); out.release()
    write_progress(progress_path,88,"H.264 변환 중...")

    if pose_json_path and ps:
        try:
            with open(pose_json_path,"w") as f:
                json.dump({"samples":ps,"fps":round(fps,2),"total_frames":fi,
                           "sample_interval_sec":SAMPLE_EVERY_SEC,"rotation_applied":ra},
                          f,separators=(",",":"),cls=NumpyEncoder)
            print(f"[YOLOpose] pose JSON: {len(ps)} samples",flush=True)
        except Exception as e:
            print(f"[YOLOpose] pose JSON err: {e}",flush=True)

    tmp=output_path+".raw.mp4"; os.rename(output_path,tmp)
    try:
        pr=subprocess.Popen(
            ["ffmpeg","-y","-i",tmp,"-c:v","libx264","-preset","ultrafast",
             "-crf","26","-movflags","+faststart","-an",output_path],
            stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        tpct=88
        while pr.poll() is None:
            time.sleep(3); tpct=min(97,tpct+1)
            write_progress(progress_path,tpct,"H.264 변환 중...")
        if pr.wait(timeout=10)==0: os.unlink(tmp)
        else: os.rename(tmp,output_path)
    except Exception as e:
        if os.path.exists(tmp): os.rename(tmp,output_path)
        print(f"[YOLOpose] ffmpeg err: {e}",flush=True)

    inf=max(fi//INFER_EVERY,1); cr=ct/inf; ss=max(0,min(100,int(50+cr*20)))
    write_progress(progress_path,100,f"Done! {fi} frames",
                   {"contacts":ct,"hold_detections":0,"stability_score":ss,
                    "total_frames":fi,"infer_frames":inf})
    print(f"[YOLOpose] Done: {fi} frames | contacts={ct}",flush=True)


if __name__=="__main__":
    if len(sys.argv)<4:
        print("Usage: yolo_pose_analyze.py <input> <output> <progress_json> [pose_json]")
        sys.exit(1)
    analyze_video(sys.argv[1],sys.argv[2],sys.argv[3],
                  sys.argv[4] if len(sys.argv)>=5 else None)
