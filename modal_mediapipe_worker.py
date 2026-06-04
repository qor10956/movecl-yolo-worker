"""
Modal GPU worker — YOLOv8m-pose (COCO 17 keypoints)
골격 오버레이 영상 생성

라이선스: YOLOv8 AGPL-3.0 (소스 공개)

Endpoint:
  POST <web_url>
    multipart/form-data with field name "video" (.mp4)
  Response:
    Content-Type: video/mp4
    X-Pose-Stats: JSON
    Body: processed mp4 (H.264, iOS-compatible)

Deploy:
  modal deploy modal_mediapipe_worker.py
"""
import os
import sys
import json
import tempfile
import subprocess as _sp

import modal

image = (
    # CUDA 12.1 + cuDNN 8 — onnxruntime-gpu 1.20.0은 CUDA 12.x 공식 지원
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0", "libglib2.0-dev")
    .pip_install(
        "rtmlib",
        "onnxruntime-gpu==1.18.0",
        "opencv-python-headless",
        "numpy<2",
        "ultralytics",               # AGPL-3.0 — YOLOv8m-seg person detection
        "fastapi[standard]",
        "python-multipart",
        "google-generativeai>=0.8.0",
    )
    # onnxruntime-gpu CUDA 11.x wheel이 libcublasLt.so.11 찾는 문제 해결
    # apt로 cuBLAS 설치 후 .so.11 symlink 생성
    .run_commands(
        "apt-get update -qq && apt-get install -y --no-install-recommends libcublas-12-1 && "
        "SRC=$(find /usr -name 'libcublasLt.so.12*' 2>/dev/null | head -1) && "
        "ln -sf $SRC /usr/lib/x86_64-linux-gnu/libcublasLt.so.11 && "
        "ldconfig && echo 'cuBLAS symlink OK:' $SRC"
    )
    .add_local_file("yolo_pose_analyze.py", "/root/yolo_pose_analyze.py")
)

app = modal.App("movecl-mediapipe", image=image)


@app.function(
    gpu="A10G",
    timeout=600,
    scaledown_window=300,
    min_containers=0,
    secrets=[],
)
@modal.asgi_app()
def fastapi_app():
    import subprocess
    from fastapi import FastAPI, UploadFile, File
    from fastapi.responses import Response

    sys.path.insert(0, "/root")
    os.chdir("/root")
    from yolo_pose_analyze import analyze_video

    web = FastAPI()

    @web.get("/healthz")
    def healthz():
        return {"ok": True, "engine": "yolo-pose+rtm-hand"}

    def _get_video_rotation(path: str) -> int:
        """ffprobe로 영상 rotation 메타데이터 읽기 (0/90/180/270)."""
        try:
            r = _sp.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_streams", path],
                capture_output=True, text=True, timeout=15,
            )
            data = json.loads(r.stdout)
            for s in data.get("streams", []):
                if s.get("codec_type") != "video":
                    continue
                # ffmpeg 최신 포맷: side_data_list → {"rotation": -90}
                for sd in s.get("side_data_list", []):
                    rot = sd.get("rotation")
                    if rot is not None:
                        return abs(int(rot))
                # 구버전 포맷: tags → {"rotate": "90"}
                tags = s.get("tags", {})
                if "rotate" in tags:
                    return abs(int(tags["rotate"]))
        except Exception as e:
            print(f"[ffprobe] rotation 감지 실패: {e}", flush=True)
        return 0

    @web.post("/")
    async def render(video: UploadFile = File(...)):
        with tempfile.TemporaryDirectory() as td:
            raw_path  = os.path.join(td, "raw.mp4")
            in_path   = os.path.join(td, "in.mp4")
            out_path  = os.path.join(td, "out.mp4")
            prog_path = os.path.join(td, "prog.json")

            with open(raw_path, "wb") as f:
                while True:
                    chunk = await video.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)

            # ffmpeg 자동 rotation 처리 (autorotate 기본 ON)
            # -noautorotate + 수동 transpose 제거 → ffmpeg이 rotation 메타데이터 직접 적용
            # iPhone H264_960x540 등 다양한 export preset에서 안정적으로 동작
            vf = "scale='if(gt(iw,ih),min(960,iw),-2)':'if(gt(iw,ih),-2,min(960,ih))'"
            print(f"[worker] ffmpeg auto-rotate → scale filter: {vf}", flush=True)
            try:
                _sp.run(
                    ["ffmpeg", "-y", "-i", raw_path,
                     "-vf", vf,
                     "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-an",
                     in_path],
                    check=True, capture_output=True,
                )
            except _sp.CalledProcessError:
                in_path = raw_path

            pose_path = os.path.join(td, "pose.json")
            try:
                analyze_video(in_path, out_path, prog_path, pose_path)
            except SystemExit as e:
                return Response(
                    content=json.dumps({"error": f"analyze_video exited: {e.code}"}).encode(),
                    media_type="application/json",
                    status_code=500,
                )

            if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
                return Response(
                    content=json.dumps({"error": "output video missing or empty"}).encode(),
                    media_type="application/json",
                    status_code=500,
                )

            with open(out_path, "rb") as f:
                video_bytes = f.read()

            stats: dict = {}
            try:
                with open(prog_path) as f:
                    stats = json.load(f)
            except Exception:
                pass

            # YOLO와 동일한 헤더 포맷 — 클라이언트 코드 변경 불필요
            stats_payload = {
                "contacts":        stats.get("contacts", 0),
                "hold_detections": stats.get("hold_detections", 0),
                "stability_score": stats.get("stability_score", 50),
                "total_frames":    stats.get("total_frames", 0),
                "infer_frames":    stats.get("infer_frames", 0),
            }

            resp_headers = {"X-Pose-Stats": json.dumps(stats_payload)}
            # Gemini Files API 업로드는 Railway에서 수행 (Modal GPU 점유 시간 단축)
            # X-Gemini-File-Uri 헤더 제거 — Railway가 job.outputPath로 업로드

            # 포즈 JSON → X-Pose-Data 헤더 (서버에서 Gemini 기술 분석용)
            # 샘플 수 40개로 제한 + float 소수점 3자리 반올림 → 헤더 크기 ~30KB 이내
            if os.path.exists(pose_path):
                try:
                    with open(pose_path) as f:
                        pose_data = json.load(f)

                    def _round_floats(obj, digits=3):
                        if isinstance(obj, float):
                            return round(obj, digits)
                        if isinstance(obj, dict):
                            return {k: _round_floats(v, digits) for k, v in obj.items()}
                        if isinstance(obj, list):
                            return [_round_floats(v, digits) for v in obj]
                        return obj

                    samples = pose_data.get("samples", [])
                    # Gemini 코칭 + 크롭 추적용: 관절 각도 + 무게중심 포함
                    # 60샘플 × ~160자 ≈ 9.6KB — HTTP 헤더 한도 이내
                    MAX_SAMPLES = 60
                    if len(samples) > MAX_SAMPLES:
                        step = len(samples) / MAX_SAMPLES
                        samples = [samples[int(i * step)] for i in range(MAX_SAMPLES)]
                    # 주요 관절 각도 + 무게중심 + 발 높이 포함
                    slim_samples = []
                    for s in samples:
                        if "t" not in s:
                            continue
                        entry = {"t": round(s["t"], 2)}
                        # 관절 각도 (정수 반올림 — 팔꿈치/무릎/힙)
                        for key in ("le", "re", "lk", "rk", "lh", "rh"):
                            if key in s:
                                entry[key] = round(s[key])
                        # 몸통 기울기 (소수 1자리)
                        if "tt" in s:
                            entry["tt"] = round(s["tt"], 1)
                        # 무게중심 (크롭 추적 + COG 분석용)
                        if "com" in s and len(s.get("com", [])) >= 2:
                            entry["com"] = [round(s["com"][0], 3), round(s["com"][1], 3)]
                        # 발 높이 (힐훅/토훅 감지)
                        for key in ("lhr", "rhr", "ltr", "rtr"):
                            if key in s:
                                entry[key] = round(s[key], 3)
                        # 손목 높이 (락오프 감지)
                        for key in ("lwr", "rwr"):
                            if key in s:
                                entry[key] = round(s[key], 3)
                        slim_samples.append(entry)
                    slim_data = {"samples": slim_samples}
                    pose_str = json.dumps(slim_data, separators=(",", ":"))
                    resp_headers["X-Pose-Data"] = pose_str
                    print(f"[MediaPipe] pose header: {len(slim_samples)} samples, {len(pose_str)} bytes", flush=True)
                except Exception as e:
                    print(f"[MediaPipe] pose header error: {e}", flush=True)

            return Response(
                content=video_bytes,
                media_type="video/mp4",
                headers=resp_headers,
            )

    return web
