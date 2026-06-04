# movecl-yolo-worker

**무브클(MoveCl)** iOS 앱의 클라이밍 영상 분석 GPU 워커 오픈소스 공개.

YOLOv8m-pose를 사용해 클라이밍 영상에서 골격(skeleton) 오버레이 영상을 생성합니다.  
AGPL-3.0 라이선스 의무에 따라 소스코드를 공개합니다.

## 파일 구성

| 파일 | 설명 |
|------|------|
| `modal_mediapipe_worker.py` | Modal GPU 워커 (FastAPI) — 메인 엔드포인트 |
| `yolo_pose_analyze.py` | YOLOv8m-pose 포즈 분석 (워커에서 임포트) |
| `mediapipe_analyze.py` | RTMPose 분석 로직 (레거시, 현재 미사용) |

## 동작 방식

1. 클라이언트(API 서버)가 `.mp4` 영상을 multipart POST로 전송
2. Modal GPU 워커가 **YOLOv8m-pose**로 COCO 17 키포인트 감지
3. 프레임별 골격 오버레이 영상 생성
4. 처리된 `.mp4` + 포즈 JSON 반환

## 배포 방법 (Modal)

```bash
# Modal 계정 필요: https://modal.com

pip install modal
modal token new

# 워커 배포
modal deploy modal_mediapipe_worker.py

# 배포 후 URL 확인
# https://YOUR_ACCOUNT--movecl-mediapipe-fastapi-app.modal.run
```

## 환경변수

`.env.example` 참고.

## 라이선스

- **YOLOv8** (Ultralytics): [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)
- 이 저장소: AGPL-3.0

## 관련 프로젝트

- **무브클(MoveCl)**: 클라이밍 AI 코칭 iOS 앱
- App Store: [링크](https://apps.apple.com/app/id6766141018)
