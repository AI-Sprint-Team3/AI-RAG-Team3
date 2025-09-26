#!/bin/bash                # 이 스크립트를 bash로 실행하겠다는 선언 (shebang)
set -e                     # 스크립트 내 명령이 실패하면 즉시 종료

CHROMA_URL="${CHROMA_SERVER:-http://chroma:8000}"  # CHROMA_SERVER env가 있으면 사용, 없으면 기본값 사용
HEALTH_PATH="/api/v2/heartbeat"                    # Chroma의 건강체크 엔드포인트 경로
TIMEOUT=${WAIT_TIMEOUT:-60}                        # 최대 대기시간(초). env WAIT_TIMEOUT이 있으면 사용, 아니면 60초
SLEEP_INTERVAL=2                                   # 실패시 재시도 간격(초)

echo "Waiting for Chroma at ${CHROMA_URL}${HEALTH_PATH} (timeout ${TIMEOUT}s)..."  # 사용자 안내 메시지

start_ts=$(date +%s)                               # 현재 시간(초)을 기록
while true; do                                     # 무한 루프 시작
  if curl -s "${CHROMA_URL}${HEALTH_PATH}" >/dev/null 2>&1; then
    # curl이 성공하면(HTTP 200 등) Chroma가 준비된 것
    echo "Chroma is available."
    break                                          # 루프 탈출
  fi
  now_ts=$(date +%s)                               # 재시도 시점 시간
  elapsed=$((now_ts - start_ts))                   # 경과 시간 계산
  if [ "$elapsed" -ge "$TIMEOUT" ]; then
    # 이미 타임아웃을 초과했으면 에러 출력 후 종료
    echo "Timed out waiting for Chroma after ${TIMEOUT}s" >&2
    exit 1
  fi
  sleep $SLEEP_INTERVAL                             # 대기 후 재시도
done

exec "$@"                                          # 인자로 넘겨진 명령(예: python main.py)을 실행(현재 쉘 대체)