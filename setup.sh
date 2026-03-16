#!/usr/bin/env bash
set -euo pipefail

# ── slm-factory 원클릭 셋업 ──────────────────────────────────────────
# uv 기반 의존성 설치 + Ollama 모델 준비까지 한 번에 수행합니다.
# 사용법: ./setup.sh
# ─────────────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${CYAN}▸${NC} $*"; }
ok()    { echo -e "${GREEN}✓${NC} $*"; }
warn()  { echo -e "${YELLOW}⚠${NC} $*"; }
fail()  { echo -e "${RED}✗${NC} $*"; exit 1; }

TEACHER_MODEL="${SLM_TEACHER_MODEL:-qwen3.5:9b}"
OLLAMA_URL="${OLLAMA_HOST:-http://localhost:11434}"

echo ""
echo -e "${BOLD}slm-factory 셋업${NC}"
echo "────────────────────────────────────────"
echo ""

# ── 1. uv 확인 / 설치 ──────────────────────────────────────────────
info "uv 확인 중..."
if command -v uv &>/dev/null; then
    ok "uv $(uv --version 2>/dev/null | head -1)"
else
    info "uv 설치 중..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # 설치 직후 PATH 반영
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    command -v uv &>/dev/null || fail "uv 설치 실패. https://docs.astral.sh/uv/ 참고"
    ok "uv 설치 완료"
fi

# ── 2. 의존성 설치 ──────────────────────────────────────────────────
info "의존성 설치 중... (첫 실행 시 2~5분 소요)"
uv sync --extra all 2>&1 | tail -1
ok "의존성 설치 완료"

# ── 3. Ollama 확인 ──────────────────────────────────────────────────
info "Ollama 확인 중..."
if ! command -v ollama &>/dev/null; then
    warn "Ollama가 설치되어 있지 않습니다"
    echo ""
    echo "  설치: https://ollama.com"
    echo "  설치 후 다시 ./setup.sh 를 실행하세요"
    echo ""
    fail "Ollama 필요"
fi
ok "Ollama 설치됨"

# Ollama 서버 확인
if ! curl -sf "${OLLAMA_URL}/api/tags" &>/dev/null; then
    warn "Ollama 서버가 실행 중이 아닙니다"
    info "Ollama 시작 중..."
    ollama serve &>/dev/null &
    OLLAMA_PID=$!
    # 서버 준비 대기 (최대 10초)
    for i in $(seq 1 20); do
        if curl -sf "${OLLAMA_URL}/api/tags" &>/dev/null; then
            break
        fi
        sleep 0.5
    done
    if ! curl -sf "${OLLAMA_URL}/api/tags" &>/dev/null; then
        fail "Ollama 서버 시작 실패. 별도 터미널에서 'ollama serve'를 실행하세요"
    fi
    ok "Ollama 서버 시작됨 (PID: ${OLLAMA_PID})"
else
    ok "Ollama 서버 실행 중"
fi

# ── 4. Teacher 모델 다운로드 ────────────────────────────────────────
info "Teacher 모델 확인 중... (${TEACHER_MODEL})"
if ollama list 2>/dev/null | grep -q "${TEACHER_MODEL%%:*}"; then
    ok "${TEACHER_MODEL} 준비됨"
else
    info "${TEACHER_MODEL} 다운로드 중... (첫 실행 시 5~10분 소요)"
    ollama pull "${TEACHER_MODEL}"
    ok "${TEACHER_MODEL} 다운로드 완료"
fi

# ── 완료 ────────────────────────────────────────────────────────────
echo ""
SHELL_NAME="$(basename "${SHELL:-/bin/sh}")"

echo -e "${GREEN}${BOLD}준비 완료!${NC}"
echo "────────────────────────────────────────"
echo ""

case "$SHELL_NAME" in
  fish)
    ACTIVATE_CMD="source .venv/bin/activate.fish"
    ;;
  csh|tcsh)
    ACTIVATE_CMD="source .venv/bin/activate.csh"
    ;;
  *)
    ACTIVATE_CMD="source .venv/bin/activate"
    ;;
esac

echo -e "  ${CYAN}# 가상환경 활성화${NC}"
echo "  $ACTIVATE_CMD"
echo ""
echo -e "  ${CYAN}# 사용법${NC}"
echo "  slm-factory init my-project          # 프로젝트 생성"
echo "  cp docs/*.pdf my-project/documents/  # 문서 추가"
echo "  slm-factory run                      # 전체 파이프라인"
echo "  slm-factory serve                    # 웹 채팅 서비스"
echo ""
