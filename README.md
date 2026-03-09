# slm-factory

도메인 문서를 학습하여 특화된 소형 언어모델(SLM)을 자동 생성하는 Teacher-Student 지식 증류 프레임워크

## 빠른 시작

```bash
git clone https://github.com/DevDnA/slm-factory.git
cd slm-factory
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"

ollama serve                # 별도 터미널
ollama pull qwen3:8b        # Teacher 모델 (8GB VRAM) 또는 qwen3.5:9b (24GB+)

slm-factory init my-project
cp /path/to/documents/*.pdf my-project/documents/
slm-factory tool wizard --config my-project/project.yaml
```

## 파이프라인

```
문서 (PDF/HWPX/HTML/TXT/DOCX)
  → Parse → [Ontology] → Generate → Validate → [Score / Augment / Analyze] → Convert → Train → Export
  → 도메인 특화 SLM 완성 (Ollama 즉시 서빙)
```

## 주요 기능

- **다중 형식 파싱** — PDF, HWPX(한글), HTML, TXT/MD, DOCX
- **유연한 Teacher LLM** — Ollama(로컬) 또는 OpenAI 호환 API
- **QA 검증 + 품질 평가** — 규칙/임베딩 필터링, LLM 1~5점 평가
- **데이터 증강** — 질문 패러프레이즈로 학습 데이터 확장
- **온톨로지 추출** — 문서에서 엔티티·관계를 자동 추출하여 QA 생성 품질 향상
- **LoRA 파인튜닝** — 효율적 학습 + 조기 종료
- **원클릭 Ollama 배포** — Modelfile 자동 생성
- **자동 진화** — `tool evolve` 한 번으로 증분→학습→품질게이트→배포
- **TUI** — QA 리뷰, 파이프라인 대시보드

## 문서

> **[devdna.github.io/slm-factory](https://devdna.github.io/slm-factory/)**

| 문서 | 내용 |
|------|------|
| [사용 가이드](https://devdna.github.io/slm-factory/guide.html) | 설치, 튜토리얼, 트러블슈팅 |
| [기술 확장 가이드](https://devdna.github.io/slm-factory/integration-guide.html) | RAG·온톨로지 기술 조합 전략, 연동 방법, 장단점 |
| [빠른 참조](https://devdna.github.io/slm-factory/quick-reference.html) | 명령어 치트시트 |
| [CLI 레퍼런스](https://devdna.github.io/slm-factory/cli-reference.html) | 전체 명령어 옵션 |
| [설정 레퍼런스](https://devdna.github.io/slm-factory/configuration.html) | project.yaml 전체 설정 |
| [아키텍처](https://devdna.github.io/slm-factory/architecture.html) | 설계 철학, 패턴, 데이터 흐름 |
| [개발 가이드](https://devdna.github.io/slm-factory/development.html) | 모듈 확장, 기여 방법 |

## 시스템 요구사항

- **Python** 3.11+
- **GPU** — NVIDIA CUDA (8GB+) / Apple Silicon (MPS) / CPU 폴백
- **Ollama** — [ollama.com](https://ollama.com)

## 라이선스

추후 결정
