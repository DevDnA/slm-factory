# 자주 묻는 질문 (FAQ)

## 목차

- [설치 및 시작](#설치-및-시작)
  - [Q1. Python 버전이 맞는지 어떻게 확인하나요?](#q1-python-버전이-맞는지-어떻게-확인하나요)
  - [Q2. `slm-factory init` 후 다음에 뭘 해야 하나요?](#q2-slm-factory-init-후-다음에-뭘-해야-하나요)
- [Ollama 관련](#ollama-관련)
  - [Q3. "Ollama 연결 실패" 에러가 나옵니다](#q3-ollama-연결-실패-에러가-나옵니다)
  - [Q4. "모델을 찾을 수 없습니다" 에러가 나옵니다](#q4-모델을-찾을-수-없습니다-에러가-나옵니다)
  - [Q5. Ollama 대신 다른 LLM을 사용할 수 있나요?](#q5-ollama-대신-다른-llm을-사용할-수-있나요)
- [QA 생성 및 데이터](#qa-생성-및-데이터)
  - [Q6. QA 쌍이 0개 생성됩니다](#q6-qa-쌍이-0개-생성됩니다)
  - [Q7. 생성된 QA 품질이 낮습니다](#q7-생성된-qa-품질이-낮습니다)
  - [Q8. 한국어 문서인데 영어 QA가 생성됩니다](#q8-한국어-문서인데-영어-qa가-생성됩니다)
- [학습 관련](#학습-관련)
  - [Q9. "CUDA out of memory" 에러가 나옵니다](#q9-cuda-out-of-memory-에러가-나옵니다)
  - [Q10. GPU 없이 학습할 수 있나요?](#q10-gpu-없이-학습할-수-있나요)
  - [Q11. 학습을 중간에 멈추면 어떻게 되나요?](#q11-학습을-중간에-멈추면-어떻게-되나요)
- [내보내기 및 배포](#내보내기-및-배포)
  - [Q12. 학습된 모델을 어떻게 실행하나요?](#q12-학습된-모델을-어떻게-실행하나요)

---

## 설치 및 시작

### Q1. Python 버전이 맞는지 어떻게 확인하나요?

**증상**: 설치 중 Python 버전 관련 에러가 발생하거나, 패키지 설치가 실패합니다.

**원인**: slm-factory는 Python 3.11 이상을 요구합니다. 이전 버전에서는 일부 패키지가 호환되지 않습니다.

**해결 방법**:

1. 현재 Python 버전을 확인합니다:
   ```bash
   python --version
   ```
   또는
   ```bash
   python3 --version
   ```

2. 출력이 `Python 3.11.x` 이상이어야 합니다. 만약 3.10 이하라면 Python을 업그레이드하십시오:
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install python3.11

   # macOS (Homebrew)
   brew install python@3.11

   # Windows
   # python.org에서 3.11 이상 설치 프로그램 다운로드
   ```

3. 여러 Python 버전이 설치된 경우 명시적으로 지정합니다:
   ```bash
   python3.11 -m pip install -e ".[all]"
   ```

**참고**: 가상 환경(venv 또는 conda)을 사용하는 것을 권장합니다. 이를 통해 시스템 Python과 독립적으로 프로젝트를 관리할 수 있습니다.

---

### Q2. `slm-factory init` 후 다음에 뭘 해야 하나요?

**증상**: 프로젝트를 생성했지만 다음 단계가 불명확합니다.

**원인**: 초기 설정 후 문서 추가와 Ollama 준비가 필요합니다.

**해결 방법**:

1. 학습할 문서를 `documents/` 디렉토리에 추가합니다:
   ```bash
   cp /path/to/your/documents/*.pdf my-project/documents/
   ```
   지원 형식: PDF, HWPX, HTML, TXT, MD, DOCX

2. Ollama 서버를 실행합니다 (별도 터미널):
   ```bash
   ollama serve
   ```

3. Teacher 모델을 다운로드합니다:
   ```bash
   ollama pull qwen3:8b
   ```
   또는 다른 모델 (예: `llama3.1:8b`, `gemma2:9b`)

4. wizard 명령으로 파이프라인을 시작합니다:
   ```bash
   slm-factory tool wizard --config my-project/project.yaml
   ```

5. wizard가 단계별로 안내하며, 각 단계에서 확인을 요청합니다. 처음 사용자는 wizard만 실행하면 됩니다.

**참고**: wizard 대신 수동으로 실행하려면 `slm-factory run --config my-project/project.yaml` 명령을 사용하십시오.

---

## Ollama 관련

### Q3. "Ollama 연결 실패" 에러가 나옵니다

**증상**:
```
Error: Failed to connect to Ollama at http://localhost:11434
RuntimeError: Cannot connect to Ollama. Ollama가 실행 중인지 확인하세요.
```

**원인**: Ollama 서버가 실행되지 않았거나, 포트가 다르거나, 방화벽이 차단하고 있습니다.

**해결 방법**:

1. Ollama 서버가 실행 중인지 확인합니다:
   ```bash
   # 별도 터미널에서 실행
   ollama serve
   ```

2. Ollama가 정상적으로 응답하는지 테스트합니다:
   ```bash
   curl http://localhost:11434/api/tags
   ```
   정상이면 JSON 응답이 출력됩니다.

3. 포트가 다른 경우 `project.yaml`에서 수정합니다:
   ```yaml
   teacher:
     api_base: "http://localhost:11434"  # 포트 확인
   ```

4. 방화벽이 11434 포트를 차단하는지 확인합니다:
   ```bash
   # Linux
   sudo ufw status
   sudo ufw allow 11434

   # macOS
   # 시스템 환경설정 → 보안 및 개인 정보 보호 → 방화벽 옵션
   ```

**참고**: Ollama가 설치되지 않았다면 [ollama.com](https://ollama.com)에서 다운로드하십시오.

---

### Q4. "모델을 찾을 수 없습니다" 에러가 나옵니다

**증상**:
```
Error: Model 'qwen3:8b' not found
RuntimeError: Teacher 모델을 사용할 수 없습니다
```

**원인**: 지정한 모델이 Ollama에 다운로드되지 않았습니다.

**해결 방법**:

1. 사용 가능한 모델 목록을 확인합니다:
   ```bash
   ollama list
   ```

2. 모델이 목록에 없으면 다운로드합니다:
   ```bash
   ollama pull qwen3:8b
   ```
   다운로드 시간은 모델 크기에 따라 수 분에서 수십 분 소요됩니다.

3. `project.yaml`의 모델 이름이 정확한지 확인합니다:
   ```yaml
   teacher:
     model: "qwen3:8b"  # 대소문자 및 태그 확인
   ```

4. 모델 이름이 `ollama list` 출력과 정확히 일치하는지 확인합니다.

**참고**: 권장 Teacher 모델은 `qwen3:8b` (다국어), `llama3.1:8b` (영어), `gemma2:9b` (고품질)입니다.

---

### Q5. Ollama 대신 다른 LLM을 사용할 수 있나요?

**증상**: OpenAI API, vLLM, LiteLLM 등 다른 백엔드를 사용하고 싶습니다.

**원인**: 기본 설정은 Ollama이지만, OpenAI 호환 API를 지원합니다.

**해결 방법**:

1. **OpenAI API 사용**:
   ```yaml
   teacher:
     backend: "openai"
     model: "gpt-4o-mini"
     api_base: "https://api.openai.com"
     api_key: "sk-proj-..."  # 실제 API 키
     temperature: 0.3
   ```

2. **vLLM 서버 사용** (로컬 GPU 서버):
   ```yaml
   teacher:
     backend: "openai"
     model: "meta-llama/Llama-3.1-8B-Instruct"
     api_base: "http://localhost:8000/v1"
     api_key: "dummy"  # vLLM은 키가 필요 없지만 필드는 채워야 함
     temperature: 0.3
     max_concurrency: 16  # vLLM은 높은 동시성 지원
   ```

3. **LiteLLM 프록시 사용** (여러 LLM 통합):
   ```yaml
   teacher:
     backend: "openai"
     model: "claude-3-5-sonnet-20241022"
     api_base: "http://localhost:4000"
     api_key: "sk-1234"
     temperature: 0.3
   ```

**참고**: `backend: "openai"`로 설정하면 `/v1/chat/completions` 엔드포인트를 제공하는 모든 서비스와 호환됩니다.

---

## QA 생성 및 데이터

### Q6. QA 쌍이 0개 생성됩니다

**증상**:
```
Generated 0 QA pairs from 5 documents
Warning: No QA pairs generated
```

**원인**: 질문 카테고리가 비어있거나, 문서 내용이 비어있거나, Teacher 모델이 응답하지 않습니다.

**해결 방법**:

1. `project.yaml`의 `questions.categories`가 비어있지 않은지 확인합니다:
   ```yaml
   questions:
     categories:
       overview:
         - "What is the main purpose of this document?"
         - "Who are the target users?"
   ```

2. 문서가 실제로 텍스트를 포함하는지 확인합니다:
   ```bash
   slm-factory run --until parse --config project.yaml
   # 출력에서 "Parsed 5 documents" 확인
   ```

3. 파싱된 문서 내용을 확인합니다:
   ```bash
   cat output/parsed_documents.json
   # "content" 필드가 비어있지 않은지 확인
   ```

4. Teacher 모델이 정상 응답하는지 테스트합니다:
   ```bash
   ollama run qwen3:8b "Hello"
   # 응답이 출력되는지 확인
   ```

5. 로그를 확인하여 에러 메시지를 찾습니다:
   ```bash
   slm-factory -v generate --config project.yaml
   # -v 옵션으로 상세 로그 출력
   ```

**참고**: 문서가 암호화되어 있거나 스캔된 이미지 PDF인 경우 텍스트 추출이 실패할 수 있습니다.

---

### Q7. 생성된 QA 품질이 낮습니다

**증상**: QA 쌍이 생성되지만 답변이 부정확하거나, 너무 짧거나, 문서와 관련이 없습니다.

**원인**: Teacher 모델의 성능, 프롬프트 설정, 또는 검증 규칙이 부적절합니다.

**해결 방법**:

1. **품질 점수 평가 활성화** (Teacher LLM이 1~5점으로 평가):
   ```yaml
   scoring:
     enabled: true
     threshold: 3.5  # 3.5점 이상만 통과
     max_concurrency: 4
   ```

2. **Temperature 조절** (낮을수록 일관성 있는 답변):
   ```yaml
   teacher:
     temperature: 0.2  # 기본값 0.3에서 감소
   ```

3. **시스템 프롬프트 수정** (더 구체적인 지시):
   ```yaml
   questions:
     system_prompt: >
       당신은 제공된 문서를 기반으로 질문에 답변하는 전문가입니다.
       문서 내용에만 근거하여 정확하고 상세하게 답변하십시오.
       구체적인 숫자, 날짜, 이름을 포함하십시오.
       문서에 정보가 없으면 "문서에 해당 정보가 포함되어 있지 않습니다"라고 답변하십시오.
   ```

4. **더 강력한 Teacher 모델 사용**:
   ```yaml
   teacher:
     model: "gemma2:9b"  # 또는 "llama3.1:8b"
   ```

5. **의미적 검증 활성화** (답변이 문서에 근거하는지 확인):
   ```yaml
   validation:
     groundedness:
       enabled: true
       threshold: 0.35
   ```

**참고**: 품질 점수 평가와 의미적 검증은 추가 Teacher LLM 호출이 필요하므로 시간이 더 소요됩니다.

---

### Q8. 한국어 문서인데 영어 QA가 생성됩니다

**증상**: 한국어 문서를 파싱했지만 질문과 답변이 영어로 생성됩니다.

**원인**: 질문 카테고리와 시스템 프롬프트가 영어로 작성되어 있습니다.

**해결 방법**:

1. **언어 설정 변경**:
   ```yaml
   project:
     language: "ko"  # 기본값 "en"에서 변경
   ```

2. **한국어 질문 카테고리 사용**:
   ```yaml
   questions:
     categories:
       개요:
         - "이 문서의 주요 목적은 무엇입니까?"
         - "대상 사용자는 누구입니까?"
       기술:
         - "주요 기술 사양은 무엇입니까?"
         - "시스템 요구 사항은 무엇입니까?"
   ```

3. **한국어 시스템 프롬프트 작성**:
   ```yaml
   questions:
     system_prompt: >
       당신은 제공된 문서를 기반으로 질문에 답변하는 도움이 되는 어시스턴트입니다.
       문서 내용에만 근거하여 답변하십시오. 추측하거나 정보를 만들어내지 마십시오.
       간결하고 사실적으로 답변하십시오.
       문서에 관련 정보가 없으면 "문서에 해당 정보가 포함되어 있지 않습니다"라고 답변하십시오.
   ```

4. **한국어 거부 패턴 추가**:
   ```yaml
   validation:
     reject_patterns:
       - "(?i)i don't know"
       - "(?i)알 수 없"
       - "(?i)정보가 없"
       - "(?i)포함되어 있지 않"
   ```

**참고**: 한국어 지원이 우수한 Teacher 모델은 `qwen3:8b`입니다. `llama3.1:8b`는 한국어 지원이 제한적입니다.

---

## 학습 관련

### Q9. "CUDA out of memory" 에러가 나옵니다

**증상**:
```
RuntimeError: CUDA out of memory. Tried to allocate 2.50 GiB
torch.cuda.OutOfMemoryError
```

**원인**: GPU VRAM이 부족합니다. 모델 크기, 배치 크기, 시퀀스 길이가 GPU 메모리를 초과합니다.

**해결 방법**:

1. **배치 크기 감소** (가장 빠른 해결책):
   ```yaml
   training:
     batch_size: 2  # 기본값 4에서 감소
     gradient_accumulation_steps: 8  # 실제 배치 크기 유지 (2×8=16)
   ```

2. **양자화 활성화** (VRAM 사용량 50% 감소):
   ```yaml
   training:
     quantization:
       enabled: true
       bits: 4
   ```

3. **시퀀스 길이 감소**:
   ```yaml
   student:
     max_seq_length: 2048  # 기본값 4096에서 감소
   ```

4. **더 작은 Student 모델 선택**:
   ```yaml
   student:
     model: "google/gemma-3-1b-it"  # 1B 파라미터
   ```

5. **8GB VRAM 권장 설정** (RTX 3060, RTX 4060):
   ```yaml
   student:
     model: "google/gemma-3-1b-it"
     max_seq_length: 2048
   
   training:
     batch_size: 2
     gradient_accumulation_steps: 8
     quantization:
       enabled: true
       bits: 4
   ```

**참고**: 양자화를 사용하면 품질 저하는 거의 없지만, CPU 오프로드 시 학습 속도가 느려질 수 있습니다.

---

### Q10. GPU 없이 학습할 수 있나요?

**증상**: GPU가 없는 환경에서 학습을 시도합니다.

**원인**: CPU 학습은 가능하지만 매우 느립니다 (GPU 대비 10~100배 느림).

**해결 방법**:

1. **CPU 학습 설정** (bf16 비활성화):
   ```yaml
   training:
     bf16: false  # CPU는 bfloat16 미지원
     batch_size: 1
     gradient_accumulation_steps: 16
   ```

2. **매우 작은 모델 사용**:
   ```yaml
   student:
     model: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # 1.1B 파라미터
   ```

3. **에포크 수 감소**:
   ```yaml
   training:
     num_epochs: 5  # 기본값 20에서 감소
   ```

4. **권장: 클라우드 GPU 사용**
   - Google Colab (무료 T4 GPU): [colab.research.google.com](https://colab.research.google.com)
   - Kaggle Notebooks (무료 P100 GPU): [kaggle.com/code](https://www.kaggle.com/code)
   - Paperspace Gradient (유료): [paperspace.com](https://www.paperspace.com)

**참고**: CPU 학습은 테스트 목적으로만 권장하며, 실제 프로덕션 모델 학습에는 GPU가 필수입니다.

---

### Q11. 학습을 중간에 멈추면 어떻게 되나요?

**증상**: 학습 중 Ctrl+C로 중단하거나, 시스템이 종료되었습니다.

**원인**: 학습이 중단되었지만 체크포인트가 자동 저장되어 있습니다.

**해결 방법**:

1. **체크포인트 확인**:
   ```bash
   ls output/checkpoints/
   # checkpoint-100, checkpoint-200 등 디렉토리 확인
   ```

2. **wizard로 재실행** (자동으로 중단 지점부터 재개):
   ```bash
   slm-factory tool wizard --config project.yaml
   # wizard가 기존 파일을 감지하고 재개 여부를 묻습니다
   ```

3. **수동으로 재개** (특정 체크포인트에서):
   ```bash
   # 학습 데이터가 이미 있으면 학습만 재실행
   slm-factory train --config project.yaml --data output/training_data.jsonl
   ```

4. **조기 종료 설정 확인** (자동으로 최적 시점에 멈춤):
   ```yaml
   training:
     early_stopping:
       enabled: true
       patience: 3  # 3 에포크 동안 개선 없으면 중단
       threshold: 0.01
   ```

**참고**: `save_strategy: "epoch"`로 설정되어 있으면 매 에포크마다 체크포인트가 저장됩니다. 최종 모델은 `output/checkpoints/adapter/`에 저장됩니다.

---

## 내보내기 및 배포

### Q12. 학습된 모델을 어떻게 실행하나요?

**증상**: 학습이 완료되었지만 모델을 어떻게 사용하는지 모릅니다.

**원인**: 모델을 Ollama에 등록하거나 HuggingFace Transformers로 로드해야 합니다.

**해결 방법**:

1. **Ollama로 실행** (권장, 가장 간단):
   ```bash
   cd output/merged_model
   ollama create my-project-model -f Modelfile
   ollama run my-project-model
   ```

2. **대화 테스트**:
   ```bash
   ollama run my-project-model
   >>> 이 문서의 주요 목적은 무엇입니까?
   # 모델이 답변을 생성합니다
   ```

3. **HuggingFace Transformers로 로드** (Python 코드):
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer
   
   model_path = "./output/merged_model"
   tokenizer = AutoTokenizer.from_pretrained(model_path)
   model = AutoModelForCausalLM.from_pretrained(
       model_path,
       device_map="auto",
       torch_dtype="auto"
   )
   
   # 대화 생성
   messages = [
       {"role": "user", "content": "이 문서의 주요 목적은 무엇입니까?"}
   ]
   text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
   inputs = tokenizer(text, return_tensors="pt").to(model.device)
   outputs = model.generate(**inputs, max_new_tokens=256)
   print(tokenizer.decode(outputs[0], skip_special_tokens=True))
   ```

4. **LoRA 어댑터만 사용** (merge_lora: false인 경우):
   ```python
   from peft import PeftModel
   from transformers import AutoModelForCausalLM
   
   base_model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it")
   model = PeftModel.from_pretrained(base_model, "./output/checkpoints/adapter")
   ```

**참고**: Ollama 방식이 가장 간단하며, 채팅 인터페이스와 API 서버를 즉시 제공합니다.

---

## 관련 문서

- [README](../README.md) — 설치 및 빠른 시작
- [설정 레퍼런스](configuration.md) — project.yaml 상세 설명
- [아키텍처 가이드](architecture.md) — 내부 구조 이해
- [모듈별 상세 문서](modules.md) — 각 모듈의 API 및 확장 방법
