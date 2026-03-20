# Local RAG System

🚀 완전 로컬에서 동작하는 CPU 친화형 RAG (Retrieval Augmented Generation) 시스템

이 프로젝트는 `GPU 없어도 일단 RAG는 돌아가야 한다`는 꽤 현실적인 목표에서 시작했습니다.
거창한 서버보다 먼저, 노트북 CPU에서 문서 넣고 검색해 보고 질문까지 이어지는 흐름을 빠르게 검증하는 데 초점을 맞췄습니다.

또한 `llmfit`에서 영감을 받아, 현재 시스템의 RAM/GPU/런타임 상태를 보고 `지금 이 PC에서 제일 덜 무리하는 LLM`을 추천하도록 구성했습니다.
한마디로, 모델 선택도 감으로 하지 말고 PC에게 먼저 물어보자는 방향입니다.

## 😄 왜 만들었나

- GPU가 없거나 약해도 `RAG 구조 자체`는 충분히 테스트할 수 있어야 했습니다.
- 문서 처리, 검색, 랭킹, 질의 흐름을 먼저 CPU에서 검증하고, 나중에 더 센 장비로 올리는 편이 개발 속도가 좋았습니다.
- 로컬 환경마다 사양 차이가 커서, `내 PC에서는 뭐가 맞지?`를 매번 수동으로 고민하기 귀찮았습니다.
- 그래서 `llmfit` 스타일로 하드웨어를 보고 LLM을 추천하게 했습니다. 개발자가 밤새 고민하는 대신, 일단 기계가 자기 사정을 말하게 한 셈입니다.

## ✨ 주요 기능

| 기능 | 설명 |
|------|------|
| **CPU 테스트 친화** | GPU가 없어도 문서 인덱싱, 검색, 질의 흐름을 로컬에서 먼저 검증 가능 |
| **하이브리드 검색** | BM25 (키워드) + Vector (의미) 검색 결합 |
| **혼합 랭킹** | RRF + 점수 정규화 + 동적 가중치 + 양쪽 매칭 부스트로 검색 결과 통합 |
| **한글 최적화** | 조사 분리, 문장 경계 인식, 불용어 처리 |
| **다양한 문서** | PDF, Markdown, Word, UTF-8 텍스트/코드 파일 지원 |
| **llmfit 스타일 추천** | 실행 PC의 RAM/GPU/런타임을 보고 현재 시스템에 맞는 LLM 제안 |
| **다중 런타임** | Ollama 또는 OpenAI-compatible 로컬 서버(LM Studio 등) 자동 감지 |
| **완전 로컬** | 인터넷 없이 로컬 모델로 질의 가능 |
| **소스 동기화 인덱싱** | 같은 파일/소스를 다시 넣으면 변경 여부를 감지해 스킵하거나 기존 소스를 다시 인덱싱 |

참고:
기본 디렉터리 인덱싱 대상은 `.txt`, `.md`, `.pdf`, `.docx`, `.py`, `.js`, `.ts`, `.json`, `.yaml`, `.yml` 입니다.
텍스트 계열 파일은 현재 `UTF-8` 인코딩 기준으로 읽습니다.

## ⚡ 성능

| 작업 | 속도 | 비고 |
|------|------|------|
| 문서 인덱싱 | ~15-20 docs/s | 모델 로딩 후 |
| 벡터 검색 | < 10ms | 10K 문서 기준 |
| 하이브리드 검색 | < 50ms | BM25 + Vector |
| 캐시 적중 | < 1ms | LRU 캐시 |
| **검색 정확도** | **100%** | 1위 문서 정확도 |

## 📦 설치

```bash
# 1. 의존성 설치
# Ollama 경로와 LM Studio/vLLM 같은 OpenAI-compatible 경로가 함께 설치됩니다.
pip install -r requirements.txt

# 2. 로컬 LLM 런타임 준비
# 선택 A: Ollama
winget install Ollama.Ollama

# 선택 B: LM Studio
# https://lmstudio.ai/ 설치 후 Local Server 활성화

# 3. 권장 모델 다운로드 예시
# 16GB Windows 노트북 기본 추천 예시
ollama pull qwen3:4b
```

참고:
`requirements.txt`에는 `ollama`와 `openai` SDK가 함께 포함되어 있어 `Ollama`, `LM Studio`, `vLLM` 같은 OpenAI-compatible 로컬 서버를 같은 코드 경로로 붙일 수 있습니다.

## 🚀 빠른 시작

아래 순서는 `CPU에서도 일단 돌아가는지`를 가장 빨리 확인하는 흐름입니다.
처음부터 큰 모델과 복잡한 서버를 붙이기보다, 현재 PC에서 검색과 질의 파이프라인이 먼저 살아 있는지 확인하는 쪽이 훨씬 덜 지칩니다.

### 데모 실행
```bash
python main.py
```

### 현재 PC 진단

먼저 이 두 명령으로 `내 PC가 지금 어떤 모델까지 무난한지` 확인합니다.

```bash
python cli.py doctor
python cli.py models
```

### CLI 사용

가장 추천하는 첫 실험은 `작은 문서 하나 인덱싱 -> 검색 -> 질의` 순서입니다.

```bash
# 인덱싱
python cli.py index ./documents
python cli.py index ./file.pdf

# 특정 소스 삭제
python cli.py remove-source ./file.pdf

# 검색
python cli.py search "인공지능이란?"
python cli.py search "machine learning" --mode hybrid

# 질의 (LLM 응답)
python cli.py query "RAG의 장점은?"

# 대화형 채팅
python cli.py chat
```

### Python API

CLI로 흐름이 확인되면, 그 다음에 Python API로 붙이는 편이 디버깅이 쉽습니다.

```python
from config import RAGConfig
from rag_engine import RAGEngine

# 엔진 초기화
engine = RAGEngine(RAGConfig())

# 문서 인덱싱
engine.index_text("인공지능은 미래 기술입니다.", source="AI문서")

# 검색 (관련성 점수 0-100%)
results = engine.search("인공지능", top_k=3)
for r in results:
    print(f"[{r.relevance_score:.0%}] {r.document.content[:50]}...")

# LLM 질의
response = engine.query("인공지능이란?")
print(response.answer)

# Upsert (있으면 업데이트, 없으면 삽입)
engine.upsert_text("업데이트된 내용", source="AI문서")
```

## 🏗️ 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                      User Query                         │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│          Query Preprocessing (한글 정규화/조사 분리)      │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│                   Hybrid Search                         │
│  ┌─────────────────┐       ┌─────────────────────────┐  │
│  │   BM25 Search   │       │     Vector Search       │  │
│  │  (SQLite FTS5)  │       │  (FAISS + Embeddings)   │  │
│  └────────┬────────┘       └───────────┬─────────────┘  │
│           └──────────┬─────────────────┘                │
│                      ▼                                  │
│  ┌─────────────────────────────────────────────────┐    │
│  │  RRF + 점수 정규화 + 동적 가중치 + 양쪽 매칭 부스트  │    │
│  └─────────────────────────────────────────────────┘    │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│      Local LLM Runtime (Auto-selected by hardware)     │
│   Ollama / LM Studio / vLLM(OpenAI-compatible)         │
└─────────────────────────────────────────────────────────┘
```

## 🔀 런타임 선택 가이드

`local_rag`는 LLM 호출 계층을 `런타임 독립적`으로 구성해 두었기 때문에, 아래 3가지 경로를 모두 수용할 수 있습니다.
즉, 애플리케이션 코드는 최대한 그대로 두고, 실제 추론 엔진만 현재 PC 사정에 맞춰 바꿔 끼울 수 있습니다.

| 런타임 | 권장 환경 | 장점 | 주의점 |
|--------|-----------|------|--------|
| `Ollama` | Windows 로컬 PC, 단일 사용자, 빠른 시작 | 설치/모델 관리가 단순하고 로컬 개발에 가장 실용적 | 고처리량 서버 성능은 `vLLM`보다 제한적 |
| `LM Studio` | GUI 중심 로컬 테스트, OpenAI-compatible API 필요 시 | 로컬 서버를 켜면 바로 API 연동 가능 | 모델/서버 설정을 직접 맞춰야 함 |
| `vLLM` | Linux/WSL + 강한 GPU, 동시 요청 많은 서버 | 높은 처리량, continuous batching, OpenAI-compatible API | Windows native 기본 지원이 아니고 운영 복잡도가 높음 |

### 현재 PC 기준 권장 결론

- 현재 PC(`16GB RAM + Intel Iris Xe + Windows`)에서는 `Ollama`가 기본 선택으로 가장 적합합니다.
- 이유는 단순합니다. 이 환경에서는 `빨리 켜지고, 덜 복잡하고, CPU/내장 GPU 기준으로도 가장 현실적`이기 때문입니다.
- `vLLM`은 분명 강력하지만, 이 노트북급 환경에서는 성능 이점보다 설치·호환·운영 부담이 더 크게 느껴질 가능성이 높습니다.
- 특히 `vLLM`은 GPU 경로에서 사실상 `Windows native 기본값`이라기보다 `WSL/Linux + 더 강한 GPU` 쪽에 더 잘 맞습니다.

### 언제 vLLM을 고려할까?

- `WSL2` 또는 `Linux`에서 운영할 수 있을 때
- `Intel Arc`, `NVIDIA`, `AMD` 등 더 강한 추론용 GPU를 사용할 때
- 여러 요청을 동시에 처리하는 로컬 서버/사내 서버 구성이 필요할 때
- OpenAI-compatible API를 통해 다른 앱이나 에이전트 프레임워크와 연결할 때

정리하면, `지금 당장 내 노트북에서 테스트`가 목표면 `Ollama`, `로컬 GUI 실험`이면 `LM Studio`, `제대로 서빙`하려면 `vLLM` 쪽입니다.

### local_rag에서 vLLM을 붙이는 방법

이 프로젝트는 이미 `OpenAI-compatible` 경로를 지원하므로, `vLLM`을 직접 내장할 필요는 없습니다.

1. `vLLM` 서버를 별도로 실행합니다.
2. `openai_base_url`을 해당 서버 주소로 설정합니다.
3. `llm_provider="openai-compatible"`로 두면 `local_rag`가 같은 인터페이스로 연결합니다.

예시:

```bash
# Linux/WSL 예시
vllm serve Qwen/Qwen3-4B \
    --dtype auto \
    --api-key local-dev-key
```

```python
from config import RAGConfig

config = RAGConfig(
        llm_provider="openai-compatible",
        llm_model="Qwen/Qwen3-4B",
        openai_base_url="http://127.0.0.1:8000/v1",
        openai_api_key="local-dev-key",
)
```

요약하면:

- `지금 PC에서 가장 무난한 기본값`: `Ollama`
- `GUI로 가볍게 붙여 보는 로컬 테스트`: `LM Studio`
- `성능 욕심이 생긴 뒤의 API 서버`: `vLLM`

## 🔧 2026 업데이트 포인트

- `llmfit` 스타일의 하드웨어 탐지 추가
- 실행 중인 로컬 런타임 자동 탐지 (`Ollama`, `OpenAI-compatible`)
- 고정 모델(`qwen2.5`) 제거, PC 사양에 맞는 기본 모델 자동 선택
- 임베딩 차원 하드코딩 제거, 선택 모델군에 맞게 자동 조정
- 2026 기준 소형/중형 로컬 모델군 반영 (`Qwen3`, `Gemma 3n`, `Phi-4`)
- 동일 소스 재인덱싱 시 변경 없는 파일은 자동 스킵, 변경된 파일은 BM25/FAISS를 함께 동기화
- 특정 소스 삭제 후 벡터 인덱스까지 재동기화하는 유지보수 명령 추가
- source fingerprint에 chunk 설정 버전을 포함해, 청크 정책 변경 시 잘못된 스킵이 발생하지 않도록 보완

## 🔧 설정

기본값은 `CPU에서도 먼저 테스트 가능한 흐름`을 기준으로 잡아 두었습니다.
즉, 처음부터 가장 큰 모델을 밀어 넣기보다 `현재 PC에서 무난하게 돌아가는 조합`을 먼저 쓰는 쪽에 가깝습니다.

### 기본 설정 (`config.py`)

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `embedding_model` | `intfloat/multilingual-e5-small` | 경량 다국어 임베딩 |
| `llm_model` | 자동 선택 | PC/런타임 적합 모델 |
| `llm_provider` | 자동 선택 | `ollama` 또는 `openai-compatible` |
| `chunk_size` | 512 | 청크 크기 |
| `top_k` | 5 | 검색 결과 수 |
| `bm25_weight` | 0.5 | BM25 가중치 |
| `vector_weight` | 0.5 | Vector 가중치 |

### 성능 설정

속도와 메모리 사용량이 걱정되면, 아래 항목부터 보는 편이 가장 효율적입니다.

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `faiss_index_type` | `flat` | 인덱스 타입 (`flat`/`ivf`/`hnsw`) |
| `cache_embeddings` | `True` | 임베딩 캐시 |
| `sqlite_wal_mode` | `True` | SQLite WAL 모드 |
| `dynamic_weights` | `True` | 동적 가중치 조정 |

### FAISS 인덱스 선택

대충 고르는 것 같아 보여도, 여기서는 문서 수에 따라 체감이 제법 달라집니다.
작게 시작할 때는 `flat`, 규모가 커지면 `ivf`나 `hnsw`로 넘어가는 편이 안전합니다.

| 인덱스 | 문서 수 | 특징 |
|--------|---------|------|
| `flat` | < 10K | 정확한 검색 |
| `ivf` | 10K - 1M | 빠른 근사 검색 |
| `hnsw` | 10K - 10M | 그래프 기반, 높은 recall |

## 🇰🇷 한글 최적화

이 프로젝트는 한글 문서를 `영어 파이프라인에 억지로 태우는 방식`보다, 한국어 문장 구조와 조사 특성을 어느 정도 존중하는 쪽으로 설계했습니다.
그래서 검색 정확도는 벡터 모델 하나보다, 전처리와 랭킹 보정에서 더 많이 올라가는 편입니다.

### 주요 기능

| 기능 | 설명 |
|------|------|
| **Unicode NFC 정규화** | 자모 분리 텍스트 결합 |
| **조사 분리** | 은/는/이/가/을/를 등 30+ 조사 패턴 |
| **불용어 제거** | 한글/영어 불용어 사전 |
| **문장 경계 인식** | 종결어미 (다/요/습니다) 기반 분리 |
| **동적 가중치** | 질문형 쿼리는 Vector 가중치 ↑ |
| **키워드 추출** | 빈도 기반 핵심어 추출 |

### 2026 권장 LLM 모델 티어

| 모델 | 크기 | 한글 성능 | 설치 |
|------|------|----------|------|
| `qwen3:4b` | 4B | ⭐⭐⭐⭐ | `ollama pull qwen3:4b` |
| `qwen3:8b` | 8B | ⭐⭐⭐⭐⭐ | `ollama pull qwen3:8b` |
| `qwen3:14b` | 14B | ⭐⭐⭐⭐⭐ | `ollama pull qwen3:14b` |
| `phi4-mini` | 3.8B | ⭐⭐⭐ | `ollama pull phi4-mini` |
| `gemma3n:e2b` | 2B급 | ⭐⭐⭐ | `ollama pull gemma3n:e2b` |

### 현재 PC 기준 예상 기본값

- `16GB RAM + Intel Iris Xe + Ollama 미설치` 환경에서는 `qwen3:4b` 급이 현실적인 기본값입니다.
- Ollama가 없으면 `LM Studio` 로컬 서버(`http://127.0.0.1:1234/v1`)도 자동 감지합니다.
- 더 큰 모델은 설치되어 있어도 응답 속도와 메모리 압박이 커질 수 있으므로, `일단 돌아가는가`보다 `오래 버티는가`가 더 중요해지는 순간부터는 `doctor` 결과를 먼저 보는 편이 낫습니다.

## 📂 프로젝트 구조

```
local_rag/
├── config.py              # 설정
├── document_processor.py  # 문서 처리 + 한글 토크나이저
├── embedding_model.py     # 임베딩 (캐싱, 전처리)
├── vector_store.py        # FAISS 벡터 스토어
├── bm25_store.py          # SQLite FTS5 + BM25
├── hybrid_search.py       # 하이브리드 검색 + RRF 기반 혼합 랭킹
├── model_selection.py     # 하드웨어/런타임 탐지 + 모델 선택
├── llm_client.py          # 런타임 적응형 로컬 LLM 클라이언트
├── rag_engine.py          # 메인 RAG 엔진
├── cli.py                 # CLI 인터페이스
├── main.py                # 데모
└── rag_store/             # 데이터 (자동 생성)
```

## 🛠️ 최적화 기법

여기서 말하는 최적화는 벤치마크 숫자만 예쁘게 만드는 종류보다는, `로컬 PC에서 덜 버벅이고 덜 다시 하게 만드는` 쪽에 가깝습니다.
즉, CPU 환경에서도 체감 차이가 나는 것들만 남겼습니다.

### 정확도 향상
- **쿼리 전처리**: 한글/영어 자동 감지 및 프리픽스 추가
- **점수 정규화**: Min-Max 정규화로 공정한 비교
- **양쪽 매칭 부스트**: BM25 & Vector 모두에서 발견 시 1.2배
- **동적 가중치**: 쿼리 특성에 따른 자동 조정

### 성능 최적화
- **Singleton 패턴**: 모델 중복 로딩 방지
- **LRU 캐시**: 동일 쿼리 < 1ms 응답
- **배치 처리**: 64개 단위 병렬 임베딩
- **WAL 모드**: SQLite 동시 읽기/쓰기

## 🧪 권장 점검 순서

```bash
# 1. 현재 PC 진단
python cli.py doctor

# 2. 로컬 런타임/모델 확인
python cli.py models

# 3. 문서 인덱싱
python cli.py index ./documents

# 같은 소스를 다시 넣었을 때 변경이 없으면 자동 스킵
python cli.py index ./documents

# 4. 질의
python cli.py query "RAG의 장점은?"
```

### 회귀 테스트

```bash
# CLI 회귀 점검
python run_cli_regression.py

# 단위 회귀 테스트
python -m unittest discover -s tests -p "test_*.py" -v
```

- 포함 범위: `document_processor`, `model_selection`, `hybrid_search`

## 🔗 관련 프로젝트

아래 프로젝트는 접근 방식이 꽤 다르기 때문에, `내가 지금 만들려는 게 빠른 로컬 검색인지, 고정밀 문서 추론인지` 비교할 때 참고하기 좋습니다.

### [PageIndex](https://github.com/VectifyAI/PageIndex)
벡터 없는 추론 기반 RAG
- No Vector DB, No Chunking
- 문서를 트리 구조로 인덱싱
- FinanceBench 98.7% 정확도

### 비교

한 줄로 요약하면, 이 프로젝트는 `로컬에서 빨리 돌려 보고 바로 만져 보는 쪽`, PageIndex는 `복잡한 문서를 더 깊게 추론하는 쪽`에 더 가깝습니다.

| 기준 | Local RAG (이 프로젝트) | PageIndex |
|------|------------------------|-----------|
| 환경 | 완전 로컬 | OpenAI API |
| 속도 | ⚡ 빠름 (< 50ms) | 느림 (LLM 추론) |
| 비용 | 무료 | API 비용 |
| 정확도 | 좋음 | 최고 (전문 문서) |
| 적합 | 일반 문서, 빠른 검색 | 복잡한 전문 문서 |

## 📝 라이선스

MIT License

## 📚 참고 자료

- [Hacker News: How are you doing RAG locally?](https://news.ycombinator.com/item?id=46616529)
- [FAISS Documentation](https://faiss.ai/)
- [SQLite FTS5](https://www.sqlite.org/fts5.html)
- [Ollama](https://ollama.ai/)
- [Sentence Transformers](https://www.sbert.net/)
- [vllm](https://github.com/vllm-project/vllm)
- [llmfit](https://github.com/AlexsJones/llmfit)
