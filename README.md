# Legal Agent System with KMMLU Evaluation

KMMLU Criminal-Law 테스트셋을 활용한 법률 Agent System 평가 과제입니다.

## 프로젝트 구조

```
.
├── Dockerfile              # Docker 컨테이너 설정
├── README.md              # 프로젝트 문서
├── data/                  # 데이터 디렉토리
│   ├── processed_texts.json   # 전처리된 텍스트
│   ├── raw_pdfs/         # 원본 PDF 파일들
│   └── vectors/          # 벡터 저장소
├── docker-compose.yml     # Docker Compose 설정
├── output/               # 출력 파일 디렉토리
│   ├── batch_input.jsonl  # OpenAI Batch API 입력
│   ├── batch_output.jsonl # OpenAI Batch API 출력
│   └── kmmlu_score_optimized.txt # 평가 결과
├── pyproject.toml        # Poetry 프로젝트 설정
├── poetry.lock          # Poetry 의존성 잠금 파일
├── run.sh               # 실행 스크립트
└── src/                 # 소스 코드
    ├── agent.py         # Agent 시스템 구현
    ├── preprocess_pdf.py # PDF 전처리 모듈
    └── retriever.py     # 검색 모듈
```

## 실행 방법

1. 환경 변수 설정
```bash
# .env 파일 생성
OPENAI_API_KEY=your_api_key_here
```

2. Docker 실행
```bash
docker-compose up --build
```
run.sh 스크립트는 다음 작업을 순차적으로 수행합니다:
1. PDF 데이터 전처리
2. 벡터 데이터베이스 구축
3. Agent 시스템 초기화
4. KMMLU Criminal-Law 평가 실행

## 시스템 구성

1. PDF 전처리 (preprocess_pdf.py)
- PDF 문서 텍스트 추출 및 정제
- 데이터 출처 : https://www.law.go.kr/lsInfoP.do?lsiSeq=222447#0000
- PDF OCR 및 정규표현식 기반 전처리
- JSON 형식 저장

2. 검색 시스템 (retriever.py)
- text-embedding-small 모델 기반 임베딩
- FAISS 벡터 검색
- 관련 문서 검색

3. Agent 시스템 (agent.py)
- GPT4-mini 기반 추론
- KMMLU 평가 구현

## 사용 모델
- LLM: GPT4-mini
- Embedding: text-embedding-small

## 평가 결과
평가 결과는 `output/kmmlu_score_optimized.txt`에서 확인 가능합니다.