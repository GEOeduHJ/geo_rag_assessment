# 2025년 7월 18일 금요일

## 오늘 진행 상황:
1. 문항 유형에 따른 프롬프트 연결 로직 수정
2. 백지도 문항 채점용 새 Python 컴포넌트 개발 (참고: `new_idea2 copy.md`)
    - LLava Image-to-Text 모델 통합:
        - `utils/map_item.py` 수정: LLava 모델 로딩 및 이미지 텍스트 추출 함수 구현
        - `main.py` 수정: `grade_map_question` 호출 시 `UploadedFile` 객체 전달
3. Langgraph 활용 LLM 응답 신뢰도 강화 방안 탐색

## 이전 작업 내용:
- LLM 모델 관리 및 프롬프트 템플릿 개발 (`models/llm_manager.py`, `prompts/prompt_templates.py`)
- 데이터 로딩, 임베딩, 텍스트 분할, 벡터 데이터베이스 관리 등 데이터 처리 유틸리티 구현 (`utils/data_loader.py`, `utils/embedding.py`, `utils/text_splitter.py`, `utils/vector_db.py`)
- 정보 검색 및 학생 답변 처리 로직 개발 (`utils/retrieval.py`, `utils/student_answer_loader.py`)
- 대시보드 및 채점 기준표 관리 기능 구현 (`utils/dashboard.py`, `utils/rubric_manager.py`)
- FAISS 기반 벡터 데이터베이스 인덱스 구축 및 관리 (`vector_db/faiss_index/`)