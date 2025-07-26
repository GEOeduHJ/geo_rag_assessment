flowchart TD
    classDef userInput fill:#cde4ff,stroke:#333,stroke-width:1px
    classDef process fill:#e2f0d9,stroke:#333,stroke-width:1px
    classDef decision fill:#fff2cc,stroke:#333,stroke-width:1px
    classDef database fill:#fbe5d6,stroke:#333,stroke-width:1px
    classDef model fill:#d9e1f2,stroke:#666,stroke-width:1px,stroke-dasharray: 3 3
    classDef trigger fill:#ffcccc,stroke:#b30000,stroke-width:2px
    classDef result fill:#d4fcd7,stroke:#333,stroke-width:1px

    subgraph "1. 설정 및 데이터 준비 (Sidebar)"
        direction LR
        A1["LLM 모델 선택<br>(GROQ, OpenAI, Google)"]:::userInput
        A2["Source Data 업로드<br>(PDF, DOCX 등)"]:::userInput
        A3("data_loader.py"):::process
        A4("text_splitter.py"):::process
        A5("embedding.py"):::process
        A5_MODEL["BM-K/ko-sroberta-multitask"]:::model
        A6[("벡터 DB 구축<br>(FAISS)")]:::database
        A7("vector_db.py"):::process
        
        A1 --> A2 --> A3 --> A4 --> A5 --> A6 --> A7
        A5 --- A5_MODEL
    end

    subgraph "2. 평가 기준 및 답안 입력 (Main)"
        direction LR
        B1{문항 유형 선택<br>(텍스트/백지도)}:::decision
        B2["평가 루브릭 입력"]:::userInput
        B3["학생 답안 업로드<br>(Excel)"]:::userInput
        B4["학생 백지도 업로드<br>(Image)"]:::userInput

        B1 --> B2
        B1 -- 텍스트 --> B3
        B1 -- 백지도 --> B4
    end

    C1["채점 시작"]:::trigger

    subgraph "3. RAG 기반 자동 채점"
        direction TD
        C2{채점 로직 분기}:::decision
        
        subgraph "텍스트 기반 채점"
            direction LR
            T1("retrieval.py<br><b>Retrieve</b>"):::process
            T2("retrieval.py<br><b>Rerank</b>"):::process
            T2_MODEL["cross-encoder/ms-marco-MiniLM-L-6-v2"]:::model
            T3("prompt_templates.py<br><b>Prompt 생성</b>"):::process
            T4("llm_manager.py<br><b>LLM 호출</b>"):::process
            T4_MODEL["선택된 LLM<br>(gemma, gpt, gemini 등)"]:::model
            T5("PydanticOutputParser<br><b>결과 파싱</b>"):::process

            C2 -- 텍스트 --> T1 --> T2 --> T3 --> T4 --> T5
            T2 --- T2_MODEL
            T4 --- T4_MODEL
        end

        subgraph "백지도 기반 채점"
            direction LR
            M1("map_item.py<br><b>Image-to-Text</b>"):::process
            M1_MODEL["llava-hf/llava-1.5-7b-hf"]:::model
            M2("retrieval.py<br><b>Retrieve</b>"):::process
            M3("retrieval.py<br><b>Rerank</b>"):::process
            M3_MODEL["cross-encoder/ms-marco-MiniLM-L-6-v2"]:::model
            M4("map_item.py<br><b>Prompt 생성</b>"):::process
            M5("llm_manager.py<br><b>LLM 호출</b>"):::process
            M5_MODEL["선택된 LLM<br>(gemma, gpt, gemini 등)"]:::model
            
            C2 -- 백지도 --> M1 --> M2 --> M3 --> M4 --> M5 --> T5
            M1 --- M1_MODEL
            M3 --- M3_MODEL
            M5 --- M5_MODEL
        end
    end

    subgraph "#4. 결과 표시 및 분석"
        direction LR
        R1["최종 결과 집계<br>(DataFrame)"]:::result
        R2["결과 요약 테이블"]:::result
        R3["대시보드 시각화<br>(dashboard.py)"]:::result
        R4["Excel 다운로드"]:::result

        T5 --> R1
        R1 --> R2 & R3 & R4
    end

    A7 --> C1
    B2 --> C1
    B3 --> C1
    B4 --> C1
    C1 --> C2