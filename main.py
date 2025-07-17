import streamlit as st
from utils.data_loader import load_document
from utils.text_splitter import split_documents
from utils.embedding import get_embedding_model
from utils.vector_db import create_vector_db, load_vector_db
from models.llm_manager import LLMManager
from utils.rubric_manager import display_rubric_editor
from utils.student_answer_loader import load_student_answers
from utils.retrieval import get_retriever, retrieve_documents, rerank_documents
from prompts.prompt_templates import get_grading_prompt
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import io
import time

st.set_page_config(layout="wide", page_title="RAG 기반 지리과 서답형 자동채점 플랫폼")

st.title("RAG 기반 지리과 서답형 자동채점 플랫폼")

@st.cache_resource
def get_llm_manager():
    return LLMManager()

llm_manager = get_llm_manager()

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

# Pydantic 모델 정의
class 채점결과(BaseModel):
    주요_채점_요소_1_점수: int = Field(description="주요 채점 요소 1에 대한 점수")
    주요_채점_요소_2_점수: int = Field(description="주요 채점 요소 2에 대한 점수")
    합산_점수: int = Field(description="모든 주요 채점 요소 점수의 합산")
    점수_판단_근거: dict = Field(description='각 주요 채점 요소별 점수 판단 근거 (예: {"주요_채점_요소_1": "근거 내용"})')

class 피드백(BaseModel):
    교과_내용_피드백: str = Field(description="교과 내용에 대한 구체적인 피드백")
    의사_응답_여부: bool = Field(description="학생 답안이 의사 응답(bluffing)인지 여부 (True/False)")
    의사_응답_설명: str = Field(description="의사 응답인 경우 설명, 아니면 빈 문자열")

class GradingOutput(BaseModel):
    채점결과: 채점결과
    피드백: 피드백

# 파서 인스턴스 생성
parser = PydanticOutputParser(pydantic_object=GradingOutput)

def process_student_answer(student_name, student_answer, retriever, llm, rubric, question_type, llm_manager):
    try:
        retrieved_docs = retrieve_documents(retriever, student_answer)
        reranked_docs = rerank_documents(retrieved_docs, student_answer)
        retrieved_docs_content = "\n\n".join([doc.page_content for doc in reranked_docs])
        
        # PydanticOutputParser의 format_instructions를 프롬프트에 추가
        format_instructions = parser.get_format_instructions()
        grading_prompt = get_grading_prompt(question_type, rubric, student_answer, retrieved_docs_content, format_instructions)
        
        llm_response_str = llm_manager.call_llm_with_retry(llm, grading_prompt)

        if not llm_response_str:
            return {"이름": student_name, "오류": "LLM 응답을 받지 못했습니다."}

        try:
            # PydanticOutputParser를 사용하여 응답 파싱
            parsed_output = parser.parse(llm_response_str)
            
            score_results = parsed_output.채점결과.model_dump()
            feedback_results = parsed_output.피드백.model_dump()
            
            # 점수 판단 근거 추출 및 최종 결과에 포함
            점수_판단_근거 = score_results.pop("점수_판단_근거", {})
            
            referenced_docs_info = [f"{doc.metadata.get('source', 'Unknown')} (p.{doc.metadata.get('page', 'N/A')})" for doc in reranked_docs]

            return {
                "이름": student_name,
                "답안": student_answer,
                "채점결과": score_results,
                "피드백": feedback_results,
                "점수_판단_근거": 점수_판단_근거, # 여기에 추가
                "참고문서": "; ".join(referenced_docs_info)
            }
        except Exception as e:
            return {"이름": student_name, "오류": f"LLM 응답 파싱 오류: {e}. 원본 응답: {llm_response_str[:200]}..."}

    except Exception as e:
        return {"이름": student_name, "오류": f"채점 중 오류 발생: {e}"}

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ 설정")
    st.header("1. LLM 모델 선택 및 데이터 준비")
    llm_provider = st.selectbox("LLM 모델 제공사 선택", ("GROQ", "OpenAI", "Google"), index=0)
    llm_model = ""
    if llm_provider == "GROQ":
        llm_model = st.selectbox("GROQ 모델 선택", ("gemma2-9b-it", "llama-3.1-8b-instant", "llama-3.1-70b-versatile"))
    elif llm_provider == "OpenAI":
        llm_model = st.selectbox("OpenAI 모델 선택", ("gpt-3.5-turbo", "gpt-4o"))
    elif llm_provider == "Google":
        llm_model = st.selectbox("Google Gemini 모델 선택", ("gemini-1.5-pro-latest", "gemini-1.5-flash-latest", "gemini-2.5-pro", "gemini-2.5-flash", "gemini-pro"))

    if llm_model:
        st.session_state['selected_llm'] = llm_manager.get_llm(llm_provider, llm_model)
        if st.session_state['selected_llm']:
            st.success(f"{llm_provider}의 {llm_model} 모델이 선택되었습니다.")
        else:
            st.error(f"{llm_provider}의 {llm_model} 모델 초기화에 실패했습니다. API 키를 확인해주세요.")

    st.subheader("Source Data 업로드")
    uploaded_file = st.file_uploader("문항 개발 시 사용된 원본 자료를 업로드하세요", type=["pdf", "xlsx", "xls", "docx", "doc", "txt"], key="file_uploader")

    if uploaded_file is not None:
        if st.session_state.get('uploaded_file_name') != uploaded_file.name:
            with st.spinner("파일 처리 중..."):
                documents = load_document(uploaded_file)
                if documents:
                    st.session_state['source_documents'] = documents
                    st.session_state['uploaded_file_name'] = uploaded_file.name
                    st.session_state.pop('chunks', None)
                    st.success(f"'{uploaded_file.name}'에서 {len(documents)}개의 문서를 로드했습니다.")
                else:
                    st.error(f"'{uploaded_file.name}' 파일 처리 중 오류가 발생했습니다.")
    
    st.subheader("Chunking 및 Embedding")
    if 'source_documents' in st.session_state:
        chunk_size = st.slider("청크 크기", 100, 2000, 1000, 50)
        chunk_overlap = st.slider("청크 오버랩", 0, 500, 200, 50)
        if st.button("청킹 실행"):
            with st.spinner("문서를 청크로 분할 중..."):
                st.session_state['chunks'] = split_documents(st.session_state['source_documents'], chunk_size, chunk_overlap)
        if 'chunks' in st.session_state:
            st.success(f"총 {len(st.session_state['chunks'])}개의 청크가 생성되었습니다.")
    else:
        st.info("Source Data가 업로드되면 Chunking 설정이 활성화됩니다.")

    st.header("2. 벡터 DB 구축")
    embeddings_model = get_embedding_model()
    if 'vector_db' in st.session_state and st.session_state['vector_db']:
        st.success("벡터 DB가 준비되었습니다.")
    elif 'chunks' in st.session_state:
        if st.button("벡터 DB 구축 또는 로드"):
            with st.spinner("벡터 DB 처리 중..."):
                vector_db = load_vector_db(embeddings_model)
                if vector_db is None:
                    vector_db = create_vector_db(st.session_state['chunks'], embeddings_model)
                if vector_db:
                    st.session_state['vector_db'] = vector_db
                    st.rerun()
                else:
                    st.error("벡터 DB 처리 중 오류가 발생했습니다.")
    else:
        st.info("청크가 생성되면 벡터 DB를 구축할 수 있습니다.")

# Main content
st.header("3. 평가 기준 및 학생 답안 입력")
question_type = st.radio("문항 유형 선택", ("단답형", "제한형", "확장형"))
st.info(f"선택된 문항 유형: {question_type}")
display_rubric_editor()

st.subheader("학생 답안 업로드")
uploaded_student_answers = st.file_uploader("학생 답안 Excel 파일을 업로드하세요", type=["xlsx", "xls"], key="student_answers_uploader")
if uploaded_student_answers:
    student_answers_df = load_student_answers(uploaded_student_answers)
    if student_answers_df is not None:
        st.session_state['student_answers_df'] = student_answers_df
        st.success("학생 답안이 성공적으로 로드되었습니다.")
        st.dataframe(student_answers_df)
    else:
        st.error("학생 답안 로드에 실패했습니다.")

st.header("4. RAG 기반 유사 문서 검색 및 채점")
if st.button("채점 시작"):
    if 'vector_db' not in st.session_state or not st.session_state['vector_db']:
        st.error("벡터 DB가 구축되지 않았습니다.")
    elif 'student_answers_df' not in st.session_state or st.session_state['student_answers_df'].empty:
        st.error("학생 답안이 로드되지 않았습니다.")
    elif 'final_rubric' not in st.session_state or not st.session_state['final_rubric']:
        st.error("평가 루브릭이 입력되지 않았습니다.")
    elif 'selected_llm' not in st.session_state or not st.session_state['selected_llm']:
        st.error("LLM 모델이 선택되지 않았습니다.")
    else:
        st.info("채점을 시작합니다...")
        start_time = time.time() # 채점 시작 시간 기록

        retriever = get_retriever(st.session_state['vector_db'])
        llm = st.session_state['selected_llm']
        rubric = st.session_state['final_rubric']
        
        graded_results = []
        progress_bar = st.progress(0)
        total_students = len(st.session_state['student_answers_df'])

        # 순차적으로 학생 답안 채점 (API 제한 및 스레드 오류 방지)
        for i, (index, row) in enumerate(st.session_state['student_answers_df'].iterrows()):
            student_name = row["이름"]
            student_answer = row["답안"]
            
            # 각 학생 답안 처리
            result = process_student_answer(student_name, student_answer, retriever, llm, rubric, question_type, llm_manager)
            graded_results.append(result)
            
            # 진행 상황 업데이트
            progress_bar.progress((i + 1) / total_students)

        st.session_state['graded_results'] = graded_results
        end_time = time.time() # 채점 종료 시간 기록
        elapsed_time = end_time - start_time
        st.success(f"모든 학생 답안 채점 완료! (총 소요 시간: {elapsed_time:.2f}초)")

st.header("5. 최종 결과")
if 'graded_results' in st.session_state and st.session_state['graded_results']:
    results_df = pd.DataFrame(st.session_state['graded_results'])
    st.subheader("채점 결과 요약")

    # 표시할 컬럼 목록 정의
    display_columns = ["이름", "답안", "채점결과", "피드백", "참고문서", "오류"]
    # 실제 DataFrame에 있는 컬럼만 필터링
    existing_columns = [col for col in display_columns if col in results_df.columns]
    
    # 화면 표시용 데이터프레임 생성 (복잡한 객체는 문자열로 변환)
    display_df = results_df[existing_columns].copy()
    for col in ["채점결과", "피드백"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].astype(str)

    # 필터링 및 변환된 컬럼으로 DataFrame 표시
    st.dataframe(display_df.fillna(""))

    st.subheader("개별 학생 채점 결과 상세")
    for index, row in results_df.iterrows():
        with st.expander(f"{row['이름']} 학생의 채점 결과"):
            # 오류가 있는지 먼저 확인
            if "오류" in row and pd.notna(row["오류"]):
                st.error(f"오류: {row['오류']}")
            else:
                # 성공적인 결과 표시
                if "답안" in row:
                    st.write(f"**학생 답안:** {row['답안']}")
                if "채점결과" in row and isinstance(row['채점결과'], dict):
                    st.write("**채점 결과:**")
                    for criterion, score in row['채점결과'].items():
                        st.write(f"- {criterion}: {score}")
                    if "점수_판단_근거" in row and isinstance(row['점수_판단_근거'], dict):
                        st.write("**점수 판단 근거:**")
                        for criterion, reason in row['점수_판단_근거'].items():
                            st.write(f"- {criterion}: {reason}")
                    st.write(f"**합산 점수:** {row['채점결과'].get('합산_점수', 'N/A')}")
                if "피드백" in row and isinstance(row['피드백'], dict):
                    st.write("**피드백:**")
                    st.write(f"- 교과 내용 피드백: {row['피드백'].get('교과_내용_피드백', 'N/A')}")
                    st.write(f"- 의사 응답 여부: {row['피드백'].get('의사_응답_여부', 'N/A')}")
                    if row['피드백'].get('의사_응답_여부', False):
                        st.write(f"  - 설명: {row['피드백'].get('의사_응답_설명', 'N/A')}")
                if "참고문서" in row:
                    st.write(f"**참고 문서:** {row['참고문서']}")

    # 대시보드 시각화 (오류 없는 결과만 필터링)
    valid_results = [res for res in st.session_state['graded_results'] if "오류" not in res or pd.isna(res.get("오류"))]
    if valid_results:
        st.subheader("대시보드 시각화")
        from utils.dashboard import display_dashboard
        display_dashboard(valid_results)

    st.subheader("Excel 다운로드")
    output = io.BytesIO()
    # Excel 내보내기를 위해 실제 존재하는 컬럼만 사용
    excel_df = results_df[existing_columns].copy()
    # Excel에서 보기 좋게 dict 타입을 문자열로 변환
    for col in ["채점결과", "피드백"]:
        if col in excel_df.columns:
            excel_df[col] = excel_df[col].astype(str)
            
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        excel_df.to_excel(writer, index=False, sheet_name='채점결과')
    excel_data = output.getvalue()
    st.download_button(
        label="채점 결과 Excel 다운로드",
        data=excel_data,
        file_name="graded_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.info("채점 결과가 없습니다. 채점을 시작해주세요.")

