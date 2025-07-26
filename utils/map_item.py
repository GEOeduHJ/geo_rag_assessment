import json
from typing import List, Dict, Any
from langchain_core.prompts import PromptTemplate
from models.llm_manager import LLMManager
from utils.retrieval import retrieve_documents, rerank_documents
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
import streamlit as st # Streamlit의 cache_resource를 사용하기 위해 임포트
from PIL import Image # 이미지 처리를 위해 PIL 임포트
from transformers import pipeline, AutoProcessor, AutoModelForCausalLM, AutoModelForVision2Seq
import io # BytesIO를 사용하기 위해 임포트




# LLava 모델 로딩 및 캐싱
@st.cache_resource
def load_smolvlm_model():
    model_id = "HuggingFaceTB/SmolVLM-500M-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForVision2Seq.from_pretrained(model_id)
    return processor, model

# 이미지에서 텍스트를 추출하는 함수
def extract_text_from_image(uploaded_image_file: Any) -> Dict[str, List[str]]:
    processor, model = load_smolvlm_model()
    
    # Streamlit UploadedFile 객체에서 이미지 데이터 읽기
    image_bytes = uploaded_image_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # LLava 모델을 위한 프롬프트
    prompt = """<image>
This map was created by a student for a geography performance assessment. Infer the student's intention in creating this map."""
    
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    
    # 모델 추론
    generate_ids = model.generate(**inputs, max_new_tokens=100)
    generated_text = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
    
    # LLava 모델의 출력에서 텍스트 부분만 파싱 (모델 출력 형식에 따라 조정 필요)
    # 예시: "USER: Extract all text from this image. ASSISTANT: [extracted text]"
    # 여기서는 간단하게 ASSISTANT 이후의 텍스트를 추출합니다.
    if "ASSISTANT:" in generated_text:
        extracted_content = generated_text.split("ASSISTANT:")[1].strip()
    else:
        extracted_content = generated_text.strip()

    # 추출된 텍스트를 리스트 형태로 반환 (단어 또는 구 단위로 분리)
    # 실제 사용 시에는 더 정교한 파싱 로직이 필요할 수 있습니다.
    recognized_texts = [text.strip() for text in extracted_content.split(',') if text.strip()]
    
    return {"recognized_texts": recognized_texts}

def get_map_grading_prompt(recognized_texts: List[str], rubric: List[Dict], format_instructions: str) -> str:
    """
    백지도 문항 채점을 위한 프롬프트를 생성합니다.
    """
    rubric_str = ""
    for i, item in enumerate(rubric):
        rubric_str += f"- 주요 채점 요소 {i+1}: {item['main_criterion']}\n"
        for j, sub_item in enumerate(item['sub_criteria']):
            rubric_str += f"  - 세부 내용 {j+1} (점수: {sub_item['score']}점): {sub_item['content']}\n"

    template = """
당신은 지리 과목의 백지도 문항을 채점하는 전문 채점관입니다. 학생이 백지도에 표기한 텍스트 정보를 바탕으로 다음 지시사항에 따라 채점하고 피드백을 제공해주세요.

--- 학생이 백지도에 표기한 텍스트 ---
{recognized_texts_str}

--- 평가 루브릭 ---
{rubric_str}

--- 지시사항 ---
1. 학생이 백지도에 표기한 텍스트({recognized_texts_str})를 평가 루브릭과 비교하여 채점해주세요.
2. 평가 루브릭의 각 '주요 채점 요소'별로 점수를 부여하고, 최종 합산 점수를 계산해주세요.
3. 학생 답안에 대한 교과 내용적인 피드백을 제공해주세요. 특히, 표기된 지리적 요소의 정확성과 누락 여부에 집중해주세요.
4. 학생 답안이 '의사 응답(bluffing)'인지 여부를 판단하고, 그렇다면 그 이유를 간략하게 설명해주세요. 의사 응답은 내용 없이 길게 늘어뜨리거나, 관련 없는 내용을 포함하는 경우를 의미합니다.
5. 각 주요 채점 요소별로 점수를 부여한 근거를 상세하게 작성해주세요.
6. **반드시 아래 `format_instructions`에 명시된 JSON 형식에 맞춰 `채점결과`와 `피드백` 두 가지 최상위 키를 모두 포함하여 응답을 생성해주세요.**
{format_instructions}
"""
    prompt = PromptTemplate(
        template=template,
        input_variables=["recognized_texts_str", "rubric_str", "format_instructions"]
    )
    return prompt.format(recognized_texts_str=", ".join(recognized_texts), rubric_str=rubric_str, format_instructions=format_instructions)

def grade_map_question(
    student_name: str,
    uploaded_image: Any, # Streamlit UploadedFile 객체로 변경
) -> Dict:
    """
    학생의 백지도 답안에서 텍스트를 추출합니다.
    """
    try:
        # 1. Image-to-Text 모델 호출
        image_to_text_output = extract_text_from_image(uploaded_image)
        recognized_texts = image_to_text_output.get("recognized_texts", [])
        
        if not recognized_texts:
            return {"이름": student_name, "오류": "이미지에서 텍스트를 인식하지 못했습니다."}

        return {
            "이름": student_name,
            "인식된_텍스트": recognized_texts, # 원본 답안 대신 인식된 텍스트 저장
        }

    except Exception as e:
        return {"이름": student_name, "오류": f"백지도 텍스트 추출 중 오류 발생: {e}"}