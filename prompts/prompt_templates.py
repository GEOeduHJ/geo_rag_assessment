from langchain_core.prompts import PromptTemplate

def get_grading_prompt(question_type: str, rubric: list, student_answer: str, retrieved_docs: str, format_instructions: str) -> str:
    """
    문항 유형, 루브릭, 학생 답안, 검색된 문서를 기반으로 채점 프롬프트를 생성합니다.
    """
    rubric_str = ""
    for i, item in enumerate(rubric):
        rubric_str += f"- 주요 채점 요소 {i+1}: {item['main_criterion']}\n"
        for j, sub_item in enumerate(item['sub_criteria']):
            rubric_str += f"  - 세부 내용 {j+1} (점수: {sub_item['score']}점): {sub_item['content']}\n"

    base_template = """
당신은 지리 과목의 서답형 문항을 채점하는 전문 채점관입니다. 다음 지시사항에 따라 학생의 답안을 채점하고 피드백을 제공해주세요.

--- 참고 자료 ---
{retrieved_docs}

--- 평가 루브릭 ---
{rubric_str}

--- 학생 답안 ---
{student_answer}

--- 지시사항 ---
1. 학생 답안을 평가 루브릭과 참고 자료를 바탕으로 채점해주세요.
2. 평가 루브릭의 각 '주요 채점 요소'별로 점수를 부여하고, 최종 합산 점수를 계산해주세요.
3. 학생 답안에 대한 교과 내용적인 피드백을 제공해주세요.
4. 학생 답안이 '의사 응답(bluffing)'인지 여부를 판단하고, 그렇다면 그 이유를 간략하게 설명해주세요. 의사 응답은 내용 없이 길게 늘어뜨리거나, 관련 없는 내용을 포함하는 경우를 의미합니다.
5. 각 주요 채점 요소별로 점수를 부여한 근거를 상세하게 작성해주세요.
{format_instructions}

--- 문항 유형 ---
{question_type}

"""

    prompt = PromptTemplate(
        template=base_template,
        input_variables=["retrieved_docs", "rubric_str", "student_answer", "question_type", "format_instructions"]
    )
    return prompt.format(retrieved_docs=retrieved_docs, rubric_str=rubric_str, student_answer=student_answer, question_type=question_type, format_instructions=format_instructions)
