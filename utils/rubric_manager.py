import streamlit as st

def get_default_rubric():
    """예시 루브릭 데이터를 반환합니다."""
    return [
        {
            'main_criterion': '핵심 개념 이해 및 적용',
            'sub_criteria': [
                {'score': 2, 'content': '동해안의 해안선 특징(단조로움)을 정확히 서술함.'},
                {'score': 2, 'content': '해안선 형성의 주요 원인(경동성 요곡 운동)을 언급함.'},
                {'score': 1, 'content': '융기, 침강과 같은 지각 변동과 해안선 형성의 관계를 이해하고 있음.'}
            ]
        },
        {
            'main_criterion': '지리적 사실 및 용어 사용의 정확성',
            'sub_criteria': [
                {'score': 2, 'content': '해안단구, 석호 등 동해안의 대표적인 지형을 올바르게 제시함.'},
                {'score': 2, 'content': '리아스식 해안, 갯벌 등 서해안의 특징과 비교하여 설명함.'},
                {'score': 1, 'content': '지리 용어를 적절하고 정확하게 사용함.'}
            ]
        }
    ]

def initialize_rubric():
    """세션 상태에 루브릭이 없으면 예시 루브릭으로 초기화합니다."""
    if 'rubric_items' not in st.session_state:
        st.session_state['rubric_items'] = get_default_rubric()

def add_rubric_item():
    """새로운 주요 채점 요소를 추가합니다."""
    st.session_state['rubric_items'].append({'main_criterion': '', 'sub_criteria': []})
    st.rerun()

def delete_rubric_item(index):
    """특정 주요 채점 요소를 삭제합니다."""
    if 0 <= index < len(st.session_state['rubric_items']):
        st.session_state['rubric_items'].pop(index)
        st.rerun()

def add_sub_criterion(main_index):
    """새로운 세부 채점 요소를 추가합니다."""
    if 0 <= main_index < len(st.session_state['rubric_items']):
        st.session_state['rubric_items'][main_index]['sub_criteria'].append({'score': 0, 'content': ''})
        st.rerun()

def delete_sub_criterion(main_index, sub_index):
    """특정 세부 채점 요소를 삭제합니다."""
    if 0 <= main_index < len(st.session_state['rubric_items']):
        if 0 <= sub_index < len(st.session_state['rubric_items'][main_index]['sub_criteria']):
            st.session_state['rubric_items'][main_index]['sub_criteria'].pop(sub_index)
            st.rerun()

def display_rubric_editor():
    """평가 루브릭을 표시하고 편집하는 UI를 렌더링합니다."""
    initialize_rubric()

    st.subheader("평가 루브릭 입력")

    # 루브릭 제어 버튼
    col1, col2, col3 = st.columns([1, 1, 8])
    with col1:
        if st.button("➕ 요소 추가"):
            add_rubric_item()
    with col2:
        if st.button("🔄 초기화"):
            st.session_state['rubric_items'] = get_default_rubric()
            st.rerun()

    # 루브릭 편집기
    if not st.session_state['rubric_items']:
        st.info("위 버튼을 클릭하여 루브릭 작성을 시작하세요.")
    
    for i, item in enumerate(st.session_state['rubric_items']):
        with st.expander(f"주요 채점 요소 {i+1}: {item.get('main_criterion', '')}", expanded=True):
            col1, col2 = st.columns([4, 1])
            with col1:
                item['main_criterion'] = st.text_input(
                    "주요 채점 요소 내용", 
                    value=item['main_criterion'], 
                    key=f"main_criterion_{i}"
                )
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("삭제", key=f"delete_main_{i}", type="secondary"):
                    delete_rubric_item(i)

            st.markdown("---")
            st.write("세부 채점 요소")
            
            for j, sub_item in enumerate(item['sub_criteria']):
                col_sub1, col_sub2, col_sub3 = st.columns([1, 4, 1])
                with col_sub1:
                    sub_item['score'] = st.number_input(
                        "점수", 
                        min_value=0, 
                        max_value=100, 
                        value=sub_item['score'], 
                        key=f"sub_score_{i}_{j}"
                    )
                with col_sub2:
                    sub_item['content'] = st.text_area(
                        "세부 내용", 
                        value=sub_item['content'], 
                        key=f"sub_content_{i}_{j}", 
                        height=50
                    )
                with col_sub3:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("삭제", key=f"delete_sub_{i}_{j}", type="secondary"):
                        delete_sub_criterion(i, j)
            
            if st.button("세부 요소 추가", key=f"add_sub_{i}"):
                add_sub_criterion(i)

    # 최종 루브릭을 세션 상태에 저장
    if st.session_state.get('rubric_items'):
        st.session_state['final_rubric'] = st.session_state['rubric_items']
    else:
        st.session_state['final_rubric'] = []
