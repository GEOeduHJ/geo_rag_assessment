import streamlit as st
import pandas as pd
import plotly.express as px

def display_dashboard(graded_results: list):
    """
    채점 결과를 바탕으로 대시보드를 시각화합니다.
    학생 개별 데이터 및 반 전체 평균 데이터를 표시합니다.
    """
    if not graded_results:
        st.info("시각화할 채점 결과가 없습니다.")
        return

    results_df = pd.DataFrame(graded_results)

    # '채점결과' 딕셔너리에서 '합산_점수'를 추출하여 새로운 컬럼으로 추가
    # '채점결과'가 딕셔너리가 아닌 문자열일 경우를 대비하여 안전하게 처리
    results_df['합산_점수'] = results_df['채점결과'].apply(lambda x: x.get('합산_점수', 0) if isinstance(x, dict) else 0)

    st.subheader("📊 채점 결과 대시보드")

    # 탭 생성
    tab1, tab2, tab3, tab4 = st.tabs(["반 전체 통계", "루브릭 항목별 분석", "학생 개별 점수", "개별 학생 분석"])

    with tab1:
        st.write("### 반 전체 통계")

        # 평균 점수
        avg_score = results_df['합산_점수'].mean()
        st.metric(label="반 평균 점수", value=f"{avg_score:.2f}점")

        # 점수 분포 히스토그램
        st.write("#### 학생 합산 점수 분포 (히스토그램)")
        fig_hist = px.histogram(results_df, x='합산_점수', nbins=10, title='학생 합산 점수 분포')
        st.plotly_chart(fig_hist, use_container_width=True)

    with tab2:
        st.write("### 루브릭 항목별 분석")
        # '채점결과'가 딕셔너리인 경우에만 처리
        rubric_scores_list = [res['채점결과'] for res in graded_results if isinstance(res.get('채점결과'), dict)]
        rubric_scores = pd.DataFrame(rubric_scores_list)
        
        # '합산_점수' 컬럼 제외
        rubric_scores = rubric_scores.drop(columns=['합산_점수'], errors='ignore')
        
        if not rubric_scores.empty:
            avg_rubric_scores = rubric_scores.mean().reset_index()
            avg_rubric_scores.columns = ['채점 항목', '평균 점수']
            fig_bar = px.bar(avg_rubric_scores, x='채점 항목', y='평균 점수', title='루브릭 항목별 평균 점수')
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("루브릭 항목별 점수 데이터가 없습니다.")

    with tab3:
        st.write("### 학생 개별 점수 (산포도)")
        st.write("#### 학생별 합산 점수")
        fig_scatter = px.scatter(results_df, x='이름', y='합산_점수', text='합산_점수', title='학생별 합산 점수')
        fig_scatter.update_traces(textposition='top center')
        st.plotly_chart(fig_scatter, use_container_width=True)

    with tab4:
        st.write("### 개별 학생 분석")

        student_names = results_df['이름'].unique()
        selected_student = st.selectbox("학생 선택", student_names)

        if selected_student:
            student_data = results_df[results_df['이름'] == selected_student].iloc[0]
            st.write(f"#### {selected_student} 학생의 상세 분석")
            st.write(f"**합산 점수:** {student_data['합산_점수']}점")
            st.write("**학생 답안:**")
            st.info(student_data['답안'])

            st.write("**채점 결과:**")
            # '채점결과'가 딕셔너리인 경우에만 표시
            if isinstance(student_data.get('채점결과'), dict):
                for criterion, score in student_data['채점결과'].items():
                    if criterion != '합산_점수':
                        st.write(f"- {criterion}: {score}점")
            
            st.write("**점수 판단 근거:**")
            # '점수_판단_근거'가 딕셔너리인 경우에만 표시
            if isinstance(student_data.get('점수_판단_근거'), dict):
                for criterion, reason in student_data['점수_판단_근거'].items():
                    st.write(f"- {criterion}: {reason}")
            else:
                st.info("점수 판단 근거 데이터가 없습니다.")

            st.write("**피드백:**")
            # '피드백'이 딕셔너리인 경우에만 표시
            if isinstance(student_data.get('피드백'), dict):
                st.write(f"- 교과 내용 피드백: {student_data['피드백'].get('교과_내용_피드백', 'N/A')}")
                st.write(f"- 의사 응답 여부: {student_data['피드백'].get('의사_응답_여부', 'N/A')}")
                if student_data['피드백'].get('의사_응답_여부', False):
                    st.write(f"  - 설명: {student_data['피드백'].get('의사_응답_설명', 'N/A')}")
            else:
                st.info("피드백 데이터가 없습니다.")
            
            st.write("**참고 문서:**")
            st.info(student_data['참고문서'])