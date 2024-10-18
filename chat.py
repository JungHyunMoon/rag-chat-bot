import streamlit as st
import uuid

from dotenv import load_dotenv
from llm import get_ai_response
from upstage_ocr import ocr_service
from PIL import Image

# 페이지 설정
st.set_page_config(page_title="디리아 챗봇", page_icon="🤖")

st.title("🤖 디리아 챗봇")
st.caption("디리아에 관련된 모든 것을 답해드립니다!")

load_dotenv()

# 각 세션 초기화 (세션마다 한 번만 실행)
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = str(uuid.uuid4())

if 'message_list' not in st.session_state:
    st.session_state.message_list = []

if "file_uploader_key" not in st.session_state:
    st.session_state["file_uploader_key"] = 0

if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = []

files = st.sidebar.file_uploader("체팅 입력 전 에러 이미지를 첨부해주세요", type=None,
                                 accept_multiple_files=False,
                                 key=st.session_state["file_uploader_key"], help="여기에는 도움말 작성하기", on_change=None, args=None,
                                 kwargs=None, disabled=False, label_visibility="visible")

def clear_files():
    st.session_state.uploaded_files += 1

# 이전 메세지들 표시
for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 채팅 입력 감지
if user_question := st.chat_input(placeholder="디리아에 관련된 궁금한 내용들을 말씀해주세요!"):
    # 파일이 업로드되었는지 확인
    if files:
        with st.spinner("이미지에서 텍스트를 추출하는 중입니다..."):

            # OCR 서비스 호출 (ocr_service는 이미지 리스트를 처리한다고 가정)
            ocr_text = ocr_service(files.getvalue())  # OCR 텍스트 결과를 가져옴

            # OCR 결과를 사용자 질문 앞에 추가
            user_question = f"{ocr_text}\n\n{user_question}"

    # 사용자 입력 표시
    with st.chat_message("user"):
        st.write(user_question)

    # 사용자 메시지 세션에 저장
    st.session_state.message_list.append({"role": "user", "content": user_question})

    # AI 답변 생성 중 로딩 표시
    with st.spinner("답변을 생성하는 중입니다..."):
        ai_response = get_ai_response(user_question, st.session_state['session_id'])

        # AI 응답 표시
        with st.chat_message("ai"):
            ai_message = st.write_stream(ai_response)
            st.session_state.message_list.append({"role": "ai", "content": ai_message})

    # 업로드 파일 초기화
    st.session_state["file_uploader_key"] += 1
    st.rerun()