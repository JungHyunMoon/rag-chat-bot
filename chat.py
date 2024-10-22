import streamlit as st
import uuid

from dotenv import load_dotenv
from llm import get_ai_response
from upstage_ocr import ocr_service

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

st.sidebar.title("🛠️Options \n 체팅에 적용되는 다양한 옵션들 입니다.")
st.sidebar.divider()
files = st.sidebar.file_uploader(
    label="체팅 입력 전 로그 사진을 첨부해 주세요",
    type=["JPEG", "PNG", "BMP", "PDF", "TIFF", "HEIC", "DOCX", "PPTX", "XLSX"],
    accept_multiple_files=False,
    key=st.session_state["file_uploader_key"],
    help="""
        지원 파일 형식: JPEG, PNG, BMP, PDF, TIFF, HEIC, DOCX, PPTX, XLSX
    """
)
st.sidebar.divider()
model = st.sidebar.selectbox(
    "사용할 모델을 선택해 주세요",
    ("OpenAI GPT 4o", "OpenAI GPT 4o-mini", "Solar Pro Preview"),
)
st.sidebar.write("You selected:", model)

# 이전 메세지들 표시
for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 채팅 입력 감지
if user_question := st.chat_input(placeholder="디리아에 관련된 궁금한 내용들을 말씀해주세요!"):

    # 사용자 입력 표시
    with st.chat_message("user"):
        st.write(user_question)

    # 파일이 업로드되었는지 확인
    if files:
        with st.status("OCR 서비스 수행중..", expanded=True) as status:

            st.write("이미지에서 텍스트를 추출하는 중입니다...")
            ocr_text = ocr_service(files.getvalue())  # OCR 텍스트 결과를 가져옴

            if ocr_text is None:
                st.write("File Parsing Fail")
            else:
                # OCR 결과를 사용자 질문 앞에 추가
                with st.chat_message("user"):
                    st.code(ocr_text)
                user_question = f"{ocr_text}\n\n{user_question}"

            status.update(
                label="OCR 서비스 완료!", state="complete", expanded=False
            )

    # 사용자 메시지 세션에 저장
    # OCR 결과를 담을지 고민 필요
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
