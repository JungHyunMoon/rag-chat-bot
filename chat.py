
import streamlit as st
import uuid

from dotenv import load_dotenv
from llm import get_ai_response


st.set_page_config(page_title="디리아 챗봇", page_icon="🤖")

st.title("🤖 디리아 챗봇")
st.caption("디리아에 관련된 모든것을 답해드립니다!")

load_dotenv()

# 세션 ID 초기화 (세션마다 한 번만 실행)
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = str(uuid.uuid4())

if 'message_list' not in st.session_state:
    st.session_state.message_list = []

for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if user_question := st.chat_input(placeholder="디리아에 관련된 궁금한 내용들을 말씀해주세요!"):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content": user_question})

    with st.spinner("답변을 생성하는 중입니다"):
        ai_response = get_ai_response(user_question, st.session_state['session_id'])
        with st.chat_message("ai"):
            ai_message = st.write_stream(ai_response)
            st.session_state.message_list.append({"role": "ai", "content": ai_message})
