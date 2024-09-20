from asyncio import wait

import streamlit as st

from dotenv import load_dotenv
from llm import get_ai_response
from urllib.parse import quote


st.set_page_config(page_title="디리아 챗봇", page_icon="🤖")

st.title("🤖 디리아 챗봇")
st.caption("디리아에 관련된 모든것을 답해드립니다!")

load_dotenv()


def write_sources(documents):
    sources = set()  # 중복을 제거하기 위해 set 사용

    s3_url = "https://ragresource.s3.ap-northeast-2.amazonaws.com/"

    unique_sources = ["\n*이 정보는 다음의 자료를 기반으로 제공되었습니다.*"]

    for doc in documents:
        source = doc.metadata.get("source").replace(".txt", "")
        if source not in sources:
            sources.add(source)
            link = s3_url + quote(source) # 한글 url encoding
            link = "\n\n" + f"[{source}]({link})"
            unique_sources.append(link)

    return unique_sources


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
        ai_response, ai_resource = get_ai_response(user_question)
        with st.chat_message("ai"):
            ai_message = st.write_stream(ai_response)
            st.session_state.message_list.append({"role": "ai", "content": ai_message})
        with st.chat_message(":information_source:"):
            resource = st.write_stream(write_sources(ai_resource))
            st.session_state.message_list.append({"role": "source", "content": resource})
