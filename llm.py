from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate, \
    PromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_upstage import UpstageEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from config import answer_examples

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def get_retriever():
    # embedding = OpenAIEmbeddings(model='text-embedding-3-large')
    # index_name = 'wiki-openai-index'
    # namespace = "doc_v2"
    embedding = UpstageEmbeddings(model='solar-embedding-1-large-query')
    index_name = 'wiki-upstage-index'
    namespace = "chunk_1000_v2"
    database = PineconeVectorStore.from_existing_index(index_name=index_name, namespace=namespace, embedding=embedding)
    retriever = database.as_retriever(search_kwargs={'k': 4}, return_source_documents=True)
    return retriever


def get_history_retriever():
    llm = get_llm()
    retriever = get_retriever()

    # 체팅 내역 유지를 위한 검증된 프롬프트 사용
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    return history_aware_retriever


def get_llm(model='gpt-4o-mini'):
    llm = ChatOpenAI(model=model)
    return llm


def get_dictionary_chain():
    dictionary = [
        "DIREA -> (주)디리아는 제1금융, 저축은행, 증권사, 보험사, 카드사, 공공기관 등을 대상으로 한 금융업무솔루션과 네트웍 기반 미들웨어 개발을 주력으로 하고 있는 금융IT 전문기업",
        "FEP -> Front End Process의 약자로 대외계를 지칭하는 용어",
        "MCI -> Multi Channel Interface의 약자",
        "EDI -> 기업 간 데이터 교환을 위한 표준화된 인터페이스 방식",
        "TPS -> 초당 거래 처리량",
        "REST API -> HTTP 시스템을 위한 소프트 아키텍처",
        "APIM -> API Management의 약자, API 관리 프로세스",
        "배치 -> 특정 시간을 설정하여 일괄 처리하는 작업",
        "HA -> High Availability, 이중화 구성으로 장애 복구를 지원",
        "Adaptor -> 다양한 프로토콜을 사용하여 연계하는 서비스",
        "프레임워크 -> 솔루션 개발을 돕는 소프트웨어 환경"
    ]

    llm = get_llm()

    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 분석하고, 사전에 있는 용어는 변경해주세요.
        사전 내용은 다음과 같습니다:
        사전: {dictionary}

        만약 질문에 사전에 있는 용어가 없거나, 사전에 포함되지 않은 내용이 있더라도 그 부분은 그대로 유지해주세요.
        용어를 교체할 필요가 없다면 질문을 그대로 리턴해주세요.

        질문: {{question}}
    """)

    dictionary_chain = prompt | llm | StrOutputParser()

    return dictionary_chain


def get_rag_chain():
    llm = get_llm()
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{answer}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples,
    )
    system_prompt = ("""
    You are an AI assistant that provides detailed and accurate answers based on the provided documents.
    Use the information from the documents to answer the user's query.
    If the answer is not present in the documents, say that you don't have enough information to answer.
    {context}
    """
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = get_history_retriever()


    question_answer_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=qa_prompt
    )

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_rag_chain


def get_ai_response(user_message, session_id):
    # dictionary_chain = get_dictionary_chain() # 사용시 하단 input -> question으로 수정
    rag_chain = get_rag_chain()
    # final_chain = {"input": dictionary_chain} | rag_chain
    ai_response = rag_chain.pick("answer").stream(
        {
            "input": user_message
        },
        config={
            "configurable": {"session_id": session_id}
        }
    )

    return ai_response
