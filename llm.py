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
    dictionary = []
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        그런 경우에는 질문만 리턴해주세요
        사전: {dictionary}
        
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
    dictionary_chain = get_dictionary_chain()
    rag_chain = get_rag_chain()
    final_chain = {"input": dictionary_chain} | rag_chain
    ai_response = final_chain.pick("answer").stream(
        {
            "question": user_message
        },
        config={
            "configurable": {"session_id": session_id}
        }
    )

    return ai_response
