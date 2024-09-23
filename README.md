# Direa AI Chat Bot

## Overview
This repository contains a project that utilizes LangChain and Streamlit to build a Retrieval Augmented Generation (RAG) application. The primary focus of this application is to provide insights and answers based on the DireaWiki

## Features
LangChain Integration: Utilizes LangChain to manage and interact with language models effectively.
Streamlit Interface: A user-friendly web interface created with Streamlit for seamless interaction.
Retrieval Augmented Generation (RAG): Combines retrieval-based techniques with generative models to produce accurate and context-aware answers.
Knowledge Base: Focuses on the DireaWiki as the primary knowledge base.
Chat History: Maintains a history of user interactions to provide contextually relevant answers.
Few-Shot Learning Templates: Enhances the model's responses by using predefined templates for better accuracy and consistency.

## Installation
1. Clone the repository:
```
git clone https://gitlab.direa.synology.me/techcare/rag-chat-bot.git
cd rag-chat-bot
```
2. Create and activate a virtual environment:
```
python3 -m venv venv
source venv/bin/activate
```
3. Install the required dependencies:
```
pip install -r requirements.txt
```

## Usage
1. Run the Streamlit application:
```
streamlit run chat.py
```
2. Open your web browser and navigate to the displayed local URL to interact with the application.