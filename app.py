###############################
# app.py

import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import requests

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate API Keys
if not GOOGLE_API_KEY or not OPENAI_API_KEY:
    st.error("Missing API keys! Make sure your .env file has both GOOGLE_API_KEY and OPENAI_API_KEY.")
    st.stop()

# Initialize OpenAI tools
embedding = OpenAIEmbeddings(model='text-embedding-3-small')
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.2)

# Prompt templates
qa_prompt = PromptTemplate(
    template='''You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the content is not valid or not available, say: "I don't know."

{context}

Question: {question}
''',
    input_variables=['context', 'question']
)

translation_prompt = PromptTemplate(
    template='''Translate the following transcript text to English. Return ONLY the translated text.

{context}
''',
    input_variables=['context']
)

# Helper Functions
def search_youtube(query, max_results=5):
    url = (
        f"https://www.googleapis.com/youtube/v3/search"
        f"?part=snippet&type=video&maxResults={max_results}"
        f"&q={query}&key={GOOGLE_API_KEY}"
    )
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        results = []
        for item in data["items"]:
            video_id = item["id"]["videoId"]
            title = item["snippet"]["title"]
            description = item["snippet"]["description"]
            thumbnail = item["snippet"]["thumbnails"]["default"]["url"]
            results.append({
                "video_id": video_id,
                "title": title,
                "description": description,
                "thumbnail": thumbnail
            })
        return results
    except Exception as e:
        st.error(f"Error fetching from YouTube API: {e}")
        return []

def get_transcript(video_id, lang_code='en'):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang_code])
        return transcript, None
    except (TranscriptsDisabled, NoTranscriptFound):
        try:
            list_transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
            languages = [t.language_code for t in list_transcripts]
            return None, languages
        except:
            return None, "Transcript not found or video may not exist."
    except Exception as e:
        return None, str(e)

def create_final_relevant_doc(relevant_documents):
    return '\n\n'.join(i.page_content for i in relevant_documents)

# --- Streamlit App ---
st.set_page_config(page_title="🎬 YouTube RAG App", page_icon="🎬", layout="centered")
st.title("🎬 YouTube RAG App")
st.write("Enter a search term below to find YouTube videos. Select a video to ask questions!")

query = st.text_input("🔍 Search YouTube:")

if query:
    with st.spinner("Searching YouTube..."):
        results = search_youtube(query)

    if results:
        st.subheader("Search Results:")
        for video in results:
            col1, col2 = st.columns([1, 4])
            with col1:
                st.image(video["thumbnail"])
            with col2:
                st.markdown(f"**{video['title']}**")
                st.caption(video["description"][:100] + "...")
                if st.button(f"Select: {video['title']}", key=video["video_id"]):
                    st.session_state["selected_video_id"] = video["video_id"]
                    st.session_state["selected_video_title"] = video["title"]
                    st.session_state["selected_language"] = "en"
                    st.session_state.pop("cached_transcript", None)
                    st.session_state.pop("cached_vector_store", None)
                    st.rerun()
    else:
        st.warning("No videos found. Try a different keyword.")

# --- Transcript & Question Section ---
if "selected_video_id" in st.session_state:
    video_id = st.session_state["selected_video_id"]
    video_title = st.session_state["selected_video_title"]
    selected_language = st.session_state.get("selected_language", "en")

    st.success(f"Selected video: {video_title}")
    st.video(f"https://www.youtube.com/watch?v={video_id}")

    if "cached_transcript" not in st.session_state or "cached_vector_store" not in st.session_state:
        st.info("Fetching transcript...")

        transcript, error = get_transcript(video_id, lang_code=selected_language)

        if transcript:
            final_content = ''.join(i['text'] for i in transcript)

            if selected_language != 'en':
                st.info("Translating transcript to English...")
                translation_chain = translation_prompt | llm | StrOutputParser()
                final_content = translation_chain.invoke({"context": final_content})

            st.session_state["cached_transcript"] = final_content

            # Create vector store
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.create_documents([final_content])
            vector_store = FAISS.from_documents(chunks, embedding)
            st.session_state["cached_vector_store"] = vector_store

            st.success("Transcript ready! Ask your question below.")
        elif isinstance(error, list):
            new_lang = error[0]
            if new_lang != selected_language:
                st.session_state["selected_language"] = new_lang
                st.warning(f"No English transcript. Using available language: {new_lang}")
                st.rerun()
            else:
                st.error("No valid transcripts available.")
        else:
            st.error(f"Transcript Error: {error}")

    # Ask Question UI
    if "cached_transcript" in st.session_state and "cached_vector_store" in st.session_state:
        with st.form("qa_form"):
            question = st.text_input("💬 Ask a question about the video:")
            submitted = st.form_submit_button("Submit Question")

            if submitted and question:
                retriever = st.session_state["cached_vector_store"].as_retriever(
                    search_type='similarity', search_kwargs={'k': 4}
                )

                parallel_chain = RunnableParallel({
                    'context': retriever | RunnableLambda(create_final_relevant_doc),
                    'question': RunnablePassthrough()
                })

                full_chain = parallel_chain | qa_prompt | llm | StrOutputParser()

                with st.spinner("Thinking..."):
                    answer = full_chain.invoke(question)
                st.markdown("### ✅ Answer:")
                st.write(answer)

# Restart search
if "selected_video_id" in st.session_state:
    if st.button("🔄 New Search"):
        st.session_state.clear()
        st.rerun()
