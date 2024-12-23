import streamlit as st
from langchain_pinecone import PineconeVectorStore
import google.generativeai as genai
from PIL import Image
import os
import re
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import base64
from google.cloud import aiplatform
import os
REGION = "us-central1"
PROJECT_ID = "test123-443617"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r""
aiplatform.init(project=PROJECT_ID, location=REGION)

# Environment setup
# os.environ["GOOGLE_API_KEY"] = ""
os.environ["PINECONE_API_KEY"] = ""

# Initialize Pinecone and LangChain Embeddings
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Pinecone VectorStore
index_name = "multifile"
index = pc.Index(index_name)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)


# Memory for Chat History
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True,output_key="answer")

# Conversational Retrieval Chain
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold", search_kwargs={"k": 10, "score_threshold": 0.5}
)

# Initialize Gemini Model
# model = genai.GenerativeModel(model_name="gemini-1.5-pro")
from langchain_google_genai import ChatGoogleGenerativeAI

# Wrap Gemini Model into LangChain-compatible format
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.5)

custom_prompt_template = """
You are an assistant that provides answers strictly based on the retrieved context. 
You must include relevant images or tables if they are mentioned in the context. Follow these rules:

1. **Answer**: Provide a concise answer based strictly on the context.
2. **Images**: If an image is relevant, include it in the response as: `ImagePath: image_name.png` and figure title.
3. **Tables**: If a table is relevant, display the table.
4. **No Context Found**: respond with a polite and empathetic apology, explaining the limitation. Offer assistance in finding alternative information or a way to address the user's needs effectively."

Context:
{context}

Chat History:
{chat_history}

User Question:
{question}

Answer:
"""
from langchain.prompts import PromptTemplate

# Create a PromptTemplate instance
custom_prompt = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template=custom_prompt_template
)

# LangChain ConversationalRetrievalChain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=model, 
    retriever=retriever, 
    memory=memory, 
    return_source_documents=True,
    output_key="answer" , # Specify the output key to store in memory
    combine_docs_chain_kwargs={"prompt": custom_prompt}

)

# Function to convert an image to base64
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string
# Path to your local background image
# local_image_path = r"C:\Users\BharathAmudala\Downloads\Screenshot 2024-12-20 at 12.59.45 PM 1.png"  # Replace with the local image path

# Convert image to base64
# base64_image = image_to_base64(local_image_path)

# # Background Image CSS with base64 encoding
# background_css = f"""
#     <style>
#     .stApp {{
#         background-image: url('data:image/jpeg;base64,{base64_image}');
#         background-size: cover;
#         background-position: center;
#     }}

#     /* Position chat input at the bottom-right */
#     .stTextInput {{
#         position: fixed;
#         bottom: 10px;
#         right: 10px;
#         width: 300px;
#         z-index: 100;
#         background-color: rgba(255, 255, 255, 0.8);
#     }}

#     /* Additional custom styling to make the chat input box more prominent */
#     .stTextInput input {{
#         text-align: left;  /* Align the text to the left */
#     }}
#     </style>
# """


# # Inject the custom CSS for the background
# st.markdown(background_css, unsafe_allow_html=True)

# Streamlit Chat UI
st.title("Gemini rag chatbot")
st.write("Ask questions and follow-up queries based on the responses!")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
# Process and display images or tables if mentioned
def display_images_and_tables(response_text):
    # Extract images or tables using regex
    image_paths = re.findall(r"ImagePath:\s*(\S+)", response_text)
    table_html = re.search(r"TableHTML:\s*(.*?)(?=Answer:|ImagePath:|$)", response_text, re.DOTALL)

    # pattern = r"(Figure\s[A-Z0-9\-]+:\s.*?)(?=\n|$)"
    # pattern = r"FIGURE\s+[A-Z]-\d+:\s+.*?(?=\s*$)"
    pattern = r"Figure:\s+Figure\s+[A-Z]\.\d+\s+.*?(?=\s*$)"
    # pattern = r"(?:FIGURE|Figure):?\s+(?:Figure\s+)?[A-Z]-?\d+\.?\d*:?(\s+.*)?"


    # matches = re.findall(pattern, response_text, re.MULTILINE)
    # st.write("matches",matches)
    matches = re.findall(pattern, response_text, re.MULTILINE)
    print(f"Matches:{matches}")
    for match in matches:
        st.write(match.strip() if match else "No figure title found")


    
    fig_title = None
    # Output result
    if matches:
        print("Extracted Figure Captions:")
        # for item in match:
        fig_title = (matches[0])

    # Display images
    if image_paths:
        base_path = r"C:\Users\BharathAmudala\Desktop\Doc_AI\working_model\filtered_images"
        for path in image_paths:
            image_url = os.path.join(base_path, path)
            # st.write(f"Trying to load image from: {image_url}")  # Print image URL for debugging
            try:
                # Check if the file exists
                if os.path.exists(image_url):
                    image = Image.open(image_url)
                    if fig_title:
                        st.image(image,caption=fig_title)
                    else:
                        st.image(image, caption=os.path.basename(path))
                else:
                    st.error(f"Image file does not exist: {image_url}")
            except Exception as e:
                st.error(f"Failed to load image from {image_url}: {e}")

    # Display tables
    if table_html:
        st.markdown(table_html.group(1), unsafe_allow_html=True)

# Display the previous chat history
for message in st.session_state["chat_history"]:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    elif message["role"] == "assistant":
        st.chat_message("assistant").write(message["content"])
        display_images_and_tables(message["content"])


# Streamlit Chat Input and Display
if prompt := st.chat_input("Ask me anything..."):
    # Display user message
    st.chat_message("user").write(prompt)



    # Retrieve response from the chain
    with st.spinner("Thinking..."):
        response = qa_chain({"question": prompt})
        answer = response["answer"]

        import base64

        def image_to_base64(image_path):
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string
        image_path_history = re.findall(r"ImagePath:\s*(\S+)", answer)
        base_path = r"C:\Users\BharathAmudala\Desktop\Doc_AI\working_model\filtered_images"



        # st.write("checkkk",answer+image_to_base64(img_path))


        # Display AI message
        st.chat_message("assistant").write(answer)

        # if image_path_history:
        #     img_path = os.path.join(base_path, image_path_history[0])
        #     st.session_state["chat_history"].append({"role": "user", "content": prompt})
        #     st.session_state["chat_history"].append({"role": "assistant", "content": answer+image_to_base64(img_path)})
        # else:
        #     # Add the current exchange to chat history
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        st.session_state["chat_history"].append({"role": "assistant", "content": answer})

        
        display_images_and_tables(answer)

        # Update chat history in session state
        # st.session_state["chat_history"].append({"user": prompt, "assistant": answer})


