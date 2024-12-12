import os
from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template

# Load environment variables
load_dotenv()

DB_CONNECTION_URL_2 = os.getenv("DB_CONNECTION_URL_2")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = Flask(__name__)

def validate_env_variables():
    if not DB_CONNECTION_URL_2:
        raise ValueError("Database connection URL is missing!")
    if not GOOGLE_API_KEY:
        raise ValueError("Google API key is missing!")
    if not GROQ_API_KEY:
        raise ValueError("Groq API key is missing!")

validate_env_variables()

def fetch_and_split_wikipedia_content(topic, load_max_docs=3, chunk_size=500, chunk_overlap=50):
    loader = WikipediaLoader(query=topic, load_max_docs=load_max_docs)  
    docs = loader.load() 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents(docs) 

    print(f"Created {len(splits)} chunks for the topic '{topic}'.")  # Print when chunks are made
    return splits

def store_embeddings_pgvector(texts, table_name):
    """Store embeddings in PGVector."""
    try:
        embeddings_generator = GoogleGenerativeAIEmbeddings(
            api_key=GOOGLE_API_KEY,
            model="models/embedding-001"
        )
        vectorstore = PGVector(
            connection_string=DB_CONNECTION_URL_2,
            embedding_function=embeddings_generator,
        )
        vectorstore.add_texts(texts, table_name=table_name)
        print(f"Data successfully stored in the '{table_name}' table.")  # Print when embeddings are stored
    except Exception as e:
        print(f"Error storing embeddings in PGVector: {e}")

def query_groq_with_response(user_query):
    try:
        # Initialize PGVector with the embedding function
        embeddings_generator = GoogleGenerativeAIEmbeddings(api_key=GOOGLE_API_KEY, model="models/embedding-001")
        vectorstore = PGVector(connection_string=DB_CONNECTION_URL_2, embedding_function=embeddings_generator)

        # Convert PGVector into a retriever
        retriever = vectorstore.as_retriever()

        # Retrieve relevant documents based on the user's query
        relevant_documents = retriever.get_relevant_documents(user_query)

        # Prepare the context from retrieved documents
        context = "\n".join([doc.page_content for doc in relevant_documents])

        # Query Groq with the retrieved context
        chatgroq = ChatGroq(api_key=GROQ_API_KEY, model="llama3-8b-8192", temperature=0.0, max_retries=2)
        prompt_template = PromptTemplate(
            input_variables=["query", "context"], 
            template="Query: {query}\nContext: {context}\nPlease generate a detailed response.")
        llm_chain = prompt_template | chatgroq

        # Generate a response from Groq using the context
        print(f"Generating response for query: {user_query}")
        response = llm_chain.invoke({"query": user_query, "context": context})

        return response.content if hasattr(response, 'content') else str(response)

    except Exception as e:
        print(f"Error in query generation: {e}")
        return "An error occurred while processing your query."

# Flask route to serve the main HTML page
@app.route('/')
def home():
    return render_template('index.html')

# Flask route to process the topic (add topic)
@app.route('/process_topic', methods=['POST'])
def process_topic():
    data = request.get_json()
    topic = data.get('topic')

    if not topic:
        return jsonify({"success": False, "message": "Topic is required"})

    try:
        # Fetch and process the topic
        transcriptions = fetch_and_split_wikipedia_content(topic)
        texts = [chunk.page_content for chunk in transcriptions]
        
        # Store the embeddings in PGVector
        store_embeddings_pgvector(texts, table_name='wikipedia_embeddings')

        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

# Flask route to handle user questions
@app.route('/ask_question', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question')

    if not question:
        return jsonify({"success": False, "message": "Question is required"})

    try:
        # Get the answer for the user question
        answer = query_groq_with_response(question)
        return jsonify({"success": True, "answer": answer})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
