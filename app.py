from flask import Flask, render_template, request, jsonify
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv
import json
import faiss

# Initialize Flask app
app = Flask(__name__)
load_dotenv()
client = OpenAI()
chat_history = []

# Relevant constants
EMBEDDING_MODEL = "text-embedding-ada-002"
VECTOR_DB = "embeddings.npy"
METADATA = "metadata.json"

def cosine_similarity(a, b):
  dot_product = sum([x * y for x, y in zip(a, b)])
  norm_a = sum([x ** 2 for x in a]) ** 0.5
  norm_b = sum([x ** 2 for x in b]) ** 0.5
  return dot_product / (norm_a * norm_b)

def retrieve(query, top_n=10):
    # Obtain the query embedding
    client = OpenAI()
    query_embedding = client.embeddings.create(
        model=EMBEDDING_MODEL, 
        input=query).data[0].embedding
    
    similarities = []

    # Load the vector DB and create a FAISS index
    db = np.load(VECTOR_DB)
    index = faiss.IndexFlatL2(db.shape[1])
    index.add(db)
    
    # Use FAISS to compute top 100 closest embeddings to query and obtain indices
    query_arr = np.expand_dims(np.array(query_embedding), axis=0)
    _, I = index.search(query_arr, 100)
    indices = I[0].tolist()

    # Load metadata JSON file
    with open(METADATA, "r") as f:
        metadata = json.load(f)
    
    # Obtain cosine similarity for each of the top 100 embeddings
    for ind in indices:
        embedding = db[ind]
        similarity = cosine_similarity(query_embedding, embedding.tolist())
        chunk = metadata[ind]
        similarities.append((chunk, similarity))

    # Sort by similarity in descending order, because higher similarity means more relevant chunks
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Finally, return the top N most relevant chunks
    return similarities[:top_n]

@app.route("/", methods=["POST", "GET"])
def index():
    if request.method == "POST":
        query = request.form.get("query")
        retrieved_knowledge = retrieve(query, top_n=100)

        # Update the conversation history
        chat_history.append({'role': 'user', 'content': query})

        # Construct the instruction prompt
        instruction_prompt = f"""
        You are a helpful chatbot for The Great British Baking Show. Respond to user questions with relevant details.
        Here are some retrieved pieces of context from the show:
        {retrieved_knowledge}
        """

        # Add previous chat context to the instruction
        for message in chat_history:
            instruction_prompt += f"{message['role']}: {message['content']}\n"

        response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {'role': 'developer', 'content': instruction_prompt},
        ],
        )
        response = response.choices[0].message.content

        # Update chat history with the assistant's response
        chat_history.append({'role': 'assistant', 'content': response})
        return jsonify({"answer": response})
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
