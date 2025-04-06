# Great British Bake Off RAG-Based Chatbot 

This chabot can be used to answer questions and search through the bakes that contestants made on the Great British Baking Show. Here is an example output:

<img width="598" alt="Image" src="https://github.com/user-attachments/assets/690c93b2-61b6-448d-aa1d-cfcdd6c4f375" />


## How it works
`embeddings.npy` stores embeddings obtained with the  `text-embedding-ada-002` model on the summarized descriptions of each bake across all episodes of the show. Summarized descriptions were obtained from Netflix. 

`metadata.json` stores metadata associated with each bake including baker name, challenge (signature / technical / showstopper), timestamp, season, episode number, and whether the episode is from holiday bake off or not.

Here are the steps that occur after a user prompt is provided in the chatbot: 

1. An embedding of the query is constructed using `text-embedding-ada-002`.
2. The FAISS algorithm is used to find the top 100 related bakes in the vector database to the query embedding.
3. Cosine similarity is then used to rank order the top 100  bakes based on similarity to the query embedding.
4. The GPT API is called with the top 10 bakes (along with the assoociated metadata from `metadata.json`) from step (3) to return the final output. 

## How to Run
At the moment this app only works locally. Make sure that you add your `OPENAI_API_KEY` to a .env file or set it as an environment variable. Then, run `python app.py` to load up the chatbot.
