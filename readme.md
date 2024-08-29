# Tutorial: Vector Databases and Retrieval-Augmented Generation (RAG)

---

This tutorial will guide you through the essentials of working with vector databases and implementing a Retrieval-Augmented Generation (RAG) workflow. You will learn how to set up a vector database using Pinecone, integrate it with Google Gemini for generating enhanced responses.

Before diving into the coding, it’s crucial to understand the key concepts and tools involved. Make sure to familiarize yourself with the following resources:

- Vector Embeddings
	- [What are Vector Embeddings?](https://www.pinecone.io/learn/vector-embeddings/)
- Vector Databases
	- [What is a Vector Database and how does it work?](https://www.pinecone.io/learn/vector-database/)
- RAG
	- [What is RAG?](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- Langchain
	- [Langchain Introduction](https://python.langchain.com/v0.2/docs/introduction/)
	- [Langchain messages](https://python.langchain.com/v0.1/docs/modules/model_io/chat/message_types/)



# 1. Preliminary Setup
## 1.1 Create Python Virtual Environment

### On a Mac / Linux
Navigate to the directory where you want to create the virtual environment or create a new directory.

#### Create virtual environment
Then, create a virtual environment using:

```shell
python3 -m venv venv
```


This command will create a directory named `venv` (or whatever you name it) in your project directory.


#### Activate the Virtual Environment
To activate the virtual environment, use:

```shell
source venv/bin/activate
```


You should now see `(venv)` at the beginning of your terminal prompt, indicating that the virtual environment is active.


## On a Windows
Navigate to the directory where you want to create the virtual environment or create a new directory.


#### Create virtual environment
```shell
python -m venv venv
```


This command will create a directory named `venv` (or whatever you name it) in your project directory.


#### Activate the Virtual Environment

To activate the virtual environment, use the following command:

- **Command Prompt**

```shell
venv\Scripts\activate
```


- **PowerShell**

```shell
.\venv\Scripts\Activate
```


You should now see `(venv)` at the beginning of your terminal prompt, indicating that the virtual environment is active.


## 1.2 Create a Pinecone Account

We'll be using Pinecone as our vector database. It has a generous free tier and is very easy to set up.

1. Visit the Pinecone [home page](https://pinecone.io/) and go through the sign up process.

<img src="./images/pinecone%20sign%20up.png" />


2. Click on the `Create API key` button to open the modal to create a new key.

<img src="./images/Group%205.png" />

3. Type in the name of your key 

<img src="./images/create%20pc%20api%20key.png" />

Your Pinecone account is all set up. Copy the generated API key and save it in a secure location. We'll use this later in our code.


## 1.3 Create a Google Gemini API key

We'll be using Google Gemini as our LLM for the RAG application. Gemini also has a generous free tier. The only prerequisite is having a Google account.

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey). You should see this modal pop up if it's your first time visiting the website. Click on the `Get API key` button.

<img src="./images/Group%204.png" />


2. After accepting the terms and conditions, you'll be presented with another screen as shown below.

<img src="./images/Group%203.png" />


3. Click on the `Create API key` button to open the modal. On the modal, click `Create API key in new project`.

<img src="./images/Group%202.png" />


4. Copy the generated API key and save it in a secure location. We'll use this later in our code.

<img src="./images/Group%201.png" />



# 2. Creating the RAG Application

Now that we have our Pinecone Vector database and our Gemini API key set up, we can get into coding our application.

We first have to start with installing the python dependencies required for this project.

## 1. Installing dependencies
Make sure you're in your project directory and have already activated your Python virtual environment. After that, run the following command:

### On a Mac / Linux
```shell
pip3 install langchain langchain-pinecone langchain-huggingface langchain-community python-dotenv pymupdf langchain-google-genai
```


### On a Windows
```shell
pip install langchain langchain-pinecone langchain-huggingface langchain-community python-dotenv pymupdf langchain-google-genai
```


## 2. Project Structure

For this project, we'll require a `main.py` file, a `.env` file and a `docs` directory.

- `main.py`: This file will contain all the code for our application.
- `.env`: We'll store our API keys in this file and reference them in our code. We normally add this file to the `.gitignore` file to prevent accidentally pushing your API keys to GitHub. This is a huge security risk and must be avoided.
- `docs`: We'll keep all the PDF files we want ingested into our vector database in this directory.


## 3. Code Walkthrough

First create the required files and directory: `main.py`, `.env` and the `docs` directory.

### 3. 1. Import required libraries

We import the necessary libraries and modules to handle various tasks such as Pinecone operations, loading environment variables, embedding documents, handling document loading and splitting, and interacting with Google Gemini.

As we go through the code, we'll discuss what each module is used for.


```python
import os  
import getpass  
import time  
from uuid import uuid4  
  
from dotenv import load_dotenv  
from pinecone import Pinecone, ServerlessSpec  
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain_pinecone import PineconeVectorStore  
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader  
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain_core.messages import HumanMessage, SystemMessage  
from langchain_google_genai import ChatGoogleGenerativeAI
```



### 3.4. Add your API Keys

Add these environmental variables to the `.env` file. Replace `XXXXXX` with the API keys we copied from Pinecone and Google Gemini. 

```text
PINECONE_API_KEY=XXXXXXXX
GOOGLE_API_KEY=XXXXXXXX
```


### 3.3. Load Environment Variables

**`load_dotenv()`**: Loads environment variables from the `.env` file if it exists, allowing you to manage sensitive data like API keys without hardcoding them in the script.

```python
# Load the environment variables
load_dotenv()
```


### 3.4. Set Up API Keys

We check if the Pinecone and Gemini API keys are set in the environment variables. If not, the script prompts the user to input them securely using `getpass`.

```python
# Check if the Pinecone API key is set  
if not os.getenv("PINECONE_API_KEY"):  
    os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter your Pinecone API key: ")  
  
# Check if the Google API key is set  
if not os.getenv('GOOGLE_API_KEY'):  
    os.environ['GOOGLE_API_KEY'] = getpass.getpass("Enter your Google API key: ")
```


### 3.5. Initialize Pinecone and Create an Index
We instantiate the Pinecone client using the API key.

Before we create an index, the script checks if the desired index exists. If not, it creates a new one with specific configurations (e.g., dimension, metric). The script waits for the index to be ready before proceeding.

Once the index is ready, we print out its statistics for confirmation.


```python
pinecone_api_key = os.environ.get("PINECONE_API_KEY")  
pc = Pinecone(api_key=pinecone_api_key)  
  
# Create index  
index_name = "langchain-test-index"  # change if desired  
  
# Check if the index exists  
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]  
  
if index_name not in existing_indexes:  
    pc.create_index(  
        name=index_name,  
        dimension=768,  
        metric="cosine",  
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),  
    )    
    
    while not pc.describe_index(index_name).status["ready"]:  
        time.sleep(1)  
  
# Initialize Index  
index = pc.Index(index_name)  
  
# Print the index stats  
print(index.describe_index_stats())
```


### 3.6. Load Embeddings Model and Initialize Vector Store

We load a pre-trained HuggingFace embeddings model (`msmarco-bert-base-dot-v5`) which will be used to convert text into vector representations.

The `PineconeVectorStore` is initialized, linking the embeddings model with the Pinecone index for storing and retrieving document vectors.


```python
# Load embeddings model  
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/msmarco-bert-base-dot-v5")  
  
# Initialize vector store  
vector_store = PineconeVectorStore(index=index, embedding=embeddings)
```



### 3.7. Load and Split Documents

We define a function `load_and_split_documents()` that loads PDF documents from the `docs` directory using the `DirectoryLoader` and `PyMuPDFLoader`.

The loaded documents are split into chunks of 2000 characters with a 100-character overlap using `RecursiveCharacterTextSplitter` to ensure no information is lost at chunk boundaries.

You can modify these values to have different chunk sizes and character overlaps.


```python
# Function to load and split the documents  
def load_and_split_documents(path):  
    loader = DirectoryLoader(path, glob="**/*.pdf", loader_cls=PyMuPDFLoader)  
    documents = loader.load()  
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)  
    return text_splitter.split_documents(documents)  
  
  
# Load and split the documents  
document_path = os.path.join("docs")  
documents = load_and_split_documents(document_path)
```



### 3.8. Add Documents to Vector Store

We generate a list of unique UUIDs corresponding to each document chunk. The documents, along with their unique IDs, are added to the vector store. 

This step allows us to later retrieve and query the documents based on their vector representations.

```python
# Add documents to the vector store with unique IDs
uuids = [str(uuid4()) for _ in range(len(documents))]
vector_store.add_documents(documents=documents, ids=uuids)
```


After adding the documents to the vector store, visit your Pinecone dashboard and select your index to view the uploaded document chunks. You should see something similar to this:

<img src="./images/Group%206.png" />

Clicking on the pencil icon on a document chunk will open a modal that contains more information about that chunk. Here's an example:

<img src="./images/Group%207.png" />

You can see all the document metadata as well as the unique IDs we added using `uuid` and the vector embeddings as well.


### 3.9. Creating a Function To Query the Vector Store

The `retrieve_results()` function accepts the `query` parameter queries the vector store for the top 3 most similar documents to the input query using the `similarity_search_with_score()` method. 

The `k` value can be altered to changed the number of returned values from the vector database. For example, assigning a value of `3` here means we'll only get back the top-3 results from the vector database.


```python
# Query
def retrieve_results(query):
    return vector_store.similarity_search_with_score(query, k=3)

```


### 3.10. Create an Augmented Prompt for LLM

We create a function takes the `query` and the `results`(retrieved documents), combines them into a single prompt, and formats it for the language model.

We create a `SystemMessage` to set the role and a `HumanMessage` containing the combined input, which will be passed to the LLM.

You can tweak this prompt to have different responses from the LLM. For instance, we insert an instruction that tells the LLM to respond with `I'm not sure` if it doesn't come across an answer in the retrieved documents.


```python
def create_augmented_prompt(query, results):
    # Combine the query and the relevant document contents
    combined_input = (
            "Here are some documents that might help answer the question: "
            + query
            + "\n\nRelevant Documents:\n"
            + "\n\n".join([doc.page_content for doc, _ in results])
            + "\n\nPlease provide an answer based only on the provided documents. If the answer is not found in the "
              "documents, respond with 'I'm not sure'."
    )

    # Define the messages for the model
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=combined_input),
    ]

    return messages

```


### 3.11. Setting up Google Gemini
We initialize the Google Gemini LLM with specific parameters such as model type, temperature, and retry settings. You can learn more about these settings from the [docs](https://python.langchain.com/v0.2/docs/integrations/chat/google_generative_ai/).

For this project we're using Gemini, but this can be swapped out for any LLM. For instance, you can use any of the OpenAI models but they are not free to use unlike Gemini. You can also use open source models on [HuggingFace](https://huggingface.co/models) or use APIs from platforms that allow you to make rate-limited inferences like [Groq](https://groq.com/). 

Take note, the LLM setup will differ slightly from the way we set up Gemini in this project.

```python
llm = ChatGoogleGenerativeAI(  
    model="gemini-1.5-flash",  
    temperature=0,  
    max_tokens=None,  
    timeout=None,  
    max_retries=2,  
    return_full_text=False,  
)
```


### 3.12. Interactive Chat Interface in the Terminal

The `chat()` function creates a loop where users can input queries, receive results from the vector store, generate an augmented prompt, and retrieve an answer from the LLM.

Typing 'exit' ends the chat session.


```python
def chat():  
    print("Start chatting with the AI! Type 'exit' to end the conversation.")  
  
    while True:  
        query = input("You: ")  
        if query.lower() == "exit":  
            break  
  
        results = retrieve_results(query)  
        prompt = create_augmented_prompt(query, results)  
        answer = llm.invoke(prompt)  
  
        print(f"AI: {answer.content}")  
  
  
if __name__ == "__main__":  
    chat()
```


You should have a fully functional RAG application at this point 🎉

### 3.13. Trying out some queries

By default, we have 3 research papers in the docs directory. You can add any other PDFs you would like Gemini to answer questions on. Let's try a few queries.

The screenshots below show 3 questions that were asked based on the uploaded documents and the answers provided by Gemini based on our retrieved documents.

Screenshot 1

<img src="./images/first%20q.png" />


Screenshot 2

<img src="./images/second%20q.png" />


Screenshot 3

<img src="./images/third%20q.png" />


You can also attempt asking a question that is not contained within our documents like in the screenshot below. Due to our prompt, Gemini will not be able to answer it.

<img src="./images/what%20is%20pc.png" />



# 4. Conclusion
That brings us to the end of our tutorial. You can try different settings, prompts, embedding models, vector databases and LLMs to play around with RAG and vector databases as well as explore other possibilities. The underlying principle stays the same but you can use a variety of tools and technology to achieve varied results.
