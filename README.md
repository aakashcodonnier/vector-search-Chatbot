
# Vector Search Chatbot with SQL Database



## How It Works

### 1. Data Scraping
- The scraper extracts content from specified websites (currently configured for case studies)
- Extracts title, content, URL, date, author, and categories
- Creates vector embeddings using the `all-MiniLM-L6-v2` model
- Stores everything in MySQL database

### 2. Vector Storage
- Content is stored in MySQL with vector embeddings as JSON
- Uses Sentence Transformers for semantic encoding
- Cosine similarity for semantic search

### 3. Chat Interface
- User submits a question
- Question is converted to vector embedding
- System finds top 2 most semantically similar articles
- References are extracted and sent to LLaMA2 for response generation
- Returns AI-generated answer with source references

## Components

### Backend (`backend/main.py`)
- FastAPI server with `/chat` endpoint
- Handles vector similarity search
- Integrates with Ollama for LLaMA2 inference
- Comprehensive error handling and logging

### Database (`database/db.py`)
- MySQL connection management
- Context manager for proper connection handling

### Scraper (`scraper/scrape_and_embed.py`)
- Web scraping with BeautifulSoup
- Vector embedding generation
- SQL storage with embeddings

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Set up MySQL database named `case_studies_db`
3. Update database credentials in `database/db.py`
4. Install Ollama and pull the `llama2` model
5. Run the scraper: `python scraper/scrape_and_embed.py`
6. Start the server: `uvicorn backend.main:app --reload` Backend ChatBot API Run
7. Open http://127.0.0.1:8000/docs




## Usage

1. Run the server
2. Send POST requests to `/chat` with a `question` parameter
3. Receive AI-generated responses with source references

## API Endpoints

- `GET /` - Health check
- `POST /chat` - Chat endpoint (accepts JSON with `question` field)
- `GET /docs` - Interactive API documentation


# EXAMPLE 
1. 
Request:
{
  "question": "lung cancer case study"
}


Response:

{
  "answer": "...",
  "references": [
    "Article Title 1",
    "Article Title 2"
  ]
}


2. Request:
{
  "question": "Ovarian cancer"
}

Response:
{
  "answer": "Based on the data provided, there is no information available on the effectiveness of an alkaline lifestyle and diet in treating ovarian cancer. The data focuses primarily on lung cancer and brain cancer, with some information on the preventability of lung cancer through lifestyle and diet changes. Therefore, I must respond with \"Not found in database\" for ovarian cancer.",
  "references": [
    "Is There a Cure for Brain Cancer?",
    "Can You Prevent or Reverse Any Cancerous Condition With Lifestyle and Diet?"
  ]
}

3. Request:
{
  "question": "Breast cancer"
}


Response:
{
  "answer": "Based on the data provided, there is a correlation between eating red meat and an increased risk of breast cancer in premenopausal women. According to the study in the British Medical Journal, each additional serving of red meat per day was associated with a 13% increased risk in breast cancer. Therefore, it can be concluded that eating red meat may increase the risk of breast cancer in these women.\n\nHowever, it is important to note that this study only found an association between red meat consumption and breast cancer, and did not establish a cause-and-effect relationship. Further research is needed to confirm these findings and to better understand the underlying mechanisms.\n\nIn terms of a cure for brain tumors, the answer is more complex. While some people have been able to achieve remission through alternative treatments such as the pH Miracle Alkaline Lifestyle and Diet, as described in your answer, there is currently no definitive cure for brain tumors. Surgery, radiation therapy, and chemotherapy are common treatment options for brain cancer, but their effectiveness can vary depending on the type and location of the tumor.\n\nIt's important to consult with a medical professional for proper diagnosis and treatment if you or someone you know is experiencing symptoms of breast cancer or brain cancer.",
  "references": [
    "Eating Meat Increases Risk for Cancer!",
    "Is There a Cure for Brain Cancer?"
  ]
}


4. Request:
{
  "question": "Brain cancer case study"
}


Response:

{
  "answer": "Based on the data provided, there is a brain cancer case study of a patient named Ashley who followed the pH Miracle Alkaline Lifestyle and Diet with success. Ashley was 21 years old when she had a seizure while at the movie theater and was rushed to the hospital. Despite being pregnant and married, Ashley elected not to have brain surgery, radiation, or chemotherapy and instead followed the alkaline lifestyle and diet recommended by Robert Young, a research scientist at the pH Miracle Center. Ashley has now been in remission from brain cancer for over 15 years.\n\nNot found in database. There is no additional information provided in the data about other brain cancer case studies or the effectiveness of the pH Miracle Alkaline Lifestyle and Diet in treating brain cancer.",
  "references": [
    "Is There a Cure for Brain Cancer?",
    "Glenn Stone’s pH Miracle Story – Reversing Cancer NOW!"
  ]
}


# Question ...........

1. { "question": "What is ovarian cancer?" }


2. { "question": "Is cancer reversible naturally?" }


3. { "question": "Cancer prevention through diet" }


4. { "question": "Breast cancer alkaline diet" }
