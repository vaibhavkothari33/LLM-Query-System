# 🤖 LLM-Powered Intelligent Query-Retrieval System

A sophisticated document query system that processes insurance, legal, HR, and compliance documents using semantic search and LLM-powered reasoning. This system provides intelligent document analysis with explainable decisions and clause matching.

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Navigate to project directory
cd "D:\Desktop\Code Folder\Artifical Intelligense\Bajaj"

# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run setup script
python setup.py
```

### 2. Add Documents
Place your PDF, DOCX, or email files in the `./documents` folder:
```
documents/
├── policy_document.pdf
├── contract.docx
├── employee_handbook.pdf
└── compliance_guidelines.docx
```

### 3. Start the API Server
```bash
python main.py --server --port 8001
```

### 4. Access Interactive Documentation
Open your browser and go to: **http://localhost:8001/docs**

## 📚 System Architecture

### Core Components

1. **Document Processor**: Handles PDF, DOCX, and email file ingestion
2. **Vector Search Engine**: Uses FAISS for semantic search with embeddings
3. **LLM Query Processor**: Generates intelligent responses using context
4. **Scoring System**: Calculates confidence and relevance scores
5. **REST API**: FastAPI-based interface with authentication

### Technology Stack
- **Backend**: FastAPI, Python 3.8+
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **Document Processing**: PyPDF2, python-docx, BeautifulSoup
- **Authentication**: Bearer Token
- **Documentation**: Swagger UI (Auto-generated)

## 🔌 API Endpoints

### Base URL
```
http://localhost:8001
```

### Authentication
All endpoints (except health check) require Bearer token authentication:
```
Authorization: Bearer 36d49ac587c7cb7331f48ad3067cd8057811970de89b734f8326aa39d665c8c9
```

### 1. Health Check
**GET** `/api/v1/health`

Check if the system is operational.

**Response:**
```json
{
  "status": "healthy",
  "message": "Query retrieval system is operational"
}
```

**Example:**
```bash
curl http://localhost:8001/api/v1/health
```

### 2. System Statistics
**GET** `/api/v1/stats`

Get system statistics including document chunks and configuration.

**Headers:**
```
Authorization: Bearer 36d49ac587c7cb7331f48ad3067cd8057811970de89b734f8326aa39d665c8c9
```

**Response:**
```json
{
  "total_document_chunks": 180,
  "embedding_model": "all-MiniLM-L6-v2",
  "vector_dimension": 384,
  "similarity_threshold": 0.3
}
```

### 3. Process Query
**POST** `/api/v1/query`

Process a natural language query against the document database.

**Headers:**
```
Authorization: Bearer 36d49ac587c7cb7331f48ad3067cd8057811970de89b734f8326aa39d665c8c9
Content-Type: application/json
```

**Request Body:**
```json
{
  "query": "What are the key skills and experience mentioned?",
  "document_types": ["pdf", "docx"],
  "max_results": 5
}
```

**Parameters:**
- `query` (string, required): Natural language query
- `document_types` (array, optional): Filter by document types ["pdf", "docx", "email"]
- `max_results` (integer, optional): Maximum number of results (default: 5)

**Response:**
```json
{
  "query": "What are the key skills and experience mentioned?",
  "answer": "Based on the provided documents, the key skills include...",
  "confidence": 0.85,
  "matched_clauses": [
    {
      "clause_id": "resume_page1_chunk0",
      "content": "Experienced software engineer with 5+ years...",
      "confidence": 0.92,
      "document_source": "vaibhavkothari_resume",
      "page_number": 1
    }
  ],
  "reasoning": "I found relevant information in the resume document...",
  "processing_time": 0.245,
  "token_usage": {
    "prompt_tokens": 150,
    "completion_tokens": 75,
    "total_tokens": 225
  }
}
```

### 4. Webhook Endpoint
**POST** `/api/v1/webhook`

Receive webhook notifications for query processing events.

**Headers:**
```
Authorization: Bearer 36d49ac587c7cb7331f48ad3067cd8057811970de89b734f8326aa39d665c8c9
Content-Type: application/json
```

**Request Body:**
```json
{
  "event_type": "query_processed",
  "query_id": "q_12345",
  "query": "What are the key skills?",
  "status": "completed",
  "timestamp": "2024-01-15T10:30:00Z",
  "metadata": {
    "processing_time": 0.245,
    "confidence": 0.85,
    "chunks_retrieved": 3
  }
}
```

**Response:**
```json
{
  "status": "received",
  "message": "Webhook processed successfully",
  "webhook_id": "wh_67890"
}
```

## 🖥️ Using Interactive Documentation (/docs)

### Accessing the Docs
1. Start your server: `python main.py --server --port 8001`
2. Open browser: http://localhost:8001/docs
3. You'll see the Swagger UI interface

### Testing Custom Queries

#### Step 1: Authenticate
1. Click the **"Authorize"** button (🔒) at the top right
2. Enter your API key: `36d49ac587c7cb7331f48ad3067cd8057811970de89b734f8326aa39d665c8c9`
3. Click **"Authorize"**

#### Step 2: Test Health Check
1. Find the **GET** `/api/v1/health` endpoint
2. Click **"Try it out"**
3. Click **"Execute"**
4. You should see a 200 response with status "healthy"

#### Step 3: Test Query Processing
1. Find the **POST** `/api/v1/query` endpoint
2. Click **"Try it out"**
3. Enter your custom query in the request body:

```json
{
  "query": "What is the educational background mentioned?",
  "max_results": 3
}
```

4. Click **"Execute"**
5. Review the response with answer, confidence, and reasoning

#### Step 4: Test System Stats
1. Find the **GET** `/api/v1/stats` endpoint
2. Click **"Try it out"**
3. Click **"Execute"**
4. Check the total document chunks and system configuration

### Example Custom Queries to Try

```json
{
  "query": "What are the technical skills listed?",
  "max_results": 5
}
```

```json
{
  "query": "What is the work experience?",
  "document_types": ["pdf"],
  "max_results": 3
}
```

```json
{
  "query": "What are the responsibilities mentioned?",
  "max_results": 4
}
```

## 🔧 CLI Commands

### Server Mode
```bash
# Start API server
python main.py --server --port 8000

# Start on different port if 8000 is busy
python main.py --server --port 8002
```

### CLI Mode
```bash
# Process single query
python main.py --documents ./documents --query "What are the key skills?"

# Interactive mode
python main.py --documents ./documents

# Rebuild index from scratch
python main.py --documents ./documents --rebuild
```

### Testing
```bash
# Run automated tests
python test_api.py

# Test specific functionality
python setup.py
```

## 📊 System Configuration

### Configuration File: `config.py`

```python
@dataclass
class Config:
    # Vector database settings
    embedding_model: str = "all-MiniLM-L6-v2"
    vector_dimension: int = 384
    faiss_index_path: str = "./data/faiss_index.bin"
    document_store_path: str = "./data/documents.pkl"
    
    # LLM settings
    max_context_length: int = 4000
    max_response_tokens: int = 500
    temperature: float = 0.1
    
    # Retrieval settings
    top_k_documents: int = 5
    similarity_threshold: float = 0.3
    
    # API settings
    api_key: str = "36d49ac587c7cb7331f48ad3067cd8057811970de89b734f8326aa39d665c8c9"
```

### Environment Variables
Create a `.env` file for production:
```env
API_KEY=your_production_api_key
OPENAI_API_KEY=your_openai_key
EMBEDDING_MODEL=all-MiniLM-L6-v2
SIMILARITY_THRESHOLD=0.3
```

## 🔄 Webhook Integration

### Setting Up Webhooks

The system supports webhook notifications for various events:

#### 1. Query Processing Events
```json
{
  "event_type": "query_processed",
  "query_id": "q_12345",
  "query": "What are the key skills?",
  "status": "completed",
  "timestamp": "2024-01-15T10:30:00Z",
  "metadata": {
    "processing_time": 0.245,
    "confidence": 0.85,
    "chunks_retrieved": 3,
    "answer_length": 150
  }
}
```

#### 2. System Events
```json
{
  "event_type": "system_startup",
  "timestamp": "2024-01-15T10:00:00Z",
  "metadata": {
    "total_chunks": 180,
    "model_loaded": "all-MiniLM-L6-v2",
    "index_status": "loaded"
  }
}
```

#### 3. Error Events
```json
{
  "event_type": "query_error",
  "query_id": "q_12345",
  "error": "No relevant documents found",
  "timestamp": "2024-01-15T10:30:00Z",
  "metadata": {
    "query": "What are the key skills?",
    "similarity_threshold": 0.3
  }
}
```

### Webhook Configuration
Add webhook URLs to your configuration:
```python
webhook_urls = [
    "https://your-app.com/webhook",
    "https://slack.com/api/webhooks/your-slack-webhook"
]
```

## 📁 Project Structure

```
Bajaj/
├── main.py                 # Main application file
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── README.md             # This documentation
├── setup.py              # Setup and validation script
├── test_api.py           # API testing script
├── evaluation_framework.py # Evaluation and testing framework
├── documents/            # Document storage
│   ├── vaibhavkothari_resume.pdf
│   └── other_documents.pdf
├── data/                 # Generated data
│   ├── faiss_index.bin   # FAISS vector index
│   └── documents.pkl     # Document chunks
└── venv/                 # Virtual environment
```

## 🧪 Testing

### Automated Tests
```bash
# Run all tests
python test_api.py

# Test specific endpoints
curl -X GET http://localhost:8001/api/v1/health
curl -X GET http://localhost:8001/api/v1/stats \
  -H "Authorization: Bearer 36d49ac587c7cb7331f48ad3067cd8057811970de89b734f8326aa39d665c8c9"
```

### Manual Testing
1. Use the interactive docs at http://localhost:8001/docs
2. Test with different query types
3. Verify confidence scores and reasoning
4. Check processing times

## 🚀 Production Deployment

### 1. Environment Setup
```bash
# Install production dependencies
pip install gunicorn uvicorn[standard]

# Set environment variables
export API_KEY=your_production_key
export OPENAI_API_KEY=your_openai_key
```

### 2. Start Production Server
```bash
# Using Gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8001

# Using Uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8001 --workers 4
```

### 3. Reverse Proxy (Nginx)
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 🔒 Security Considerations

1. **API Key**: Change the default API key in production
2. **HTTPS**: Use SSL/TLS in production
3. **Rate Limiting**: Implement rate limiting for API endpoints
4. **Input Validation**: All inputs are validated using Pydantic
5. **CORS**: Configure CORS settings for your domain

## 📈 Monitoring and Logging

### Log Levels
- **INFO**: System operations, query processing
- **WARNING**: Missing documents, low confidence
- **ERROR**: Processing failures, API errors

### Metrics to Monitor
- Query processing time
- Confidence scores
- Token usage
- Error rates
- System uptime

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Check what's using the port
   netstat -ano | findstr :8000
   
   # Use different port
   python main.py --server --port 8001
   ```

2. **Documents Not Found**
   ```bash
   # Check documents directory
   ls documents/
   
   # Rebuild index
   python main.py --documents ./documents --rebuild
   ```

3. **Authentication Errors**
   - Verify API key is correct
   - Check Authorization header format
   - Ensure token is not expired

4. **Low Confidence Scores**
   - Add more relevant documents
   - Adjust similarity threshold in config
   - Improve query specificity

### Getting Help

1. Check the logs for error messages
2. Verify all dependencies are installed
3. Test with the interactive documentation
4. Run the test suite: `python test_api.py`

---

**Happy Querying! 🎉**
