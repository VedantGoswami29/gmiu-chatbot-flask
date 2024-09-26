# ChatBot based on Ollama
### Pull this repository
`git pull https://github.com/VedantGoswami29/gmiu-chatbot-flask`

### Download Ollama in your local machine from below link, if you haven't installed on your machine
[Download Ollama](https://ollama.com/download)

### Clone Ollama model that you wish
`ollama clone <model_name>`

As Example, **llama3**, **mistral**, etc. You can checkout from [Ollama Librabry](https://ollama.com/library)

### Installing Requirements
`pip install -r requirements.txt`

### Make sure start Ollama local server on your machine
`ollama serve`

### Now initializing your ChatBot 
`python3 bot_response.py <book_name> <model_name> <db_name>`

This may take few minutes to execute depending on your hardware.
### Start Flask app
`python3 app.py`