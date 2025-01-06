# ChatBot based on Ollama
### Clone this repository
`git clone https://github.com/VedantGoswami29/gmiu-chatbot-flask`

### Download Ollama in your local machine from below link, if you haven't installed on your machine
[Download Ollama](https://ollama.com/download)

### Make sure start Ollama local server on your machine
`ollama serve`

### Pull Ollama model that you wish
`ollama pull <model_name>`

As Example, **llama3**, **mistral**, etc. You can checkout from [Ollama Librabry](https://ollama.com/library)

### Installing Requirements
`pip install -r requirements.txt`

### Pull "mxbai-embed-large" your machine
`ollama pull mxbai-embed-large`

### Now initializing your ChatBot (If you have .txt file)
`python3 embedding_genrator.py`

Here **file_name** is data that you want make ChatBot.
This may take few minutes to execute depending on your hardware.
### Start Flask app
`python3 app.py`