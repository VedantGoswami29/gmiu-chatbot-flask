from flask import Flask, render_template, request
from bot_response import ChatBot

app = Flask(__name__)

# Initialize the chatbot with the path to the PDF
chatbot = ChatBot('book.pdf', 'mistral', 'db')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/getanswer')
def get_answer():
    user_input = request.args.get('msg', '')
    answer = chatbot.askQuestion(user_input)
    return answer

if __name__ == '__main__':
    # Set your custom port number here
    app.run(debug=True, port=8100)