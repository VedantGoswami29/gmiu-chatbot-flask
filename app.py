from flask import Flask, render_template, request
from bot_response import PDFChatbot

app = Flask(__name__)

# Initialize the chatbot with the path to the PDF
chatbot = PDFChatbot(pdf_path='book.pdf')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/getanswer')
def get_answer():
    user_input = request.args.get('msg', '')
    answer = chatbot.get_answer(user_input)
    return answer

if __name__ == '__main__':
    # Set your custom port number here
    app.run(debug=True, port=8100)