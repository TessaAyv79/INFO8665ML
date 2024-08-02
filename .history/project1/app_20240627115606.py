from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    # templates klasöründeki exp4.html dosyasını render et
    return render_template('exp4.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)