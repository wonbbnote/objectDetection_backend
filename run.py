from main import app #main 폴더에서 app import

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)