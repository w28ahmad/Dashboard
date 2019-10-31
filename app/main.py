from config import create_app

server = create_app()

if __name__ == '__main__':
    server.run(debug=True, host='0.0.0.0')
