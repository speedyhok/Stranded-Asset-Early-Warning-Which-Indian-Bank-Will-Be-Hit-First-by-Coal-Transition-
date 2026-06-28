# Python Web Service entry point for Render
# Copyright (c) 2026 Mohibul Hoque
# Email: hokworks@gmail.com
# LinkedIn: linkedin.com/in/speedymohibul
# License: MIT License

from flask import Flask, send_from_directory, jsonify
import os

app = Flask(__name__, static_folder='dashboard/dist', static_url_path='')

@app.route('/api/health')
def health():
    return jsonify({"status": "healthy", "service": "climate-contagion-model"})

# Serve React static assets
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    # Render binds the application to the PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
