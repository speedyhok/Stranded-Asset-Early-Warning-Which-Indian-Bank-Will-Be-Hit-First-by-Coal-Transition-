# Python Web Service entry point for Render
# Copyright (c) 2026 Mohibul Hoque
# Email: hokworks@gmail.com
# LinkedIn: linkedin.com/in/speedymohibul
# License: MIT License

from flask import Flask, send_from_directory, jsonify
import os

base_dir = os.path.abspath(os.path.dirname(__file__))
static_dir = os.path.join(base_dir, 'dashboard', 'dist')
app = Flask(__name__, static_folder=static_dir, static_url_path='')

@app.route('/api/health')
def health():
    return jsonify({"status": "healthy", "service": "climate-contagion-model"})

# Serve React static assets
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if not os.path.exists(app.static_folder):
        return f"""
        <html>
            <head><title>Frontend Build Missing</title></head>
            <body style="font-family: sans-serif; padding: 40px; text-align: center; color: #333; line-height: 1.6;">
                <h1 style="color: #ef4444;">React Frontend Build Not Found</h1>
                <p>The Flask backend is running successfully, but the compiled React files were not found.</p>
                <p><strong>To resolve this on Render:</strong></p>
                <p>Verify that your <strong>Build Command</strong> is configured exactly as follows under your Render Service Settings:</p>
                <pre style="background: #f4f4f5; padding: 12px; border-radius: 6px; display: inline-block; text-align: left; font-family: monospace;">cd dashboard && npm install && npm run build && cd .. && pip install -r requirements.txt</pre>
            </body>
        </html>
        """, 404

    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        if not os.path.exists(os.path.join(app.static_folder, 'index.html')):
            return "index.html not found in static folder", 404
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    # Render binds the application to the PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
