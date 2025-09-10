#!/usr/bin/env python3
"""
Simple HTTP Server for Premier League Predictor Web Interface
============================================================
Serves the HTML webpage and JSON data for the Premier League predictor.
"""

import http.server
import socketserver
import os
import webbrowser
from threading import Timer

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers to allow local file access
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def open_browser():
    """Open the web browser after a short delay."""
    webbrowser.open('http://localhost:8000')

def main():
    """Start the HTTP server."""
    PORT = 8000
    
    # Change to the directory containing the files
    os.chdir('/home/joost/Premier League Prediction Model')
    
    # Create server
    with socketserver.TCPServer(("", PORT), CustomHTTPRequestHandler) as httpd:
        print(f"🌐 Premier League Predictor Server")
        print(f"📡 Server running at http://localhost:{PORT}")
        print(f"📁 Serving files from: {os.getcwd()}")
        print(f"🔗 Open your browser and go to: http://localhost:{PORT}")
        print(f"⏹️  Press Ctrl+C to stop the server")
        print("=" * 60)
        
        # Open browser after 1 second
        Timer(1.0, open_browser).start()
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print(f"\n🛑 Server stopped")
            httpd.shutdown()

if __name__ == "__main__":
    main()
