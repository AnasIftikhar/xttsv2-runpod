#!/usr/bin/env python3
"""
RunPod Serverless Handler for XTTS API Server
This handler starts the XTTS server and proxies requests to it
"""

import runpod
import subprocess
import time
import requests
import os
import base64
from threading import Thread

# Global variables
server_process = None
server_ready = False

def start_xtts_server():
    """Start XTTS API server in background"""
    global server_process, server_ready
    
    print("🚀 Starting XTTS API server...")
    
    cmd = [
        "python3", "-m", "xtts_api_server",
        "--host", "0.0.0.0",
        "--port", "8020",
        "--use-cache",
        "--deepspeed"
    ]
    
    try:
        server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Monitor server startup
        max_wait = 90
        waited = 0
        
        while waited < max_wait:
            try:
                response = requests.get("http://localhost:8020/docs", timeout=3)
                if response.status_code == 200:
                    server_ready = True
                    print("✅ XTTS API server is ready!")
                    return True
            except:
                pass
            
            time.sleep(3)
            waited += 3
            
            if waited % 15 == 0:
                print(f"⏳ Waiting for server... ({waited}s/{max_wait}s)")
        
        print("❌ Server failed to start within timeout")
        return False
        
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False

def handler(event):
    """
    Main RunPod handler function
    
    Input format:
    {
        "input": {
            "text": "Text to convert to speech",
            "speaker_wav": "URL or path to speaker voice sample (optional)",
            "language": "en" (default),
            "temperature": 0.75,
            "speed": 1.0
        }
    }
    
    Returns audio as base64 encoded string
    """
    global server_ready
    
    # Check if server is ready
    if not server_ready:
        return {
            "error": "Server is still initializing. Please wait a moment and try again.",
            "server_ready": False
        }
    
    try:
        # Extract input
        input_data = event.get("input", {})
        
        # Validate required fields
        text = input_data.get("text", "").strip()
        if not text:
            return {"error": "Missing required parameter: 'text'"}
        
        # Build request payload with defaults
        payload = {
            "text": text,
            "speaker_wav": input_data.get("speaker_wav", ""),
            "language": input_data.get("language", "en"),
            "temperature": input_data.get("temperature", 0.75),
            "length_penalty": input_data.get("length_penalty", 1.0),
            "repetition_penalty": input_data.get("repetition_penalty", 5.0),
            "top_k": input_data.get("top_k", 50),
            "top_p": input_data.get("top_p", 0.85),
            "speed": input_data.get("speed", 1.0)
        }
        
        print(f"🎤 Generating TTS for: '{text[:50]}...'")
        
        # Call XTTS API
        response = requests.post(
            "http://localhost:8020/tts_to_audio/",
            json=payload,
            timeout=120
        )
        
        if response.status_code == 200:
            # Encode audio to base64
            audio_base64 = base64.b64encode(response.content).decode('utf-8')
            
            print(f"✅ Audio generated successfully ({len(response.content)} bytes)")
            
            return {
                "audio": audio_base64,
                "content_type": response.headers.get('content-type', 'audio/wav'),
                "text_length": len(text),
                "audio_size_bytes": len(response.content)
            }
        else:
            return {
                "error": f"TTS generation failed (HTTP {response.status_code})",
                "details": response.text[:500]
            }
            
    except requests.Timeout:
        return {"error": "Request timeout. Text might be too long."}
    except Exception as e:
        return {"error": f"Handler error: {str(e)}"}

# Initialize
if __name__ == "__main__":
    print("=" * 60)
    print("🎵 XTTS RunPod Serverless Handler")
    print("=" * 60)
    
    # Start server in background thread
    server_thread = Thread(target=start_xtts_server, daemon=True)
    server_thread.start()
    
    # Wait for server with timeout
    server_thread.join(timeout=120)
    
    if not server_ready:
        print("❌ CRITICAL: Server failed to initialize")
        print("Check logs above for errors")
        exit(1)
    
    print("\n" + "=" * 60)
    print("🎉 Handler ready - starting RunPod serverless")
    print("=" * 60 + "\n")
    
    # Start RunPod handler
    runpod.serverless.start({"handler": handler})
