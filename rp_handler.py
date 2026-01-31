#!/usr/bin/env python3
"""
Fixed RunPod Handler for XTTS Voice Cloning
With automatic TOS acceptance
"""

import runpod
import base64
import os
import sys
import traceback
import tempfile
from pathlib import Path

# CRITICAL: Set environment variable to auto-accept Coqui TOS
os.environ['COQUI_TOS_AGREED'] = '1'

# Try to import TTS
try:
    from TTS.api import TTS
    print("[INIT] TTS library imported successfully", flush=True)
except ImportError as e:
    print(f"[ERROR] Failed to import TTS library: {e}", flush=True)
    sys.exit(1)

# Global variable for TTS model
tts_model = None

def initialize_model():
    """Initialize the XTTS model"""
    global tts_model
    
    print("="*70, flush=True)
    print("🎵 XTTS RunPod Handler - Initializing", flush=True)
    print("="*70, flush=True)
    
    try:
        print("[INIT] Loading XTTS v2 model...", flush=True)
        print("[INIT] This may take 30-60 seconds on first run...", flush=True)
        
        # Initialize TTS model (TOS auto-accepted via environment variable)
        tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        
        # Move to GPU if available
        if tts_model.is_cuda_available:
            print("[INIT] CUDA is available, moving model to GPU...", flush=True)
            tts_model = tts_model.to("cuda")
            print("[INIT] ✅ Model loaded on GPU", flush=True)
        else:
            print("[INIT] ⚠️  CUDA not available, using CPU (slower)", flush=True)
        
        print("="*70, flush=True)
        print("✅ XTTS Model loaded successfully!", flush=True)
        print("="*70, flush=True)
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to initialize model: {e}", flush=True)
        traceback.print_exc()
        return False

def handler(event):
    """
    Main handler function for TTS generation
    
    Expected input format:
    {
        "input": {
            "text": "Text to convert to speech (required)",
            "speaker_wav": "Base64 encoded audio for voice cloning (optional)",
            "language": "en" (default, supports: en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, ko, hi)
        }
    }
    
    Returns:
    {
        "audio": "Base64 encoded WAV audio",
        "content_type": "audio/wav",
        "size_bytes": 12345
    }
    """
    
    global tts_model
    
    # Check if model is initialized
    if tts_model is None:
        print("[ERROR] Model not initialized", flush=True)
        return {
            "error": "Model not initialized. Please wait and try again.",
            "status": "model_not_ready"
        }
    
    try:
        # Extract input data
        input_data = event.get("input", {})
        
        # Get required parameter
        text = input_data.get("text", "").strip()
        if not text:
            return {
                "error": "Missing required parameter: 'text'",
                "status": "invalid_input"
            }
        
        # Get optional parameters
        speaker_wav_b64 = input_data.get("speaker_wav", "")
        language = input_data.get("language", "en")
        
        print(f"[REQUEST] Generating TTS", flush=True)
        print(f"[REQUEST] Text length: {len(text)} characters", flush=True)
        print(f"[REQUEST] Language: {language}", flush=True)
        print(f"[REQUEST] Voice cloning: {'Yes' if speaker_wav_b64 else 'No'}", flush=True)
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_output:
            output_path = tmp_output.name
        
        # Handle voice cloning if speaker audio is provided
        speaker_path = None
        if speaker_wav_b64:
            try:
                print("[CLONING] Processing speaker audio...", flush=True)
                
                # Remove data URL prefix if present
                if "," in speaker_wav_b64 and speaker_wav_b64.startswith("data:"):
                    speaker_wav_b64 = speaker_wav_b64.split(",", 1)[1]
                
                # Decode base64 audio
                speaker_bytes = base64.b64decode(speaker_wav_b64)
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_speaker:
                    speaker_path = tmp_speaker.name
                    tmp_speaker.write(speaker_bytes)
                
                print(f"[CLONING] Speaker audio decoded ({len(speaker_bytes)} bytes)", flush=True)
                
            except Exception as e:
                print(f"[ERROR] Failed to process speaker audio: {e}", flush=True)
                return {
                    "error": f"Invalid speaker_wav: {str(e)}",
                    "status": "invalid_speaker_audio"
                }
        
        # Generate speech
        try:
            print("[TTS] Generating audio...", flush=True)
            
            if speaker_path:
                # Voice cloning mode
                tts_model.tts_to_file(
                    text=text,
                    file_path=output_path,
                    speaker_wav=speaker_path,
                    language=language
                )
            else:
                # Default voice mode
                tts_model.tts_to_file(
                    text=text,
                    file_path=output_path,
                    language=language
                )
            
            print("[TTS] ✅ Audio generation complete", flush=True)
            
        except Exception as e:
            print(f"[ERROR] TTS generation failed: {e}", flush=True)
            traceback.print_exc()
            return {
                "error": f"TTS generation failed: {str(e)}",
                "status": "generation_failed"
            }
        
        # Read generated audio
        try:
            with open(output_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
            
            # Encode to base64
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            print(f"[SUCCESS] Audio generated: {len(audio_bytes)} bytes", flush=True)
            
            # Cleanup temporary files
            try:
                os.unlink(output_path)
                if speaker_path:
                    os.unlink(speaker_path)
            except:
                pass
            
            return {
                "audio": audio_b64,
                "content_type": "audio/wav",
                "size_bytes": len(audio_bytes),
                "text_length": len(text),
                "language": language,
                "voice_cloned": bool(speaker_wav_b64),
                "status": "success"
            }
            
        except Exception as e:
            print(f"[ERROR] Failed to read output audio: {e}", flush=True)
            return {
                "error": f"Failed to read output: {str(e)}",
                "status": "read_failed"
            }
    
    except Exception as e:
        print(f"[ERROR] Unexpected error in handler: {e}", flush=True)
        traceback.print_exc()
        return {
            "error": f"Handler error: {str(e)}",
            "status": "handler_error"
        }

# Main execution
if __name__ == "__main__":
    print("\n" + "="*70, flush=True)
    print("🚀 Starting XTTS RunPod Serverless Handler", flush=True)
    print("="*70 + "\n", flush=True)
    
    # Initialize the model
    if not initialize_model():
        print("[FATAL] Model initialization failed. Exiting...", flush=True)
        sys.exit(1)
    
    print("\n" + "="*70, flush=True)
    print("🎉 Handler is ready and waiting for requests!", flush=True)
    print("="*70 + "\n", flush=True)
    
    # Start RunPod serverless handler
    try:
        runpod.serverless.start({"handler": handler})
    except Exception as e:
        print(f"[FATAL] Failed to start RunPod handler: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
