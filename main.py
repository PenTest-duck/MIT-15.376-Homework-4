import whisper
from kokoro import KPipeline
import cv2
import sounddevice as sd
from openai import OpenAI
from scipy.io.wavfile import write
import numpy as np
import base64
from threading import Thread
import queue
from dotenv import load_dotenv

load_dotenv()

# Initialize models
client = OpenAI()
whisper_model = whisper.load_model("base")
pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")

# Audio recording settings
SAMPLE_RATE = 16000
audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    """Callback for audio recording"""
    if status:
        print(status)
    audio_queue.put(indata.copy())

def tts(text: str):
    """Text-to-speech using Kokoro"""
    generator = pipeline(text=text, voice="af_heart")
    for i, (gs, ps, audio) in enumerate(generator):
        # Play audio directly instead of saving
        sd.play(audio, samplerate=24000)
        sd.wait()

def encode_image_to_base64(image):
    """Encode OpenCV image to base64 data URI"""
    _, buffer = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"

def draw_text_on_frame(frame, text, color=(0, 255, 0)):
    """Draw status text on top right of frame"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Position: top right with padding
    x = frame.shape[1] - text_width - 20
    y = text_height + 20
    
    # Draw background rectangle for better visibility
    cv2.rectangle(frame, (x - 10, y - text_height - 10), 
                  (x + text_width + 10, y + baseline + 10), 
                  (0, 0, 0), -1)
    
    # Draw text
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)
    
    return frame

def process_audio_and_respond(audio_stream, captured_image, state_dict):
    """Process audio and get AI response in a separate thread"""
    try:
        # Stop audio recording (moved to thread to not block main loop)
        if audio_stream:
            audio_stream.stop()
            audio_stream.close()
        
        # Collect all audio data
        audio_data = []
        while not audio_queue.empty():
            audio_data.append(audio_queue.get())
        
        if not audio_data:
            print("No audio recorded")
            state_dict["state"] = "idle"
            return
        
        audio_array = np.concatenate(audio_data, axis=0)
        
        # Save audio for Whisper (it needs a file)
        write("AUDIO.wav", SAMPLE_RATE, audio_array)
        print("Audio saved")
        
        # Transcribe audio
        print("Transcribing audio...")
        result = whisper_model.transcribe("AUDIO.wav")
        transcription = result["text"]
        print(f"Transcription: {transcription}")
        
        # Encode image to base64
        image_base64 = encode_image_to_base64(captured_image)
        
        # Send to OpenAI
        print("Sending to GPT...")
        response = client.responses.create(
            model="gpt-5-mini",
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": transcription
                        },
                        {
                            "type": "input_image",
                            "image_url": image_base64,
                        },
                    ],
                },
            ],
            tools=[
                {
                    "type": "web_search",
                    "user_location": {
                        "type": "approximate",
                        "country": "US",
                        "city": "Cambridge",
                        "region": "Massachusetts",
                    },
                },
            ],
            reasoning={
                "effort": "low",
            },
            text={
                "verbosity": "low",
            },
        )
        
        print(f"AI Response: {response.output_text}")
        
        # Update state to speaking
        state_dict["state"] = "speaking"
        
        print("Speaking...")
        tts(response.output_text)
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Return to idle
    state_dict["state"] = "idle"

def main():
    cam = cv2.VideoCapture(0)  # default camera
    if not cam.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Live camera feed started.")
    print("Press SPACE to start/stop recording, ESC to exit.")
    
    state_dict = {"state": "idle"}  # Shared state dict for thread communication
    captured_image = None
    audio_stream = None
    processing_thread = None
    
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Draw status text based on state
        display_frame = frame.copy()
        current_state = state_dict["state"]
        if current_state == "listening":
            display_frame = draw_text_on_frame(display_frame, "Listening...", (0, 255, 255))
        elif current_state == "processing":
            display_frame = draw_text_on_frame(display_frame, "Thinking...", (0, 165, 255))
        elif current_state == "speaking":
            display_frame = draw_text_on_frame(display_frame, "Speaking...", (255, 0, 255))
        else:  # idle state
            display_frame = draw_text_on_frame(display_frame, "Press SPACE to talk", (0, 255, 0))
        
        cv2.imshow("Live Camera Feed", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(" "):
            if state_dict["state"] == "idle":
                # Start recording
                print("Starting recording...")
                state_dict["state"] = "listening"
                
                # Capture photo
                captured_image = frame.copy()
                print("Photo captured")
                
                # Start audio recording
                audio_queue.queue.clear()
                audio_stream = sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    callback=audio_callback
                )
                audio_stream.start()
                
            elif state_dict["state"] == "listening":
                # Stop recording and process
                print("Stopping recording...")
                state_dict["state"] = "processing"
                
                # Start processing in a separate thread (including audio stream cleanup)
                processing_thread = Thread(
                    target=process_audio_and_respond,
                    args=(audio_stream, captured_image, state_dict),
                    daemon=True
                )
                processing_thread.start()
                
                # Clear the reference since thread will handle cleanup
                audio_stream = None
        
        elif key == 27: # ESC
            print("Exiting...")
            break
    
    # Cleanup
    if audio_stream and audio_stream.active:
        audio_stream.stop()
        audio_stream.close()
    
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()