# MIT-15.376-Homework-4

Video conversational agent with STT (Whisper), TTS (Kokoro), OpenAI with web search.

## Implementation

STT: Whisper (base)
TTS: Kokoro 87M
LLM: gpt-5-mini with web search tool
Webcam: OpenCV

1. The OpenCV uses the webcam to display the live feed and the current state.
2. I can press space to have OpenCV take a photo and kick off the audio recording.
3. When I press space again, the audio is saved as a WAV file, then Whisper is used to transcribe it into text.
4. The text and the image are sent to OpenAI to generate a response.
5. Kokoro reads out the completed response.
6. Repeat.

The demo video shows an example asking "what can you see in the screen?", to which the image input of gpt-5-mini is able to answer well.
I then ask "considering the weather, what should I wear?", to which gpt-5-mini uses its web search tool to pull the current weather data and answer my question.

## Insights

* Local Whisper and Kokoro are terribly slow, but they did get the job done.
* If I was genuinely building a conversational agent, I would use speech-to-speech realtime agents (e.g. OpenAI realtime, ElevenLabs)
* I need to instruct the LLM to be mindful of the fact that its response will be pronounced (e.g. avoid symbols) and keep it short
* It is incredibly easy for any programmer to build a multimodal conversational agent. But it is incredibly difficult to make it fast, reliable and accurate.
