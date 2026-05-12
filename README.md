# Minimal RAG Pipeline

This project loads documents, chunks them, embeds the chunks, stores them in Pinecone, retrieves the most similar chunks for a query, and asks an LLM to answer from that context.

## Setup

1. Create and activate a Python environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set environment variables locally:
   ```bash
   GROQ_API_KEY=your_groq_key
   PINECONE_API_KEY=your_pinecone_key
   PINECONE_INDEX_NAME=rag-minimal
   PINECONE_CLOUD=aws
   PINECONE_REGION=us-east-1
   GROQ_CHAT_MODEL=llama-3.3-70b-versatile
   EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
   ```

## Use

Index documents:

```bash
python rag_pipeline.py index ./docs
```

Ask a question:

```bash
python rag_pipeline.py query "What does the policy say about refunds?"
```

Run the voice agent:

```bash
python voice_agent.py --duration 5 --countdown 3
```

If you launch through `conda run`, use `--no-capture-output` so the countdown and recording logs appear live:

```bash
conda run --no-capture-output -n rag python -u voice_agent.py --duration 5 --countdown 3
```

To hear the prompts and answer out loud, add `--tts`:

```bash
python voice_agent.py --duration 5 --countdown 3 --tts
```

The app will use your ElevenLabs key from `ELEVENLABS_API_KEY` or `ELEVEN_LABS_API_KEY`, and it defaults to the public `JBFqnCBsd6RMkjVDRZzb` voice unless you override it with `--tts-voice-id`.

You can also choose a different ElevenLabs model or output format:

```bash
python voice_agent.py --duration 5 --countdown 3 --tts --tts-model-id eleven_flash_v2_5 --tts-output-format mp3_44100_128
```

If the wrong audio source is selected, list devices and choose a real input device:

```bash
python voice_agent.py --list-devices
python voice_agent.py --duration 5 --countdown 3 --input-device 1
```

Use a microphone input for your voice. Use `Stereo Mix` only if you want to capture system audio coming from your speakers.

For debugging, you can print the retrieved chunks before the answer:

```bash
python rag_pipeline.py query "What does the paper say about hybrid RAG?" --show-context
```

## Supported files

- `.txt`
- `.md`
- `.pdf`
- `.docx`
- `.csv`
