# Sample Audio Files

This directory contains sample audio files for testing the Speech Analysis Tool.

## Adding a Real WAV File

The current `test.wav` is just a placeholder text file. For proper functionality, replace it with a real WAV audio file:

1. Obtain a WAV file with speech content (mono, 16-bit, 16kHz recommended for best results)
2. Rename it to `test.wav`
3. Place it in this directory (`static/samples/`)
4. Restart the application

## Requirements for Sample Files

For optimal analysis, sample files should:
- Be in WAV format
- Contain clear speech in English
- Be ideally 10-30 seconds in length
- Have minimal background noise
- Be recorded at 16kHz sample rate (mono)

This will ensure the most accurate analysis when users click the "Try With Sample Audio" button on the main page. 