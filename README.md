# Speech Analysis Tool for Cognitive Assessment

This application analyzes speech patterns from audio recordings to identify potential cognitive issues. It extracts various speech features and uses basic machine learning techniques to assess risk levels.

## Features

1. **Audio Processing**
   - Converts WAV files to text using speech recognition
   - Analyzes audio characteristics like pitch, pauses, and speech rate

2. **Speech Feature Extraction**
   - Pauses per sentence
   - Hesitation markers (uh, um, etc.)
   - Word recall issues detection
   - Speech rate calculation
   - Pitch variability analysis
   - Sentence completion assessment

3. **Machine Learning Analysis**
   - Basic unsupervised approach to detect abnormal patterns
   - Risk scoring based on speech features
   - Identification of potential cognitive issues

## Technology Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript
- **Audio Processing**: librosa, pydub, SpeechRecognition
- **NLP**: NLTK
- **Machine Learning**: scikit-learn

## Setup Instructions

### Prerequisites
- Python 3.7+
- pip package manager

### Installation

1. Clone the repository or download the source code

2. Create and activate a virtual environment (optional but recommended):
   ```
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
   
   Note: Some of the audio processing libraries may require additional system dependencies:
   - On Windows: you might need Visual C++ Build Tools
   - On Linux: `sudo apt-get install python3-dev libportaudio2 portaudio19-dev ffmpeg`
   - On macOS: `brew install portaudio ffmpeg`

4. Download NLTK data (automatically done when running the app, but can be done manually):
   ```python
   import nltk
   nltk.download('punkt')
   ```

### Running the Application

1. Start the Flask application:
   ```
   python app.py
   ```

2. Open a web browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

3. Use the interface to upload WAV audio files and analyze speech patterns.

## Usage

1. Record a speech sample in WAV format (mono, 16-bit, 16kHz recommended for best results)
2. Upload the file using the web interface
3. Wait for processing (may take a moment depending on file length)
4. Review the detailed analysis of speech features
5. Check the risk assessment for potential cognitive issues

## How It Works

1. **Audio Processing**:
   - The uploaded WAV file is processed to extract the audio waveform
   - Speech is transcribed to text using Google's Speech Recognition API

2. **Feature Extraction**:
   - The system analyzes both the audio signal and the transcribed text
   - Features like pauses, hesitations, and speech rate are calculated

3. **Analysis**:
   - Extracted features are compared against typical patterns
   - A risk score is calculated based on deviations from expected patterns
   - Abnormal features are highlighted for review

## Limitations

- Currently only supports WAV audio files
- Speech recognition works best with clear audio and standard accents
- The unsupervised ML approach is basic and would benefit from training with labeled data
- Not a clinical diagnostic tool - results should be interpreted by qualified professionals

## Future Improvements

- Support for more audio formats
- Advanced ML models with proper training data
- More sophisticated feature extraction
- Longitudinal analysis to track changes over time
- Mobile app support

## License

This project is made by Srijit Sardar of B.Tech ECE dept. of BIT MESRA.

## Disclaimer

This tool is for research and screening purposes only. It is not a medical device and should not be used for clinical diagnosis. Always consult healthcare professionals for proper evaluation and diagnosis of cognitive conditions. 