from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import uuid
import json
import traceback
from werkzeug.utils import secure_filename
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import speech_recognition as sr
from pydub import AudioSegment
import librosa
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import logging

# Download NLTK data
nltk.download('punkt')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'wav'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['STATIC_FOLDER'] = 'static'

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Function to convert NumPy values to Python types in dictionaries
def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def transcribe_audio(file_path):
    """Convert WAV audio to text using SpeechRecognition"""
    recognizer = sr.Recognizer()
    
    try:
        with sr.AudioFile(file_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return text
    except Exception as e:
        logger.error(f"Error in transcription: {str(e)}")
        return ""

def extract_pauses(file_path, transcription):
    """Extract pauses per sentence"""
    try:
        # Load audio file
        y, sr = librosa.load(file_path)
        
        # More sensitive silence detection - try multiple approaches
        
        # Approach 1: Energy-based silence detection (more sensitive)
        # Calculate the RMS energy
        energy = librosa.feature.rms(y=y)[0]
        
        # Normalize energy to 0-1 range
        energy_norm = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-10)
        
        # Define silence as frames with energy below threshold
        silence_threshold = 0.15  # Lower threshold to detect more pauses
        silent_frames = energy_norm < silence_threshold
        
        # Convert frame-level silence to time segments
        silent_regions = []
        in_silence = False
        current_start = 0
        
        for i, is_silent in enumerate(silent_frames):
            # Start of a silence
            if is_silent and not in_silence:
                current_start = i
                in_silence = True
            # End of a silence
            elif not is_silent and in_silence:
                silent_regions.append((current_start, i))
                in_silence = False
        
        # Add the last silence if still in one
        if in_silence:
            silent_regions.append((current_start, len(silent_frames)))
        
        # Convert frame indices to time and filter short silences
        min_pause_duration = 0.2  # Detect pauses as short as 200ms (more sensitive)
        frame_duration = len(y) / (sr * len(energy))  # Time per energy frame
        
        pauses = []
        for start, end in silent_regions:
            duration = (end - start) * frame_duration
            if duration > min_pause_duration:
                pauses.append(float(duration))
        
        # Backup approach: if first method fails, try the original method with lower threshold
        if not pauses:
            logger.info("Primary pause detection found no pauses, trying backup method")
            non_silent_regions = librosa.effects.split(y, top_db=15)  # Lower top_db for more sensitivity
            
            for i in range(len(non_silent_regions) - 1):
                pause_duration = (non_silent_regions[i+1][0] - non_silent_regions[i][1]) / sr
                if pause_duration > 0.2:  # Lower threshold for more sensitivity
                    pauses.append(float(pause_duration))
        
        # Count sentences
        sentences = sent_tokenize(transcription)
        num_sentences = len(sentences)
        
        # Calculate pauses per sentence
        if num_sentences > 0:
            pauses_per_sentence = len(pauses) / num_sentences
        else:
            pauses_per_sentence = 0.0
            
        return {
            "pauses_per_sentence": float(pauses_per_sentence),
            "total_pauses": len(pauses),
            "average_pause_duration": float(np.mean(pauses) if pauses else 0.0),
            "max_pause_duration": float(max(pauses) if pauses else 0.0)
        }
    except Exception as e:
        logger.error(f"Error extracting pauses: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "pauses_per_sentence": 0.0,
            "total_pauses": 0,
            "average_pause_duration": 0.0,
            "max_pause_duration": 0.0
        }

def detect_hesitation_markers(transcription):
    """Count hesitation markers like 'uh', 'um', etc. and detect non-lexical hesitations"""
    try:
        # No meaningful text to analyze
        if not transcription or len(transcription.strip()) < 3:
            return {
                "total_hesitations": 0,
                "hesitation_types": {},
                "hesitations_per_word": 0.0
            }
        
        # EXPANDED pattern set with many more variations and common misspellings
        basic_hesitation_patterns = r'\b(uh|uhh|um|umm|hmm|hmmmm|mm|mmm|err|er|erm|ah|ahh|ehh|eh|oh|ohh|aah|huh|hm|mhm|like|you know|i mean|so|uh-huh|uhm|em|eeh)\b'
        
        # Additional patterns that might indicate hesitation - EXPANDED
        extended_patterns = [
            # Filler words (added more)
            r'\b(well|basically|actually|literally|kind of|sort of|just|right|okay|so|then|anyway|anyhow|you see|see|look|listen|certainly|definitely|probably|maybe|possibly|I suppose|perhaps)\b',
            
            # Thinking phrases (added more)
            r'\b(i think|i guess|i suppose|let me see|let me think|let\'s see|wait|hold on|one second|give me a moment|how do i say this|how can i put this)\b',
            
            # Pause indicators
            r'\.{2,}',  # Ellipses
            r'-+',      # Dashes
            r'\([^\)]*\)',  # Text in parentheses
            r'\s{2,}',   # Multiple spaces
            
            # Stuttering patterns (expanded)
            r'\b(\w{1,2})\1+\b',  # Repeated syllables
            r'\b(\w+)-\1\b',  # Word-Word repetitions
            r'\b[a-z]-[a-z]',  # Stammering
            r'\b(\w+)(?:\s+\1){1,}\b',  # Repeated words with spaces
            
            # Common phrases indicating hesitation
            r'\b(let\'?s see|what was i saying|where was i|as i was saying|going back to)\b',
            
            # Uncertainty markers
            r'\b(i\'?m not sure|i don\'?t know exactly|somewhat|kind of|sort of|more or less)\b',
            
            # Phonological fragments (interrupted words)
            r'\b\w{1,2}-',  # Words cut off after 1-2 letters
            
            # Fillers specific to certain contexts
            r'\b(moving on|to continue|anyway|in any case)\b'
        ]
        
        # Use a more permissive case for basic hesitations
        transcription_lower = transcription.lower()
        basic_hesitations = re.findall(basic_hesitation_patterns, transcription_lower)
        
        # Count additional patterns
        extended_hesitations = []
        for pattern in extended_patterns:
            matches = re.findall(pattern, transcription_lower)
            if isinstance(matches, list) and matches:
                if isinstance(matches[0], tuple):
                    # If match returns tuples (happens with capture groups), flatten
                    matches = [m[0] if isinstance(m, tuple) else m for m in matches]
                extended_hesitations.extend(matches)
        
        # Combine both types
        all_hesitations = basic_hesitations + extended_hesitations
        
        # Get word count for ratio calculation
        words = word_tokenize(transcription)
        word_count = max(len(words), 1)  # Avoid division by zero
        
        # ADDITIONAL ANALYSIS: Check for sentence restarts
        sentences = sent_tokenize(transcription)
        restart_count = 0
        restart_markers = ['i mean', 'what i meant', 'that is', 'in other words']
        
        for sentence in sentences:
            for marker in restart_markers:
                if marker in sentence.lower():
                    restart_count += 1
                    break
        
        # Count by type
        hesitation_count_by_type = {}
        for h in all_hesitations:
            h_type = str(h).lower().strip()
            if h_type:  # Skip empty strings
                if h_type in hesitation_count_by_type:
                    hesitation_count_by_type[h_type] += 1
                else:
                    hesitation_count_by_type[h_type] = 1
        
        # Add restarts to hesitation count
        if restart_count > 0:
            hesitation_count_by_type["sentence_restarts"] = restart_count
            all_hesitations.extend(["sentence_restart"] * restart_count)
        
        # Analyze sentence patterns for hesitation detection
        repeat_words_count = 0
        
        # Check for repeated words in close proximity (sign of hesitation)
        for sentence in sentences:
            words_in_sentence = word_tokenize(sentence.lower())
            for i in range(len(words_in_sentence) - 1):
                if words_in_sentence[i] == words_in_sentence[i+1] and words_in_sentence[i] not in ['i', 'the', 'a', 'an', 'to', 'in', 'on', 'with', 'and', 'or', 'but']:
                    repeat_words_count += 1
                    hesitation_type = f"repeated_{words_in_sentence[i]}"
                    if hesitation_type in hesitation_count_by_type:
                        hesitation_count_by_type[hesitation_type] += 1
                    else:
                        hesitation_count_by_type[hesitation_type] = 1
        
        # ADDITIONAL ANALYSIS: Short utterance detection
        # Very short sentences might indicate incomplete thoughts
        short_utterances = 0
        for sentence in sentences:
            sentence_words = word_tokenize(sentence)
            if 1 < len(sentence_words) < 4:  # 2-3 word sentences
                short_utterances += 1
        
        if short_utterances > 1:  # Only count if multiple
            hesitation_count_by_type["short_utterances"] = short_utterances
            all_hesitations.extend(["short_utterance"] * min(short_utterances, 3))  # Cap contribution
        
        # ADDITIONAL ANALYSIS: Word frequency distribution
        # Unusual word frequency distribution can indicate hesitation
        if word_count > 5:  # Only meaningful for longer transcriptions
            # Count word frequencies
            word_freqs = {}
            for word in words:
                word = word.lower()
                if word in word_freqs:
                    word_freqs[word] += 1
                else:
                    word_freqs[word] = 1
            
            # Check for high frequency of basic words (potential speech struggle)
            basic_words = ['the', 'a', 'an', 'i', 'you', 'he', 'she', 'it', 'this', 'that']
            basic_word_count = sum(word_freqs.get(word, 0) for word in basic_words)
            basic_word_ratio = basic_word_count / word_count
            
            # If over 50% basic words, might indicate limitation in vocabulary usage
            if basic_word_ratio > 0.5 and word_count > 10:
                hesitation_count_by_type["high_basic_word_ratio"] = 1
                all_hesitations.append("high_basic_word_ratio")
        
        # Calculate results
        total_hesitations = len(all_hesitations) + repeat_words_count
        hesitations_per_word = total_hesitations / word_count
        
        # If we found no hesitations, try more aggressive approaches
        if total_hesitations == 0:
            # FALLBACK 1: Check for unusually short transcript for audio length
            if "audio_duration" in locals() and audio_duration > 5.0:
                words_per_second = word_count / audio_duration
                # If very slow speech (less than 1 word per second)
                if words_per_second < 1.0:
                    hesitation_count_by_type["slow_word_rate"] = 1
                    total_hesitations += 1
            
            # FALLBACK 2: Check for short words
            short_words = [w for w in words if len(w) == 1 and w.lower() not in ['i', 'a']]
            if short_words:
                hesitation_count_by_type["possible_truncated"] = len(short_words)
                total_hesitations += len(short_words)
            
            # FALLBACK 3: Check for unusual punctuation that might indicate hesitation
            unusual_punct_count = transcription.count('...') + transcription.count('--')
            if unusual_punct_count > 0:
                hesitation_count_by_type["unusual_punctuation"] = unusual_punct_count
                total_hesitations += unusual_punct_count
            
            # FALLBACK 4: For short recordings, assume at least one hesitation
            if word_count < 15 and "audio_duration" in locals() and audio_duration < 10:
                # Short recordings often have at least some hesitation
                hesitation_count_by_type["assumed_hesitation"] = 1
                total_hesitations += 1
                
        # Recalculate ratio after fallbacks
        hesitations_per_word = total_hesitations / word_count
        
        logger.info(f"Detected {total_hesitations} hesitation markers in {word_count} words")
        if hesitation_count_by_type:
            logger.info(f"Hesitation types: {hesitation_count_by_type}")
        
        return {
            "total_hesitations": total_hesitations,
            "hesitation_types": hesitation_count_by_type,
            "hesitations_per_word": float(hesitations_per_word),
            "detected_words": words,  # Include for debugging
            "hesitation_markers": all_hesitations  # Include for debugging
        }
    except Exception as e:
        logger.error(f"Error detecting hesitation markers: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "total_hesitations": 0,
            "hesitation_types": {},
            "hesitations_per_word": 0.0
        }

def analyze_speech_rate(transcription, audio_duration):
    """Calculate speech rate (words per minute)"""
    words = word_tokenize(transcription)
    
    return {
        "total_words": len(words),
        "speech_duration_minutes": audio_duration / 60,
        "words_per_minute": (len(words) / audio_duration) * 60 if audio_duration > 0 else 0
    }

def extract_pitch_features(file_path):
    """Extract pitch-related features"""
    try:
        y, sr = librosa.load(file_path)
        
        # Extract pitch (fundamental frequency) using librosa
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        
        # Find the most prominent pitch in each frame
        pitch_values = []
        for i in range(pitches.shape[1]):
            index = magnitudes[:, i].argmax()
            pitch = pitches[index, i]
            if pitch > 0:  # Exclude zero pitch values
                # Convert numpy.float32 to standard Python float
                pitch_values.append(float(pitch))
        
        if not pitch_values:
            return {
                "pitch_mean": 0.0,
                "pitch_std": 0.0,
                "pitch_min": 0.0,
                "pitch_max": 0.0
            }
        
        # Explicitly convert NumPy values to Python floats
        return {
            "pitch_mean": float(np.mean(pitch_values)),
            "pitch_std": float(np.std(pitch_values)),  # Pitch variability
            "pitch_min": float(np.min(pitch_values)),
            "pitch_max": float(np.max(pitch_values))
        }
    except Exception as e:
        logger.error(f"Error extracting pitch features: {str(e)}")
        return {
            "pitch_mean": 0.0,
            "pitch_std": 0.0,
            "pitch_min": 0.0,
            "pitch_max": 0.0
        }

def detect_word_recall_issues(transcription):
    """Detect potential word recall issues"""
    # Look for patterns that might indicate word recall issues
    patterns = [
        r'\b(what\'s that word|what\'s it called|what do you call it|can\'t remember the word)\b',
        r'\b(thing|thingy|stuff|whatchamacallit)\b',
        r'\bi mean\b'
    ]
    
    issue_count = 0
    for pattern in patterns:
        issue_count += len(re.findall(pattern, transcription.lower()))
    
    return {
        "potential_recall_issues": issue_count,
        "recall_issues_per_sentence": issue_count / len(sent_tokenize(transcription)) if transcription else 0
    }

def detect_audio_hesitations(audio_path):
    """
    Detect hesitations directly from audio by identifying regions of low spectral flux
    and low energy that may indicate non-lexical hesitation sounds.
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=None)
        
        # More sensitive parameters
        frame_length = 512  # Smaller frame for better temporal resolution
        hop_length = 256    # Increased overlap for better detection
        
        # Calculate spectral flux - helps identify changes in frequency content
        # Lower values may indicate hesitations
        spec = np.abs(librosa.stft(y, n_fft=frame_length, hop_length=hop_length))
        
        # Calculate spectral flux - the rate of change between adjacent frames
        spec_diff = np.diff(spec, axis=1)
        spec_flux = np.sum(spec_diff**2, axis=0)
        
        # Also calculate energy - helps identify silent/quiet regions
        energy = np.sum(spec**2, axis=0)
        energy_normalized = energy / np.max(energy)
        
        # NEW: Add zero-crossing rate - helps identify fricatives and hesitation sounds
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # NEW: Calculate MFCC derivatives - helps identify transitions in speech sounds
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta_mean = np.mean(np.abs(mfcc_delta), axis=0)
        
        # Calculate thresholds more dynamically
        # For spectral flux - hesitations typically have low spectral flux
        flux_threshold = np.percentile(spec_flux, 30)  # More sensitive (was 20)
        
        # For energy - hesitations typically have medium-low energy (not silence)
        energy_low = np.percentile(energy_normalized, 10)
        energy_high = np.percentile(energy_normalized, 60)  # Higher upper bound
        
        # For zero-crossing - hesitations often have distinctive ZCR
        zcr_low = np.percentile(zcr, 20)
        zcr_high = np.percentile(zcr, 80)
        
        # For MFCC delta - hesitations have low change in vocal tract configuration
        mfcc_delta_threshold = np.percentile(mfcc_delta_mean, 30)
        
        # Track frames that might be hesitations
        # NEW: Multi-feature detection with scoring
        hesitation_scores = np.zeros_like(energy_normalized)
        
        # Score each frame based on multiple features
        for i in range(len(hesitation_scores)):
            score = 0
            
            # Check spectral flux (if in range)
            if i < len(spec_flux) and spec_flux[i] < flux_threshold:
                score += 1
                
            # Check energy (if in medium-low range - not silence, not loud speech)
            if energy_low < energy_normalized[i] < energy_high:
                score += 1
                
            # Check ZCR (if in typical hesitation range)
            if i < len(zcr) and zcr_low < zcr[i] < zcr_high:
                score += 1
                
            # Check MFCC delta (if showing low articulatory change)
            if i < len(mfcc_delta_mean) and mfcc_delta_mean[i] < mfcc_delta_threshold:
                score += 1
                
            hesitation_scores[i] = score
        
        # Find regions where scores are high enough to indicate hesitation
        # NEW: Use a more sensitive threshold
        hesitation_threshold = 2  # Need at least 2 features to agree
        hesitation_regions = []
        in_hesitation = False
        start_idx = 0
        
        for i, score in enumerate(hesitation_scores):
            if score >= hesitation_threshold and not in_hesitation:
                # Start of potential hesitation
                in_hesitation = True
                start_idx = i
            elif (score < hesitation_threshold or i == len(hesitation_scores) - 1) and in_hesitation:
                # End of potential hesitation
                in_hesitation = False
                end_idx = i
                duration_sec = (end_idx - start_idx) * hop_length / sr
                
                # Only include if duration is in typical hesitation range
                # NEW: More permissive duration thresholds
                if 0.15 <= duration_sec <= 1.2:  # More permissive (was 0.2-1.0)
                    # NEW: Double-check with energy profile to confirm it's not just silence
                    segment_energy = np.mean(energy_normalized[start_idx:end_idx])
                    
                    # Ensure it's not just silence and has some audio content
                    if segment_energy > energy_low * 1.2:
                        # Convert frame indices to time positions
                        start_time = start_idx * hop_length / sr
                        end_time = end_idx * hop_length / sr
                        hesitation_regions.append({
                            "start": start_time,
                            "end": end_time,
                            "duration": duration_sec,
                            "energy": float(segment_energy),
                            "confidence": float(np.mean(hesitation_scores[start_idx:end_idx])) / 4.0  # Normalize
                        })
        
        # NEW: Post-processing to merge nearby hesitations
        if len(hesitation_regions) > 1:
            merged_regions = []
            current_region = hesitation_regions[0]
            
            for i in range(1, len(hesitation_regions)):
                if hesitation_regions[i]["start"] - current_region["end"] < 0.3:  # If less than 300ms apart
                    # Merge regions
                    current_region["end"] = hesitation_regions[i]["end"]
                    current_region["duration"] = current_region["end"] - current_region["start"]
                    # Update confidence
                    current_region["confidence"] = (current_region["confidence"] + hesitation_regions[i]["confidence"]) / 2
                else:
                    merged_regions.append(current_region)
                    current_region = hesitation_regions[i]
            
            merged_regions.append(current_region)
            hesitation_regions = merged_regions
        
        # Calculate summary metrics
        total_hesitation_count = len(hesitation_regions)
        total_hesitation_duration = sum(region["duration"] for region in hesitation_regions)
        avg_hesitation_duration = total_hesitation_duration / total_hesitation_count if total_hesitation_count > 0 else 0
        
        # NEW: Calculate audio duration for normalized metrics
        audio_duration = len(y) / sr
        hesitation_ratio = total_hesitation_duration / audio_duration if audio_duration > 0 else 0
        
        # Return detailed results
        return {
            "total_audio_hesitations": total_hesitation_count,
            "total_hesitation_duration": float(total_hesitation_duration),
            "avg_hesitation_duration": float(avg_hesitation_duration),
            "hesitation_regions": hesitation_regions,
            "hesitation_time_ratio": float(hesitation_ratio),
            "audio_duration": float(audio_duration)
        }
    except Exception as e:
        logger.error(f"Error in audio hesitation detection: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "total_audio_hesitations": 0,
            "total_hesitation_duration": 0.0,
            "avg_hesitation_duration": 0.0,
            "hesitation_regions": [],
            "hesitation_time_ratio": 0.0,
            "audio_duration": 0.0
        }

def audio_duration_is_short(y, sr):
    """Check if audio duration is short (less than 15 seconds)"""
    duration = len(y) / sr
    return duration < 15

def run_ml_analysis(features):
    """Apply basic unsupervised ML to detect anomalies"""
    try:
        # Clean the features dict to remove any debug fields
        features_clean = {
            "pauses": features["pauses"],
            "hesitations": {k: features["hesitations"][k] for k in ["total_hesitations", "hesitations_per_word", "hesitation_types", "audio_hesitations"] 
                          if k in features["hesitations"]},
            "speech_rate": features["speech_rate"],
            "pitch": features["pitch"],
            "word_recall": features["word_recall"]
        }
        
        # Convert features dictionary to array for ML processing
        feature_array = np.array([
            features_clean["pauses"]["pauses_per_sentence"],
            features_clean["pauses"]["average_pause_duration"],
            features_clean["hesitations"]["hesitations_per_word"],
            features_clean["speech_rate"]["words_per_minute"],
            features_clean["pitch"]["pitch_std"],
            features_clean["word_recall"]["recall_issues_per_sentence"]
        ]).reshape(1, -1)
        
        # We would normally have training data here, but for now we'll use some thresholds
        # This could be replaced with a proper anomaly detection model trained on real data
        
        # Defining risk thresholds (these would be determined by analysis of normal speech)
        risk_score = 0
        
        # Check different levels of hesitations based on new metrics
        hesitation_level = features_clean["hesitations"]["hesitations_per_word"]
        
        # More granular hesitation assessment
        hesitation_risk = 0
        hesitation_reason = None
        
        # Check for audio-detected hesitations (which might not show in transcription)
        audio_hesitations = features_clean["hesitations"].get("audio_hesitations", 0)
        
        if audio_hesitations > 0:
            # Consider audio-detected hesitations more important
            if audio_hesitations > 3:
                hesitation_risk = 2
                hesitation_reason = f"High number of audio-detected hesitations ({audio_hesitations})"
            else:
                hesitation_risk = 1
                hesitation_reason = f"Some audio-detected hesitations ({audio_hesitations})"
        
        # If text-based hesitation is significant, consider that too
        if hesitation_level > 0.2:
            hesitation_risk = 2
            hesitation_reason = f"Very frequent hesitations ({hesitation_level:.2f} per word)"
        elif hesitation_level > 0.1:
            hesitation_risk = max(hesitation_risk, 1)
            if not hesitation_reason:
                hesitation_reason = f"Frequent hesitations ({hesitation_level:.2f} per word)"
        
        # Add hesitation risk to overall risk
        risk_score += hesitation_risk
        
        # Standard thresholds for other metrics
        if features_clean["pauses"]["pauses_per_sentence"] > 2:
            risk_score += 1
        if features_clean["pauses"]["average_pause_duration"] > 1.5:
            risk_score += 1
        if features_clean["speech_rate"]["words_per_minute"] < 100:
            risk_score += 1
        if features_clean["word_recall"]["recall_issues_per_sentence"] > 0.5:
            risk_score += 2
        
        # Determine overall risk level
        risk_level = "Low"
        if risk_score >= 3:
            risk_level = "Moderate"
        if risk_score >= 5:
            risk_level = "High"
            
        # Create abnormal features list with better descriptions
        abnormal_features = []
        
        # Pause patterns
        if features_clean["pauses"]["pauses_per_sentence"] > 3:
            abnormal_features.append(f"Very high number of pauses ({features_clean['pauses']['pauses_per_sentence']:.1f} per sentence)")
        elif features_clean["pauses"]["pauses_per_sentence"] > 2:
            abnormal_features.append(f"High number of pauses ({features_clean['pauses']['pauses_per_sentence']:.1f} per sentence)")
        
        # Pause duration
        if features_clean["pauses"]["average_pause_duration"] > 1.5:
            abnormal_features.append(f"Long pause durations (avg {features_clean['pauses']['average_pause_duration']:.2f}s)")
        
        # Hesitations (using our enhanced detection)
        if hesitation_reason:
            abnormal_features.append(hesitation_reason)
        
        # Speech rate
        if features_clean["speech_rate"]["words_per_minute"] < 80:
            abnormal_features.append(f"Very slow speech rate ({features_clean['speech_rate']['words_per_minute']:.1f} WPM)")
        elif features_clean["speech_rate"]["words_per_minute"] < 100:
            abnormal_features.append(f"Slow speech rate ({features_clean['speech_rate']['words_per_minute']:.1f} WPM)")
        
        # Word recall issues
        if features_clean["word_recall"]["recall_issues_per_sentence"] > 0.5:
            abnormal_features.append("Word recall difficulties")
        
        # Filter out None values
        abnormal_features = [f for f in abnormal_features if f]
        
        # If no abnormal features found, add "None detected"
        if not abnormal_features:
            abnormal_features.append("No abnormal features detected")
        
        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "abnormal_features": abnormal_features,
            "hesitation_score": hesitation_risk
        }
    except Exception as e:
        logger.error(f"Error in ML analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "risk_score": 0,
            "risk_level": "Unknown",
            "abnormal_features": ["Error in analysis"],
            "hesitation_score": 0
        }

def analyze_transcript(transcript, audio_path=None):
    """
    Analyze transcript for hesitations and speaking quality metrics
    """
    try:
        # Default empty results for each analysis
        results = {
            "text_hesitation_markers": {},
            "audio_hesitations": {},
            "hesitation_score": 0,
            "speaking_metrics": {},
            "confidence": 0.0
        }
        
        # Step 1: Detect text-based hesitation markers
        text_results = detect_hesitation_markers(transcript)
        results["text_hesitation_markers"] = text_results
        
        # Step 2: Add audio-based detection if audio is provided
        if audio_path and os.path.exists(audio_path):
            audio_results = detect_audio_hesitations(audio_path)
            results["audio_hesitations"] = audio_results
            
            # Calculate combined metrics using both text and audio analysis
            total_hesitations = text_results.get("total_hesitations", 0) + audio_results.get("total_audio_hesitations", 0)
            
            # Calculate more speech metrics if we have audio
            words = transcript.split()
            word_count = len(words)
            
            audio_duration = audio_results.get("audio_duration", 0)
            if audio_duration > 0 and word_count > 0:
                # Calculate speaking rate (words per minute)
                speaking_rate = (word_count / audio_duration) * 60
                
                # Calculate articulation rate (excluding pauses)
                hesitation_time = audio_results.get("total_hesitation_duration", 0)
                effective_speaking_time = max(0.1, audio_duration - hesitation_time)
                articulation_rate = (word_count / effective_speaking_time) * 60
                
                # Calculate hesitation ratio
                hesitation_ratio = hesitation_time / audio_duration if audio_duration > 0 else 0
                
                # Store speaking metrics
                results["speaking_metrics"] = {
                    "word_count": word_count,
                    "duration_seconds": float(audio_duration),
                    "speaking_rate_wpm": float(speaking_rate),
                    "articulation_rate_wpm": float(articulation_rate),
                    "hesitation_ratio": float(hesitation_ratio),
                    "hesitation_time": float(hesitation_time)
                }
        
        # Step 3: Calculate overall hesitation score (100 = perfect, 0 = lots of hesitations)
        if len(transcript.split()) > 10:  # Only calculate if enough words in transcript
            # Base score calculation
            text_score = calculate_text_based_score(text_results, transcript)
            
            # If we have audio analysis, incorporate it
            if "audio_hesitations" in results and results["audio_hesitations"]:
                audio_score = calculate_audio_based_score(results["audio_hesitations"], transcript)
                # Combined score weighted 60% text, 40% audio
                hesitation_score = (text_score * 0.6) + (audio_score * 0.4)
                # Calculate confidence based on data quality
                confidence = 0.95  # High confidence with both text and audio
            else:
                # Text-only analysis
                hesitation_score = text_score
                confidence = 0.75  # Medium confidence with text only
                
            results["hesitation_score"] = max(0, min(100, round(hesitation_score)))
            results["confidence"] = confidence
        else:
            # Not enough text to provide reliable score
            results["hesitation_score"] = None
            results["confidence"] = 0.3
            
        return results
    
    except Exception as e:
        logger.error(f"Error analyzing transcript: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "text_hesitation_markers": {"total_markers": 0},
            "audio_hesitations": {},
            "hesitation_score": 50,
            "speaking_metrics": {},
            "confidence": 0.0
        }
        
def calculate_text_based_score(text_results, transcript):
    """Calculate a score based on text hesitation markers"""
    words = transcript.split()
    word_count = max(len(words), 1)  # Avoid division by zero
    
    # Get total markers and normalize by word count
    total_markers = text_results.get("total_hesitations", 0)
    marker_ratio = total_markers / word_count
    
    # Get weighted values for different marker types
    basic_markers = text_results.get("hesitation_types", {}).get("uh", 0) * 1.0
    extended_markers = text_results.get("hesitation_types", {}).get("sentence_restarts", 0) * 1.2
    restart_markers = text_results.get("hesitation_types", {}).get("sentence_restarts", 0) * 1.2
    repeat_markers = text_results.get("hesitation_types", {}).get("repeated_words", 0) * 0.9
    
    # Calculate weighted total
    weighted_total = basic_markers + extended_markers + restart_markers + repeat_markers
    
    # Calculate severity based on marker density
    if word_count < 30:
        # For short transcripts, be more lenient
        severity = weighted_total / max(1, word_count / 2)
    else:
        severity = weighted_total / max(1, word_count / 5)
    
    # Convert to score (100 = perfect, 0 = lots of hesitations)
    # More sophisticated curve: high scores require very few hesitations
    if severity == 0:
        return 100  # Perfect score for no hesitations
    elif severity < 0.05:
        return 95 - (severity * 100)  # 90-95 range for minimal hesitations
    elif severity < 0.1:
        return 85 - (severity * 100)  # 75-85 range for low hesitations
    elif severity < 0.2:
        return 75 - (severity * 66.7)  # 55-75 range for moderate hesitations
    elif severity < 0.3:
        return 60 - (severity * 66.7)  # 40-60 range for high hesitations
    else:
        return max(0, 40 - (severity * 50))  # 0-40 range for severe hesitations

def calculate_audio_based_score(audio_results, transcript):
    """Calculate a score based on audio hesitation analysis"""
    words = transcript.split()
    word_count = max(len(words), 1)  # Avoid division by zero
    
    # Get total audio hesitations
    hesitation_count = audio_results.get("total_audio_hesitations", 0)
    
    # Get hesitation ratio (time spent hesitating / total time)
    hesitation_ratio = audio_results.get("hesitation_time_ratio", 0)
    
    # Calculate base score (100 = perfect, 0 = lots of hesitations)
    # First factor: number of hesitations per word
    hesitation_per_word = hesitation_count / word_count
    count_score = 100 - (min(1.0, hesitation_per_word) * 100)
    
    # Second factor: time spent hesitating
    time_score = 100 - (min(1.0, hesitation_ratio * 3) * 100)
    
    # Combined score (weighted)
    return (count_score * 0.4) + (time_score * 0.6)

@app.route('/')
def index():
    """Render the homepage with the upload form"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_speech():
    """API endpoint to analyze uploaded audio file"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            logger.error("No file part in the request")
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        
        # Check if file is empty
        if file.filename == '':
            logger.error("No file selected")
            return jsonify({"error": "No selected file"}), 400
        
        # Check if file is allowed
        if not allowed_file(file.filename):
            logger.error(f"File type not allowed: {file.filename}")
            return jsonify({"error": "File type not allowed. Only WAV files are supported."}), 400
        
        # Save the file with a secure name
        filename = str(uuid.uuid4()) + '.wav'
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        logger.info(f"Saving uploaded file to {file_path}")
        file.save(file_path)
        
        # Get audio duration
        try:
            audio = AudioSegment.from_wav(file_path)
            audio_duration = len(audio) / 1000  # Convert to seconds
            logger.info(f"Audio duration: {audio_duration} seconds")
        except Exception as e:
            logger.error(f"Error processing audio file: {str(e)}")
            return jsonify({"error": f"Error processing audio file: {str(e)}"}), 400
        
        # Transcribe audio to text
        transcription = transcribe_audio(file_path)
        
        if not transcription:
            logger.error("Failed to transcribe audio")
            return jsonify({"error": "Failed to transcribe audio. Please check the file quality."}), 400
        
        logger.info(f"Transcription successful: {transcription[:50]}...")
        
        # Extract features
        logger.info("Extracting speech features...")
        pause_features = extract_pauses(file_path, transcription)
        hesitation_features = detect_hesitation_markers(transcription)
        
        # Add direct audio-based hesitation detection
        audio_hesitations = detect_audio_hesitations(file_path)
        
        # Combine text-based and audio-based hesitation detection
        if "total_hesitations" in hesitation_features and "total_audio_hesitations" in audio_hesitations:
            total_combined = hesitation_features["total_hesitations"] + audio_hesitations["total_audio_hesitations"]
            words = word_tokenize(transcription)
            word_count = max(len(words), 1)
            
            # Update hesitation features with audio-based detection
            hesitation_features["total_hesitations"] = total_combined
            hesitation_features["hesitations_per_word"] = float(total_combined / word_count)
            hesitation_features["audio_hesitations"] = audio_hesitations["total_audio_hesitations"]
            
            # Add audio hesitation durations to hesitation types
            if audio_hesitations["total_audio_hesitations"] > 0:
                hesitation_features["hesitation_types"]["audio_detected"] = audio_hesitations["total_audio_hesitations"]
        
        speech_rate_features = analyze_speech_rate(transcription, audio_duration)
        pitch_features = extract_pitch_features(file_path)
        word_recall_features = detect_word_recall_issues(transcription)
        
        # Combine features
        features = {
            "transcription": transcription,
            "pauses": pause_features,
            "hesitations": hesitation_features,
            "speech_rate": speech_rate_features,
            "pitch": pitch_features,
            "word_recall": word_recall_features
        }
        
        # Run ML analysis
        logger.info("Running ML analysis...")
        ml_results = run_ml_analysis(features)
        
        # Add ML results to features
        features["analysis"] = ml_results
        
        # Remove debugging fields before responding
        if "detected_words" in features["hesitations"]:
            del features["hesitations"]["detected_words"]
        if "hesitation_markers" in features["hesitations"]:
            del features["hesitations"]["hesitation_markers"]
        
        # Convert any NumPy types to standard Python types for JSON serialization
        features = convert_numpy_types(features)
        
        # Save results to JSON file
        result_filename = os.path.splitext(filename)[0] + '.json'
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        
        with open(result_path, 'w') as f:
            json.dump(features, f, indent=2, cls=NumpyEncoder)
        
        logger.info("Analysis completed successfully")
        return jsonify({
            "success": True,
            "result_id": os.path.splitext(filename)[0],
            "features": features
        })
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Error processing file: {str(e)}"}), 500

@app.route('/results/<result_id>')
def get_results(result_id):
    """API endpoint to retrieve previously saved results"""
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{result_id}.json")
    
    if not os.path.exists(result_path):
        return jsonify({"error": "Results not found"}), 404
    
    with open(result_path, 'r') as f:
        results = json.load(f)
    
    return jsonify(results)

@app.route('/test')
def test_endpoint():
    """Simple test endpoint to verify the server is running"""
    return jsonify({
        "status": "success",
        "message": "API is working correctly"
    })

@app.route('/test-page')
def test_page():
    """Simple test route to verify template rendering works"""
    return render_template('test.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.config['STATIC_FOLDER'], filename)

if __name__ == '__main__':
    # Increase logging level to see more details
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    # Create uploads directory if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
        logger.info(f"Created uploads directory: {app.config['UPLOAD_FOLDER']}")
    
    # Make uploads directory writable
    try:
        os.chmod(app.config['UPLOAD_FOLDER'], 0o777)
        logger.info("Set uploads directory permissions to writable")
    except Exception as e:
        logger.warning(f"Could not set uploads directory permissions: {str(e)}")
    
    app.run(debug=True) 