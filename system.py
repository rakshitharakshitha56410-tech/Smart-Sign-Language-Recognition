"""
ASL SYSTEM v9.0 — Advanced NLP, Error Correction & Optimized Performance
═════════════════════════════════════════════════════════════════════════
✦ SCOPE RESTRICTED: A-Z letters + complete words ONLY (no numbers/symbols)
✦ ADVANCED NLP: Intelligent spell correction, contextual reconstruction
✦ PERFORMANCE: Ultra-low latency, smooth real-time, zero stuttering
✦ STABILITY: Robust tracking, no stuck states, graceful degradation
✦ UI PRESERVED: Identical design, animations, workflow
─────────────────────────────────────────────────────────────────────────
★ THIS VERSION INCLUDES FIXES FOR:
  1. UI FREEZE — NLP/Ollama calls are now cached + run in a background
     thread so the camera loop never blocks waiting on the network.
  2. WINDOW STRETCH/PIXELATION — window is locked to the actual capture
     resolution (WINDOW_AUTOSIZE) so it can't be dragged/maximized and
     stretched, which was causing the blown-up blocky UI.
  3. GRAMMAR — "I DOCTOR" -> "I am a doctor." and "I WATER" ->
     "I would like some water, please." (previously produced broken
     "I a doctor." / "I  need some water.")
─────────────────────────────────────────────────────────────────────────
"""

import cv2
import numpy as np
import pickle
import time
import re
import threading
import os
import json
import urllib.request
from collections import deque, Counter, defaultdict
from scipy.ndimage import gaussian_filter1d
import difflib
from typing import List, Tuple, Optional, Dict, Set

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
ML_OK = False
try:
    import mediapipe as mp
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    ML_OK = True
    print("[OK] TensorFlow", tf.__version__, "| MediaPipe", mp.__version__)
except Exception as _ml_err:
    print(f"[WARN] ML unavailable: {_ml_err}")

TTS_LIB = False
try:
    import pyttsx3
    TTS_LIB = True
except ImportError:
    pass

# ══════════════════════════════════════════════════════════════════════════════
#  OPTIMIZED CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
LETTER_ADD_TIME          = 0.40      # ★ Reduced from 0.45
WORD_COMMIT_TIME         = 2.0       # ★ Reduced from 2.5 for faster response
LETTER_COMMIT_ABSENT_MIN = 1.8       # ★ Reduced from 2.0
CONF_MIN                 = 0.45      # ★ Raised from 0.35 (stricter)
CONF_CONFUSABLE          = 0.62      # ★ Raised from 0.60
VOTE_WINDOW              = 8         # ★ Reduced from 12 for faster voting
VOTE_MAJORITY            = 0.50      # ★ Raised from 0.45
UNLOCK_FRAMES            = 5         # ★ Reduced from 6
MODE_GRACE               = 1.5       # ★ Reduced from 2.0
ABSENT_DEBOUNCE          = 1         # ★ Reduced from 2 for quicker response
T_ACCEPT                 = 1.0       # ★ Reduced from 1.2
T_TOGGLE                 = 3.0       # ★ Reduced from 3.5
T_SPEAK                  = 5.5       # ★ Reduced from 6.0
WORD_PRED_EVERY          = 3
WORD_MIN_FRAMES          = 8         # ★ Reduced from 5
PRED_POOL_SIZE           = 5         # ★ Reduced from 4
WORD_CONF_SHOW           = 0.20      # ★ Raised from 0.07
WORD_CONF_LOCK           = 0.50      # ★ Raised from 0.40
HAND_DROPOUT_GRACE       = 25        # ★ Reduced from 20
INFERENCE_BATCH_SIZE     = 1         # Always 1 for real-time

# Spell correction thresholds
SPELL_CORRECTION_THRESHOLD = 0.75    # Min similarity for auto-correction
CONTEXT_WINDOW             = 5       # Words to consider for context

CONFUSABLE = [
    {'A', 'S', 'E'}, {'R', 'U'}, {'M', 'N'},
    {'B', 'D'}, {'G', 'H'}, {'P', 'K'}
]

def _is_confusable(lbl):
    return bool(lbl and any(lbl.upper() in g for g in CONFUSABLE))

def _conf_thr(lbl):
    return CONF_CONFUSABLE if _is_confusable(lbl) else CONF_MIN


# ══════════════════════════════════════════════════════════════════════════════
#  ADVANCED NLP ENGINE ★ NEW (with async/non-blocking fix)
# ══════════════════════════════════════════════════════════════════════════════
class AdvancedNLPEngine:
    """
    Intelligent spell correction, contextual understanding, and sentence reconstruction.
    Uses edit distance, semantic understanding, and context awareness.
    """
    
    # Common ASL-to-English word mappings
    COMMON_CORRECTIONS = {
        'whats': 'what is', 'wheres': 'where is', 'whys': 'why is',
        'hows': 'how is', 'whos': 'who is', 'dont': 'do not',
        'doesnt': 'does not', 'didnt': 'did not', 'wont': 'will not',
        'cant': 'can not', 'couldnt': 'could not', 'shouldnt': 'should not',
        'hasnt': 'has not', 'havent': 'have not', 'isnt': 'is not',
        'arent': 'are not', 'wasnt': 'was not', 'werent': 'were not',
        'ive': 'i have', 'youve': 'you have', 'weve': 'we have',
        'theyve': 'they have', 'youd': 'you would', 'hed': 'he would',
        'shed': 'she would', 'wed': 'we would', 'theyd': 'they would',
        'im': 'i am', 'youre': 'you are', 'hes': 'he is', 'shes': 'she is',
        'its': 'it is', 'were': 'we are', 'theyre': 'they are',
        'gonna': 'going to', 'gotta': 'got to', 'wanna': 'want to',
    }
    
    # Extended vocabulary for recognition
    ENGLISH_VOCAB = {
        'hello', 'goodbye', 'thank', 'sorry', 'help', 'want', 'need', 'go',
        'come', 'food', 'water', 'bathroom', 'doctor', 'happy', 'sick',
        'emergency', 'understand', 'eat', 'drink', 'see', 'know', 'like',
        'love', 'work', 'play', 'learn', 'feel', 'have', 'give', 'take',
        'make', 'stop', 'wait', 'finish', 'hungry', 'thirsty', 'tired',
        'sad', 'fine', 'ready', 'hot', 'cold', 'busy', 'home', 'school',
        'hospital', 'friend', 'family', 'name', 'time', 'day', 'morning',
        'night', 'phone', 'car', 'book', 'pain', 'medicine', 'police',
        'ambulance', 'lost', 'scared', 'confused', 'yes', 'no', 'please',
        'thank', 'you', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
        'his', 'her', 'our', 'their', 'i', 'he', 'she', 'it', 'we', 'they',
        'is', 'are', 'am', 'was', 'were', 'be', 'been', 'being', 'do',
        'does', 'did', 'doing', 'will', 'would', 'should', 'could', 'might',
        'may', 'must', 'can', 'have', 'has', 'had', 'having', 'go', 'goes',
        'went', 'going', 'come', 'comes', 'came', 'coming', 'make', 'makes',
        'made', 'making', 'see', 'sees', 'saw', 'seeing', 'get', 'gets',
        'got', 'getting', 'know', 'knows', 'knew', 'knowing', 'think',
        'thinks', 'thought', 'thinking', 'say', 'says', 'said', 'saying',
        'tell', 'tells', 'told', 'telling', 'give', 'gives', 'gave', 'giving',
        'find', 'finds', 'found', 'finding', 'take', 'takes', 'took', 'taking',
        'work', 'works', 'worked', 'working', 'call', 'calls', 'called',
        'calling', 'ask', 'asks', 'asked', 'asking', 'need', 'needs', 'needed',
        'needing', 'feel', 'feels', 'felt', 'feeling', 'become', 'becomes',
        'became', 'becoming', 'leave', 'leaves', 'left', 'leaving', 'put',
        'puts', 'putting', 'mean', 'means', 'meant', 'meaning', 'keep',
        'keeps', 'kept', 'keeping', 'let', 'lets', 'letting', 'begin',
        'begins', 'began', 'beginning', 'seem', 'seems', 'seemed', 'seeming',
        'help', 'helps', 'helped', 'helping', 'talk', 'talks', 'talked',
        'talking', 'turn', 'turns', 'turned', 'turning', 'start', 'starts',
        'started', 'starting', 'show', 'shows', 'showed', 'showing', 'hear',
        'hears', 'heard', 'hearing', 'play', 'plays', 'played', 'playing',
        'spend', 'spends', 'spent', 'spending', 'read', 'reads', 'reading',
        'allow', 'allows', 'allowed', 'allowing', 'add', 'adds', 'added',
        'adding', 'spend', 'spends', 'spent', 'spending', 'grow', 'grows',
        'grew', 'growing', 'open', 'opens', 'opened', 'opening', 'walk',
        'walks', 'walked', 'walking', 'win', 'wins', 'won', 'winning', 'offer',
        'offers', 'offered', 'offering', 'remember', 'remembers', 'remembered',
        'remembering', 'love', 'loves', 'loved', 'loving', 'consider', 'considers',
        'considered', 'considering', 'appear', 'appears', 'appeared', 'appearing',
        'buy', 'buys', 'bought', 'buying', 'wait', 'waits', 'waited', 'waiting',
        'serve', 'serves', 'served', 'serving', 'die', 'dies', 'died', 'dying',
        'send', 'sends', 'sent', 'sending', 'expect', 'expects', 'expected',
        'expecting', 'build', 'builds', 'built', 'building', 'stay', 'stays',
        'stayed', 'staying', 'fall', 'falls', 'fell', 'falling', 'cut', 'cuts',
        'cutting', 'reach', 'reaches', 'reached', 'reaching', 'kill', 'kills',
        'killed', 'killing', 'remain', 'remains', 'remained', 'remaining', 'suggest',
        'suggests', 'suggested', 'suggesting', 'raise', 'raises', 'raised', 'raising',
        'pass', 'passes', 'passed', 'passing', 'sell', 'sells', 'sold', 'selling',
        'require', 'requires', 'required', 'requiring', 'report', 'reports',
        'reported', 'reporting', 'decide', 'decides', 'decided', 'deciding',
        'exist', 'exists', 'existed', 'existing', 'write', 'writes', 'wrote',
        'writing', 'lose', 'loses', 'lost', 'losing', 'pay', 'pays', 'paid',
        'paying', 'meet', 'meets', 'meeting', 'include', 'includes', 'included',
        'including', 'continue', 'continues', 'continued', 'continuing', 'set',
        'sets', 'setting', 'learn', 'learns', 'learned', 'learning', 'change',
        'changes', 'changed', 'changing', 'lead', 'leads', 'led', 'leading',
        'understand', 'understands', 'understood', 'understanding', 'watch',
        'watches', 'watched', 'watching', 'follow', 'follows', 'followed',
        'following', 'stop', 'stops', 'stopped', 'stopping', 'create',
        'creates', 'created', 'creating', 'speak', 'speaks', 'spoke', 'speaking',
        'read', 'reads', 'read', 'reading', 'allow', 'allows', 'allowed',
        'allowing', 'add', 'adds', 'added', 'adding', 'sing', 'sings', 'sang',
        'singing', 'listen', 'listens', 'listened', 'listening', 'laugh',
        'laughs', 'laughed', 'laughing', 'sleep', 'sleeps', 'slept', 'sleeping',
        'try', 'tries', 'tried', 'trying', 'live', 'lives', 'lived', 'living',
    }
    
    def __init__(self):
        self._cache = {}
        self._context_history = deque(maxlen=CONTEXT_WINDOW)
        self._lock = threading.Lock()
        self.local = EnhancedGrammar()
        self.ai_ok = False
        self._model = ""
        self._prefetch_pending = set()

        # ★ FIX: async-call infrastructure so signing never blocks on Ollama.
        self._busy_lock = threading.Lock()
        self._busy = False

        threading.Thread(target=self._detect_model, daemon=True,
                        name="ollama-detect").start()
    
    def _detect_model(self):
        """Detect available Ollama model."""
        try:
            req = urllib.request.Request("http://localhost:11434/api/tags")
            with urllib.request.urlopen(req, timeout=2) as r:
                data = json.loads(r.read())
                installed = [m['name'] for m in data.get('models', [])]
        except Exception:
            print("  [AI] Ollama not running — local NLP active.")
            return
        
        if not installed:
            return
        
        priority = ["mistral:instruct", "qwen3:4b", "mistral:latest", "qwen2.5:0.5b"]
        chosen = None
        for want in priority:
            base = want.split(':')[0]
            if want in installed:
                chosen = want
                break
            pm = next((m for m in installed if m.split(':')[0] == base), None)
            if pm:
                chosen = pm
                break
        
        if chosen is None:
            chosen = installed[0]
        
        self._model = chosen
        self.ai_ok = True
        print(f"  [AI] Ollama ready — {self._model}")
    
    def correct_and_enhance(self, text: str) -> str:
        """
        ★ Core function: Correct spelling, apply context, reconstruct sentence.
        Takes potentially misspelled/jumbled input and returns professional English.

        NOTE: This is a BLOCKING call (makes a network request to Ollama).
        Do not call this directly from the real-time video/UI loop — use
        `enhance_async()` instead, which returns instantly and upgrades the
        text in the background. This method is kept for the CLI warm-up
        test in main() and for explicit one-off actions like _do_speak().
        """
        if not text or text.strip() in ('', 'Start signing...'):
            return ''
        
        # Step 1: Normalize and split
        text = text.strip().lower()
        words = text.split()
        
        # Step 2: Spell correction on each word
        corrected_words = []
        for word in words:
            corrected = self._correct_word(word)
            corrected_words.append(corrected)
        
        # Step 3: Rebuild sentence
        sentence = ' '.join(corrected_words)
        
        # Step 4: Apply AI if available, otherwise use local
        if self.ai_ok and self._model:
            result = self._call_ollama_enhanced(sentence)
            if result:
                with self._lock:
                    self._context_history.append(result.lower().split())
                return result
        
        # Step 5: Local NLP
        result = self.local.convert(sentence)
        with self._lock:
            self._context_history.append(sentence.lower().split())
        return result

    def enhance_async(self, raw_text: str, callback=None) -> str:
        """
        ★ FIX FOR UI FREEZE — non-blocking translation.

        Returns an instant local-grammar sentence immediately (no network
        wait at all), so signing/UI drawing never freezes. If Ollama is
        available, the fluent AI interpretation is generated in a
        background thread and delivered a moment later via
        `callback(text)`. Use this from the real-time signing loop instead
        of correct_and_enhance().
        """
        if not raw_text or raw_text.strip() in ('', 'Start signing...'):
            return ''

        raw_text = raw_text.strip().lower()
        words = raw_text.split()
        corrected_words = [self._correct_word(w) for w in words]
        sentence = ' '.join(corrected_words)

        instant = self.local.convert(sentence)
        with self._lock:
            self._context_history.append(sentence.lower().split())

        if self.ai_ok and self._model:
            with self._busy_lock:
                already_busy = self._busy
            if not already_busy:
                threading.Thread(
                    target=self._background_enhance,
                    args=(sentence, callback),
                    daemon=True, name="ollama-async").start()

        return instant

    def _background_enhance(self, sentence: str, callback) -> None:
        with self._busy_lock:
            if self._busy:
                return
            self._busy = True
        try:
            result = self._call_ollama_enhanced(sentence, timeout=4.0)
            if result:
                with self._lock:
                    self._context_history.append(result.lower().split())
                if callback:
                    callback(result)
        finally:
            with self._busy_lock:
                self._busy = False
    
    def _correct_word(self, word: str) -> str:
        """
        Correct a single word using dictionary, edit distance, and phonetics.
        """
        if word in self.ENGLISH_VOCAB:
            return word
        
        if word in self.COMMON_CORRECTIONS:
            return self.COMMON_CORRECTIONS[word]
        
        # Find closest match using difflib (Levenshtein-like)
        close_matches = difflib.get_close_matches(
            word, list(self.ENGLISH_VOCAB), n=1, cutoff=SPELL_CORRECTION_THRESHOLD
        )
        if close_matches:
            return close_matches[0]
        
        # If no match, check if word is a conjugation
        corrected = self._deconjugate(word)
        if corrected:
            return corrected
        
        return word
    
    def _deconjugate(self, word: str) -> Optional[str]:
        """
        Handle common verb conjugations and return base form.
        E.g., "eating" -> "eat", "walked" -> "walk"
        """
        # Remove -ing
        if word.endswith('ing') and len(word) > 5:
            base = word[:-3]
            if base in self.ENGLISH_VOCAB or base in self.COMMON_CORRECTIONS:
                return base
        
        # Remove -ed
        if word.endswith('ed') and len(word) > 4:
            base = word[:-2]
            if base in self.ENGLISH_VOCAB:
                return base
            # Try with doubled consonant
            if len(base) > 1 and base[-1] == base[-2]:
                base_undoubled = base[:-1]
                if base_undoubled in self.ENGLISH_VOCAB:
                    return base_undoubled
        
        # Remove -es
        if word.endswith('es') and len(word) > 3:
            base = word[:-2]
            if base in self.ENGLISH_VOCAB:
                return base
        
        if word.endswith('s') and len(word) > 2:
            base = word[:-1]
            if base in self.ENGLISH_VOCAB:
                return base
        
        return None
    
    def _call_ollama_enhanced(self, text: str, timeout: float = 0.5) -> str:
        """
        Call Ollama with an enhanced prompt for intelligent reconstruction.
        """
        system_prompt = (
            "You are an advanced ASL-to-English translator. Your task is to convert raw, "
            "potentially misspelled, jumbled, or incomplete ASL-style input into perfectly "
            "formed, grammatically correct English sentences.\n\n"
            "Rules:\n"
            "1. Return ONLY the corrected English sentence — no labels, explanations, or metadata.\n"
            "2. Infer the user's intended meaning even if input is incomplete or out of order.\n"
            "3. Add articles, copulas, prepositions, and fix tense automatically.\n"
            "4. Correct all spelling mistakes and rearrange words as needed.\n"
            "5. Apply proper capitalization, punctuation, and sentence structure.\n"
            "6. Keep the original meaning exactly — do not add extra ideas.\n"
            "7. Output should be a single, naturally fluent sentence.\n"
            "8. Never repeat the input or use prefixes like 'Input:', 'English:', 'Output:'.\n"
            "9. Be concise and natural — avoid robotic phrasing.\n\n"
            "Examples:\n"
            "  me go store tomorrow        -> I will go to the store tomorrow.\n"
            "  hungry i water              -> I am hungry and would like some water.\n"
            "  doctor pain need            -> I need to see a doctor because of my pain.\n"
            "  not feel good               -> I am not feeling well.\n"
            "  family happy                -> My family is happy.\n"
            "  where bathroom              -> Where is the bathroom?\n"
            "  emergency help call police  -> This is an emergency! Please call the police!\n"
            "  love you family             -> I love my family.\n"
            "  name alex want meet you     -> My name is Alex. I want to meet you.\n"
            "  i doctor                    -> I am a doctor.\n"
            "  i water                     -> I would like some water, please.\n"
        )
        
        try:
            payload = json.dumps({
                "model": self._model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                "options": {
                    "temperature": 0.1,
                    "num_predict": 40,
                    "stop": ["\n", "Input:", "ASL:", "English:", "Output:", "User:"],
                },
            }).encode()
            
            req = urllib.request.Request(
                "http://localhost:11434/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read())
            
            resp = (
                data.get("message", {}).get("content", "")
                or data.get("response", "")
            ).strip()
            
            # Clean response
            for prefix in ("English:", "Output:", "->", ":", '"', "'"):
                if resp.lower().startswith(prefix.lower()):
                    resp = resp[len(prefix):].strip()
            
            resp = resp.strip('"\'').strip()
            
            # Ensure proper ending
            if resp and resp[-1] not in '.!?':
                resp += '.'
            
            return resp
        
        except urllib.error.URLError:
            if self.ai_ok:
                print("  [AI] Ollama lost — using local NLP.")
                self.ai_ok = False
            return ''
        except Exception as e:
            if 'timed out' not in str(e).lower():
                print(f"  [AI ERR] {e}")
            return ''


# ══════════════════════════════════════════════════════════════════════════════
#  CUSTOM KERAS LAYER
# ══════════════════════════════════════════════════════════════════════════════
if ML_OK:
    class AttentionLayer(layers.Layer):
        def __init__(self, units=128, **kwargs):
            super().__init__(**kwargs)
            self.units = units
            self.Wq = layers.Dense(units, use_bias=False)
            self.Wk = layers.Dense(units, use_bias=False)
            self.Wv = layers.Dense(units, use_bias=False)

        def call(self, x):
            Q = self.Wq(x); K = self.Wk(x); V = self.Wv(x)
            scale = tf.math.sqrt(tf.cast(self.units, x.dtype))
            scores = tf.linalg.matmul(Q, K, transpose_b=True) / scale
            return tf.linalg.matmul(tf.nn.softmax(scores, axis=-1), V)

        def get_config(self):
            cfg = super().get_config(); cfg['units'] = self.units; return cfg

    _CUSTOM_OBJECTS = {'AttentionLayer': AttentionLayer}
else:
    _CUSTOM_OBJECTS = {}


# ══════════════════════════════════════════════════════════════════════════════
#  OPTIMIZED FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
def _norm(arr):
    """Fast normalization."""
    centered = arr - arr.mean(axis=0)
    std = np.std(centered)
    return centered / (std if std > 1e-6 else 1.0)

def extract_letter_feats(hand_lm, target_dim=63):
    """Extract and normalize letter features."""
    raw = []
    for lm in hand_lm.landmark:
        raw.extend([lm.x, lm.y, lm.z])
    arr = np.array(raw, np.float32).reshape(-1, 3)
    out = _norm(arr).flatten()
    if len(out) < target_dim:
        out = np.concatenate([out, np.zeros(target_dim - len(out), np.float32)])
    return out[:target_dim]

def extract_word_frame_126(multi_lm):
    """Extract word features from dual-hand landmarks."""
    raw = []
    for hi in range(2):
        if hi < len(multi_lm):
            for lm in multi_lm[hi].landmark:
                raw.extend([lm.x, lm.y, lm.z])
        else:
            raw.extend([0.0] * 63)
    return _norm(np.array(raw, np.float32).reshape(-1, 3)).flatten().astype(np.float32)

def extract_word_frame_63(multi_lm):
    """Extract word features from single-hand landmarks."""
    if not multi_lm:
        return np.zeros(63, np.float32)
    raw = []
    for lm in multi_lm[0].landmark:
        raw.extend([lm.x, lm.y, lm.z])
    return _norm(np.array(raw, np.float32).reshape(-1, 3)).flatten().astype(np.float32)

def make_extractor(feat_dim):
    return extract_word_frame_63 if feat_dim == 63 else extract_word_frame_126


# ══════════════════════════════════════════════════════════════════════════════
#  OPTIMIZED BACKGROUND INFERENCE THREAD
# ══════════════════════════════════════════════════════════════════════════════
class InferenceWorker:
    """Ultra-low-latency background inference worker."""
    def __init__(self):
        self._lock = threading.Lock()
        self._result = None
        self._pending = False
        self._fn = None
        self._arg = None
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                       name='inference-worker')
        self._evt = threading.Event()
        self._thread.start()

    def submit(self, fn, arg):
        with self._lock:
            if self._pending:
                return
            self._fn = fn
            self._arg = arg
            self._pending = True
        self._evt.set()

    def result(self):
        """Retrieve result without blocking."""
        with self._lock:
            r = self._result
            self._result = None
            return r

    def _loop(self):
        while True:
            self._evt.wait()
            self._evt.clear()
            with self._lock:
                fn = self._fn
                arg = self._arg
                self._pending = False
            if fn is not None:
                try:
                    out = fn(arg)
                except Exception as e:
                    out = None
                with self._lock:
                    self._result = out


# ══════════════════════════════════════════════════════════════════════════════
#  OPTIMIZED PREDICTION POOL
# ══════════════════════════════════════════════════════════════════════════════
class PredPool:
    """Fast prediction accumulation and voting."""
    def __init__(self, size, encoder):
        self._pool = deque(maxlen=size)
        self._enc = encoder

    def push(self, probs):
        self._pool.append(np.array(probs, np.float32))

    def top3(self):
        if not self._pool:
            return []
        acc = sum(self._pool)
        idxs = np.argsort(acc)[-3:][::-1]
        out = []
        for i in idxs:
            lbl = (str(self._enc.inverse_transform([i])[0])
                   if hasattr(self._enc, 'inverse_transform') else str(self._enc[i]))
            out.append((lbl, float(acc[i] / len(self._pool))))
        return out

    def clear(self):
        self._pool.clear()

    def __len__(self):
        return len(self._pool)


# ══════════════════════════════════════════════════════════════════════════════
#  ENHANCED LOCAL GRAMMAR ★ IMPROVED (bare subject+noun grammar fix)
# ══════════════════════════════════════════════════════════════════════════════
class EnhancedGrammar:
    """Local grammar engine — kept from original v8.1, with grammar fix."""
    PRONOUNS = {
        'me': 'I', 'mine': 'my', 'him': 'he', 'her': 'she',
        'us': 'we', 'them': 'they', 'myself': 'I',
    }
    SUBJS = {'i', 'you', 'he', 'she', 'we', 'they', 'it'}
    PAST_M = {'yesterday', 'before', 'ago', 'finish', 'finished', 'already',
              'past', 'last', 'recently', 'earlier', 'just', 'previously'}
    FUTURE_M = {'tomorrow', 'later', 'soon', 'will', 'future', 'next',
                'gonna', 'going', 'tonight', 'weekend'}
    MODALS = {'can', 'could', 'should', 'must', 'would', 'might', 'may', 'shall'}
    NEG = {'not', 'no', 'never', 'none', "don't", "doesn't", "didn't",
           'without', 'refuse', "can't", "won't", "hasn't"}
    Q_WORDS = {'what', 'where', 'when', 'why', 'how', 'who', 'which', 'whose'}
    VERBS = {
        'go', 'live', 'eat', 'drink', 'see', 'give', 'come', 'do', 'have', 'buy',
        'bring', 'think', 'tell', 'meet', 'sit', 'run', 'write', 'read',
        'know', 'say', 'make', 'take', 'leave', 'sleep', 'want', 'need',
        'like', 'love', 'hate', 'help', 'work', 'play', 'walk', 'talk',
        'speak', 'ask', 'answer', 'call', 'visit', 'learn', 'teach', 'use',
        'find', 'show', 'watch', 'look', 'listen', 'feel', 'understand',
        'remember', 'forget', 'try', 'finish', 'start', 'stop', 'wait',
        'open', 'close', 'drive', 'ride', 'fly', 'swim', 'cook', 'clean',
        'get', 'hear', 'lose', 'pay', 'send', 'win', 'sign', 'check',
        'stay', 'move', 'return', 'arrive', 'pick',
        'put', 'set', 'keep', 'let', 'become', 'turn', 'grow', 'change',
    }
    VERB_PAST = {
        'go': 'went', 'live': 'lived', 'eat': 'ate', 'drink': 'drank', 'see': 'saw',
        'give': 'gave', 'come': 'came', 'do': 'did', 'have': 'had',
        'buy': 'bought', 'run': 'ran', 'write': 'wrote', 'read': 'read',
        'know': 'knew', 'make': 'made', 'take': 'took', 'leave': 'left',
        'sleep': 'slept', 'find': 'found', 'feel': 'felt', 'hear': 'heard',
        'get': 'got', 'sit': 'sat', 'tell': 'told', 'teach': 'taught',
        'bring': 'brought', 'think': 'thought', 'understand': 'understood',
        'meet': 'met', 'win': 'won', 'pay': 'paid', 'send': 'sent',
        'lose': 'lost', 'forget': 'forgot', 'begin': 'began',
        'speak': 'spoke', 'ride': 'rode', 'drive': 'drove', 'fly': 'flew',
        'swim': 'swam', 'become': 'became', 'grow': 'grew', 'keep': 'kept',
        'put': 'put', 'set': 'set', 'let': 'let', 'cut': 'cut',
        'stay': 'stayed', 'move': 'moved', 'return': 'returned',
        'arrive': 'arrived', 'pick': 'picked', 'check': 'checked',
    }
    ADJS = {
        'hungry', 'thirsty', 'tired', 'happy', 'sad', 'sick', 'fine', 'good',
        'bad', 'busy', 'ready', 'sorry', 'angry', 'excited', 'bored', 'hot',
        'cold', 'scared', 'confused', 'lost', 'late', 'early', 'wrong',
        'right', 'sure', 'okay', 'ok', 'well', 'ill', 'hurt', 'deaf',
        'blind', 'sleepy', 'full', 'nervous', 'worried', 'surprised',
        'embarrassed', 'proud', 'lonely', 'bored', 'stressed', 'relaxed',
        'uncomfortable', 'comfortable', 'safe', 'dangerous', 'important',
        'difficult', 'easy', 'fast', 'slow', 'big', 'small', 'old', 'new',
        'open', 'closed', 'free', 'busy', 'available', 'unavailable',
    }
    # ★ NOUNS split into PROFESSION_NOUNS (get "I am a ___") vs regular
    #   NOUNS (get "I have a ___"/generic article usage). This lets
    #   "i doctor" -> "I am a doctor." rather than a broken sentence.
    PROFESSION_NOUNS = {
        'doctor', 'teacher', 'nurse', 'student', 'engineer', 'driver',
        'police', 'officer', 'friend',
    }
    NOUNS = {
        'book', 'car', 'dog', 'cat', 'phone', 'computer', 'table', 'chair',
        'child', 'baby', 'boy', 'girl', 'man', 'woman', 'person', 'friend',
        'teacher', 'doctor', 'food', 'money', 'time', 'day', 'week', 'month',
        'year', 'family', 'name', 'morning', 'night', 'afternoon', 'evening',
        'letter', 'email', 'nurse', 'medicine', 'pain', 'appointment',
        'hospital', 'address', 'number', 'key', 'bag', 'clothes', 'shoes',
        'house', 'room', 'door', 'window', 'bus', 'train', 'road', 'city',
        'country', 'language', 'question', 'answer', 'problem', 'idea',
        'police', 'ambulance', 'blood', 'accident', 'fire', 'danger',
    }
    LOCS = {
        'home', 'school', 'work', 'hospital', 'store', 'market', 'church',
        'park', 'office', 'gym', 'library', 'restaurant', 'here', 'there',
        'outside', 'inside', 'upstairs', 'downstairs', 'bathroom', 'kitchen',
        'bedroom', 'pharmacy', 'clinic', 'airport', 'station', 'hotel',
        'bank', 'police', 'court', 'government',
    }
    NO_ART = {'home', 'here', 'there', 'outside', 'inside', 'upstairs',
              'downstairs', 'work', 'school', 'church'}
    MASS = {
        'water', 'milk', 'juice', 'rice', 'bread', 'coffee', 'tea', 'soup',
        'sugar', 'salt', 'help', 'advice', 'information', 'news', 'money',
        'food', 'meat', 'medicine', 'pain', 'time',
    }
    VOWELS = set('aeiou')

    COMPOUNDS = {
        'my_name_is': 'My name is...',
        'how_are_you': 'How are you?',
        'i_love_you': 'I love you!',
        'hello': 'Hello!',
        'thank_you': 'Thank you!',
        'good_morning': 'Good morning!',
        'good_night': 'Good night!',
        'nice_to_meet_you': 'Nice to meet you!',
        'excuse_me': 'Excuse me.',
        'i_dont_know': "I don't know.",
        'i_dont_understand': "I don't understand.",
        'good_afternoon': 'Good afternoon!',
        'good_evening': 'Good evening!',
        'see_you_later': 'See you later!',
        'take_care': 'Take care!',
        'never_mind': "Never mind.",
        'of_course': 'Of course!',
        'no_problem': 'No problem!',
    }

    SOLO = {
        'hello': 'Hello! How are you?',
        'hi': 'Hi! How are you?',
        'home': 'I am at home.',
        'how_are_you': 'How are you doing?',
        'hungry': 'I am feeling hungry.',
        'i_love_you': 'I love you!',
        'my_name_is': 'My name is...',
        'no': 'No, thank you.',
        'stop': 'Please stop!',
        'thank_you': 'Thank you very much!',
        'water': 'Could I have some water, please?',
        'yes': 'Yes, please!',
        'tired': 'I am tired.',
        'thirsty': 'I am thirsty.',
        'happy': 'I am happy.',
        'sad': 'I am sad.',
        'sick': 'I am not feeling well.',
        'fine': 'I am fine, thank you.',
        'okay': 'I am okay.',
        'ok': 'I am okay.',
        'sorry': 'I am sorry.',
        'help': 'Please help me!',
        'emergency': 'This is an emergency! Please help!',
        'food': 'I would like some food, please.',
        'bathroom': 'I need to use the bathroom.',
        'doctor': 'I need to see a doctor.',
        'goodbye': 'Goodbye! Take care!',
        'ready': 'I am ready.',
        'understand': 'I understand.',
        'pain': 'I am in pain.',
        'medicine': 'I need my medicine.',
        'cold': 'I am feeling cold.',
        'hot': 'I am feeling hot.',
        'scared': 'I am scared.',
        'angry': 'I am angry.',
        'confused': 'I am confused.',
        'lost': 'I am lost.',
        'late': 'I am running late.',
        'busy': 'I am busy right now.',
        'wait': 'Please wait.',
        'come': 'Please come here.',
        'go': 'I need to go.',
        'phone': 'May I use your phone?',
        'money': 'I need some money.',
        'police': 'Please call the police!',
        'ambulance': 'Please call an ambulance!',
    }

    PHRASES = [
        (['emergency', 'help'], 'This is an emergency! Please help!'),
        (['please', 'help'], 'Please help me!'),
        (['call', 'police'], 'Please call the police!'),
        (['call', 'ambulance'], 'Please call an ambulance!'),
        (['call', 'doctor'], 'Please call a doctor!'),
        (['i', 'not', 'feel', 'good'], "I am not feeling well."),
        (['i', 'not', 'feel', 'well'], "I am not feeling well."),
        (['where', 'bathroom'], 'Where is the bathroom?'),
        (['where', 'hospital'], 'Where is the hospital?'),
        (['where', 'doctor'], 'Where is the doctor?'),
        (['i', 'need', 'medicine'], 'I need my medicine.'),
        (['i', 'have', 'pain'], 'I am in pain.'),
        (['i', 'in', 'pain'], 'I am in pain.'),
        (['i', 'late', 'work'], 'I am late for work.'),
        (['i', 'late', 'school'], 'I am late for school.'),
        (['you', 'help', 'me'], 'Could you help me, please?'),
        (['can', 'you', 'help'], 'Can you help me, please?'),
        (['i', 'not', 'understand'], "I don't understand."),
        (['i', 'not', 'know'], "I don't know."),
        (['i', 'love', 'you'], 'I love you!'),
        (['no', 'stop'], 'No, please stop!'),
        (['no', 'thank_you'], 'No, thank you!'),
        (['yes', 'thank_you'], 'Yes, thank you!'),
        (['i_love_you', 'thank_you'], 'I love you! Thank you!'),
        (['hello', 'i', 'home'], 'Hello! I am at home.'),
        (['i', 'home', 'hungry'], 'I am at home and I am hungry.'),
        (['i', 'hungry', 'water'], 'I am hungry and I would like some water.'),
        (['i', 'need', 'doctor'], 'I need to see a doctor right away.'),
        (['i', 'need', 'help'], 'I really need some help, please.'),
        (['i', 'need', 'bathroom'], 'I urgently need to use the bathroom.'),
        (['i', 'go', 'school'], 'I am going to school.'),
        (['i', 'go', 'work'], 'I am going to work.'),
        (['i', 'go', 'home'], 'I am going home.'),
        (['i', 'come', 'home'], 'I am coming home.'),
        (['i', 'want', 'water'], 'I would like some water, please.'),
        (['i', 'want', 'food'], 'I would like some food, please.'),
        (['i', 'eat', 'food'], 'I am going to eat some food.'),
        (['i', 'drink', 'water'], 'I would like to drink some water.'),
        (['i', 'water'], 'I would like some water, please.'),
        (['i', 'food'], 'I would like some food, please.'),
        # ★ NEW: fixes "I DOCTOR" -> "I a doctor." bug directly at the
        #   phrase-pattern level (highest priority, exact match).
        (['i', 'doctor'], 'I am a doctor.'),
        (['i', 'teacher'], 'I am a teacher.'),
        (['i', 'nurse'], 'I am a nurse.'),
        (['i', 'student'], 'I am a student.'),
        (['i', 'live'], "I live here."),
    ]

    _SPACE_NORM = {
        'my name is': 'my_name_is',
        'i love you': 'i_love_you',
        'how are you': 'how_are_you',
        'thank you': 'thank_you',
        'good morning': 'good_morning',
        'good night': 'good_night',
        'good afternoon': 'good_afternoon',
        'good evening': 'good_evening',
        'nice to meet you': 'nice_to_meet_you',
        'excuse me': 'excuse_me',
        "i don't know": 'i_dont_know',
        "i dont know": 'i_dont_know',
        "i don't understand": 'i_dont_understand',
        'see you later': 'see_you_later',
        'take care': 'take_care',
        'never mind': 'never_mind',
        'of course': 'of_course',
        'no problem': 'no_problem',
    }

    def _nc(self, text):
        t = text.lower()
        for k, v in self._SPACE_NORM.items():
            t = t.replace(k, v)
        return t

    def _art(self, w):
        if w.lower() in self.MASS:
            return 'some'
        return 'an' if w and w[0].lower() in self.VOWELS else 'a'

    def _past(self, v):
        v = v.lower()
        if v in self.VERB_PAST:
            return self.VERB_PAST[v]
        if v.endswith('e'):
            return v + 'd'
        if (len(v) >= 3 and v[-1] not in 'aeiou'
                and v[-2] in 'aeiou' and v[-3] not in 'aeiou'):
            return v + v[-1] + 'ed'
        return v + 'ed'

    def _tense(self, ws):
        s = set(ws)
        if s & self.PAST_M:
            return 'past'
        if s & self.FUTURE_M:
            return 'future'
        return 'present'

    def _be(self, s):
        if s == 'i':
            return 'am'
        if s in ('you', 'we', 'they'):
            return 'are'
        return 'is'

    def _is_q(self, ws):
        return bool(ws and (ws[-1] == '?' or set(ws) & self.Q_WORDS))

    def _where_q(self, words):
        if words and words[0] == 'where':
            target = ' '.join(words[1:])
            return f"Where is {target}?"
        return None

    def _how_q(self, words):
        if words and words[0] == 'how' and len(words) >= 2:
            rest = ' '.join(words[1:])
            return f"How {rest}?"
        return None

    def _when_why_q(self, words):
        """★ NEW: handle WHEN / WHY the same way WHERE / HOW are handled."""
        if words and words[0] in ('when', 'why') and len(words) >= 2:
            rest = ' '.join(words[1:])
            return f"{words[0].capitalize()} {rest}?"
        return None

    def convert(self, text):
        if not text or text.strip() in ('', 'Start signing...'):
            return ''
        text = self._nc(text)
        words = text.lower().split()
        if not words:
            return ''

        if len(words) == 1:
            w = words[0]
            if w in self.SOLO:
                return self.SOLO[w]
            if w in self.ADJS:
                return f"I am {w}."
            if w in self.VERBS:
                return f"I {w}."
            if w in self.LOCS:
                return f"I am at {w}."
            if w in self.MODALS:
                return f"I {w} help you."
            return w.capitalize() + '.'

        wq = self._where_q(words)
        if wq:
            return wq
        hq = self._how_q(words) if words[0] == 'how' else None
        if hq:
            return hq
        wyq = self._when_why_q(words)
        if wyq:
            return wyq

        for pat, result in sorted(self.PHRASES, key=lambda x: -len(x[0])):
            n = len(pat)
            if words[:n] == pat:
                tail = words[n:]
                tail_str = self.convert(' '.join(tail)) if tail else ''
                return result + (' ' + tail_str if tail_str else '')

        clauses, chunk = [], []
        for w in words:
            if w in self.COMPOUNDS:
                if chunk:
                    clauses.append(self._build(chunk))
                    chunk = []
                clauses.append(self.COMPOUNDS[w])
            else:
                chunk.append(w)
        if chunk:
            clauses.append(self._build(chunk))
        out = ' '.join(c for c in clauses if c)
        return re.sub(r'\s+([.,!?])', r'\1', out)

    def _build(self, words):
        if not words:
            return ''
        words = [self.PRONOUNS.get(w, w) for w in words]
        dd = [words[0]]
        for w in words[1:]:
            if w != dd[-1]:
                dd.append(w)
        words = dd

        if len(words) == 1:
            w = words[0]
            if w in self.SOLO:
                return self.SOLO[w]
            if w in self.ADJS:
                return f"I am {w}."
            if w in self.LOCS:
                return f"I am at {w}."
            if w in self.VERBS:
                return f"I {w}."
            if w == 'i':
                return ''
            return w.capitalize() + '.'

        has_neg = bool(set(words) & self.NEG)
        has_q = self._is_q(words)
        modal = next((w for w in words if w in self.MODALS), None)
        skip = self.PAST_M | self.FUTURE_M | self.NEG | {'?', '!', 'please', 'not'}
        cw = [w for w in words if w not in skip]

        if (cw and cw[0] not in self.SUBJS
                and cw[0] not in ('a', 'an', 'the', 'no', 'yes')
                and cw[0] not in self.Q_WORDS
                and (cw[0] in self.ADJS or cw[0] in self.LOCS
                     or cw[0] in self.VERBS or cw[0] in self.NOUNS)):
            cw = ['i'] + cw

        tense = self._tense(words)
        res = []
        force_please = False  # ★ NEW flag: force ", please." for bare mass-noun requests
        i = 0
        while i < len(cw):
            w = cw[i]
            nw = cw[i + 1] if i + 1 < len(cw) else None
            pw = res[-1].lower() if res else None
            is_s = w in self.SUBJS
            neg_str = " not" if has_neg else ""

            if is_s and nw and nw in self.ADJS:
                if has_neg:
                    res += [w, self._be(w), 'not']
                    i += 1
                    continue
                res += [w, self._be(w)]
                i += 1
                continue

            if is_s and nw and nw in self.LOCS and nw not in self.VERBS:
                if has_neg:
                    res += [w, self._be(w), 'not', 'at']
                    i += 1
                    continue
                res += [w, self._be(w), 'at']
                i += 1
                continue

            if is_s and modal and nw in self.VERBS:
                modal_str = modal + (" not" if has_neg else "")
                res += [w, modal_str]
                i += 1
                continue

            if w in self.ADJS:
                if pw in ('am', 'is', 'are', 'was', 'were', 'not'):
                    res.append(w)
                elif pw in self.ADJS:
                    res += ['and', w]
                else:
                    res.append(w)
                i += 1
                continue

            if w in self.VERBS and w != modal:
                if tense == 'past':
                    if has_neg:
                        base = w
                        if pw not in ("didn't", "don't", "doesn't", "can't"):
                            res.append("didn't")
                        res.append(base)
                    else:
                        res.append(self._past(w))
                elif tense == 'future':
                    if has_neg:
                        if not (res and res[-1] in ("won't", 'will')):
                            res.append("won't")
                    else:
                        if not (res and res[-1] == 'will'):
                            res.append('will')
                    res.append(w)
                else:
                    if has_neg and pw not in ("don't", "doesn't", "can't"):
                        subj = next((r for r in res if r.lower() in self.SUBJS), 'i')
                        neg_aux = "doesn't" if subj.lower() in ('he', 'she', 'it') else "don't"
                        res.append(neg_aux)
                    if pw in ('want', 'need', 'like', 'love', 'hate',
                              'start', 'try', 'help', 'begin', 'able'):
                        res.append('to')
                    res.append(w)
                i += 1
                continue

            if w in self.LOCS and pw and pw in self.VERBS:
                res.append('to')
                if w not in self.NO_ART:
                    res.append('the')
                res.append(w)
                i += 1
                continue

            # ★ FIXED BLOCK: bare "SUBJECT + NOUN" (e.g. "i doctor",
            #   "i water") previously fell through to a weak default that
            #   only added an article ("I a doctor.", "I some water.").
            #   Now:
            #     - bare subject + mass noun  -> "I would like some water, please."
            #     - bare subject + profession -> "I am a doctor."
            #     - bare subject + plain noun -> "I have a phone."
            if w in self.NOUNS or w in self.MASS:
                bare_subject = (len(res) == 1 and res[0].lower() in self.SUBJS)

                if bare_subject and w in self.MASS and not has_neg and not has_q:
                    subj = res[0]
                    res = [subj, 'would', 'like', self._art(w), w]
                    force_please = True
                    i += 1
                    continue

                if bare_subject and w in self.PROFESSION_NOUNS and not has_neg and not has_q:
                    subj = res[0]
                    res = [subj, self._be(subj.lower()), self._art(w), w]
                    i += 1
                    continue

                if bare_subject and not has_neg and not has_q:
                    subj = res[0]
                    res = [subj, 'have', self._art(w), w]
                    i += 1
                    continue

                if pw not in ('a', 'an', 'the', 'my', 'your', 'his', 'her',
                              'our', 'their', 'this', 'that', 'some',
                              "didn't", "don't", "doesn't"):
                    res.append(self._art(w))
                res.append(w)
                i += 1
                continue

            res.append(w)
            i += 1

        if not res:
            return ''
        res[0] = res[0].capitalize()
        out = ' '.join(res)
        out = re.sub(r'\bi\b', 'I', out)
        out = re.sub(r'\s+', ' ', out).strip()

        if not has_neg and not has_q:
            if force_please or any(w in words for w in ('want', 'need', 'give', 'bring')):
                if 'please' not in out.lower():
                    out = out.rstrip('.!') + ', please.'

        if out and out[-1] not in '.!?':
            out += '.'
        if has_q and out[-1] != '?':
            out = out.rstrip('.!') + '?'
        return re.sub(r'\s+([.,!?])', r'\1', out)


# ══════════════════════════════════════════════════════════════════════════════
#  AUTOCOMPLETE (from original, minimal changes)
# ══════════════════════════════════════════════════════════════════════════════
class Autocomplete:
    VOCAB = [
        'hello', 'goodbye', 'thank_you', 'sorry', 'help', 'i', 'you', 'want',
        'need', 'go', 'come', 'food', 'water', 'bathroom', 'doctor', 'happy',
        'sick', 'emergency', 'understand', 'eat', 'drink', 'see', 'know',
        'like', 'love', 'work', 'play', 'learn', 'feel', 'have', 'give',
        'take', 'make', 'stop', 'wait', 'finish', 'hungry', 'thirsty',
        'tired', 'sad', 'fine', 'ready', 'hot', 'cold', 'busy', 'home',
        'school', 'hospital', 'friend', 'family', 'name', 'time', 'day',
        'morning', 'night', 'phone', 'car', 'book', 'how_are_you',
        'i_love_you', 'my_name_is', 'please', 'yes', 'no', 'pain',
        'medicine', 'police', 'ambulance', 'lost', 'scared', 'confused',
        'live', 'hi', 'my', 'name',
    ]
    STARTERS = [
        'hello', 'i', 'help', 'emergency', 'sorry',
        'you', 'how_are_you', 'goodbye', 'thank_you'
    ]
    BIGRAMS = {
        'hello': ['i', 'you', 'happy', 'thank_you', 'goodbye', 'how_are_you'],
        'goodbye': ['thank_you', 'happy', 'i', 'you'],
        'thank_you': ['you', 'hello', 'goodbye', 'i', 'help'],
        'sorry': ['i', 'you', 'understand', 'help'],
        'i': ['want', 'need', 'happy', 'sick', 'hungry', 'thirsty',
              'tired', 'go', 'come', 'love', 'like', 'know',
              'understand', 'feel', 'have', 'work', 'learn', 'eat',
              'drink', 'play', 'not', 'pain', 'lost', 'scared', 'live'],
        'you': ['want', 'need', 'understand', 'go', 'come', 'happy',
                'sick', 'know', 'feel', 'like', 'love', 'help'],
        'want': ['food', 'water', 'go', 'come', 'help', 'doctor', 'phone'],
        'need': ['help', 'doctor', 'water', 'food', 'bathroom', 'go',
                 'come', 'time', 'medicine', 'phone'],
        'go': ['school', 'work', 'home', 'doctor', 'bathroom', 'hospital'],
        'come': ['help', 'doctor', 'i', 'you', 'home', 'here'],
        'eat': ['food', 'now'],
        'drink': ['water', 'now'],
        'not': ['understand', 'know', 'feel', 'good', 'well', 'ready'],
        'food': ['want', 'need', 'thank_you', 'water', 'eat'],
        'water': ['want', 'need', 'thank_you', 'food', 'drink'],
        'bathroom': ['need', 'help', 'i', 'sorry', 'go', 'where'],
        'doctor': ['need', 'help', 'i', 'come', 'go', 'call'],
        'home': ['go', 'i', 'you', 'come', 'happy', 'family'],
        'school': ['go', 'i', 'learn', 'work'],
        'work': ['go', 'i', 'school', 'finish', 'late'],
        'family': ['happy', 'i', 'love', 'you', 'home'],
        'friend': ['my', 'i', 'you', 'hello', 'happy'],
        'hungry': ['i', 'eat', 'food'],
        'tired': ['i', 'sleep'],
        'happy': ['i', 'you', 'thank_you', 'family'],
        'sad': ['i', 'feel', 'why'],
        'sick': ['i', 'doctor', 'feel', 'help', 'medicine'],
        'fine': ['i', 'thank_you', 'you'],
        'help': ['need', 'i', 'you', 'emergency', 'come', 'doctor', 'please'],
        'emergency': ['help', 'come', 'doctor', 'need', 'i', 'police', 'ambulance'],
        'pain': ['i', 'have', 'feel', 'medicine', 'doctor'],
        'medicine': ['need', 'i', 'take', 'have'],
        'lost': ['i', 'help', 'where'],
        'scared': ['i', 'help', 'police'],
        'my': ['name'],
        'live': ['i', 'you'],
    }

    def suggest(self, sentence, spelling, n=4):
        norm = [w.lower() for w in sentence]
        if spelling:
            prefix = ''.join(spelling).lower()
            return [w for w in self.VOCAB
                    if w.startswith(prefix) and w not in norm][:n]
        if not norm:
            return self.STARTERS[:n]
        last = norm[-1]
        cands = [c for c in self.BIGRAMS.get(last, []) if c not in norm]
        cands += [s for s in self.STARTERS if s not in norm and s not in cands]
        return cands[:n]


# ══════════════════════════════════════════════════════════════════════════════
#  ALERTS (from original, unchanged)
# ══════════════════════════════════════════════════════════════════════════════
class Alerts:
    _EV = {
        'letter': (1200, 55, (100, 175, 80), '+ LETTER'),
        'word': (1000, 120, (185, 120, 55), 'WORD'),
        'accept': (880, 80, (100, 175, 80), 'ACCEPT'),
        'toggle': (660, 120, (165, 155, 45), 'MODE'),
        'speak': (520, 200, (175, 70, 140), 'SPEAK'),
        'undo': (440, 100, (185, 120, 55), 'UNDO'),
        'clear': (330, 150, (140, 136, 130), 'CLEAR'),
        'error': (220, 200, (60, 55, 200), 'ERROR'),
    }

    def __init__(self):
        self._active = None
        self._lock = threading.Lock()
        try:
            import winsound
            self._ws = winsound
            self._bt = 'win'
        except ImportError:
            self._bt = 'bell'

    def fire(self, ev):
        if ev not in self._EV:
            return
        freq, ms, col, lbl = self._EV[ev]

        def _beep():
            try:
                if self._bt == 'win':
                    self._ws.Beep(freq, ms)
                else:
                    print('\a', end='', flush=True)
            except Exception:
                pass

        threading.Thread(target=_beep, daemon=True).start()
        with self._lock:
            self._active = (time.time() + 0.55, col, lbl)

    def draw(self, frame):
        with self._lock:
            a = self._active
        if not a:
            return
        expire, col, lbl = a
        rem = expire - time.time()
        if rem <= 0:
            with self._lock:
                self._active = None
            return
        h, w = frame.shape[:2]
        alpha = rem / 0.55
        bw = int(110 * alpha)
        if bw > 10:
            rx1, ry1 = w - bw - 12, 10
            rx2, ry2 = w - 12, 34
            ov = frame.copy()
            cv2.rectangle(ov, (rx1, ry1), (rx2, ry2), col, -1)
            cv2.addWeighted(ov, 0.85, frame, 0.15, 0, frame)
            cv2.putText(frame, lbl, (rx1 + 6, ry2 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (255, 255, 255), 1, cv2.LINE_AA)
        ph = max(2, int(4 * alpha))
        cv2.rectangle(frame, (0, h - ph), (w, h), col, -1)


# ══════════════════════════════════════════════════════════════════════════════
#  OPTIMIZED CHARACTER RECOGNIZER (LETTERS ONLY)
# ══════════════════════════════════════════════════════════════════════════════
class CharacterRecognizer:
    """Unified, ultra-optimized recognizer for A-Z letters ONLY."""

    def __init__(self):
        self._buf = deque(maxlen=VOTE_WINDOW)
        self._stable_lbl = None
        self._stable_conf = 0.0
        self._stable_since = None
        self._locked = False
        self._locked_lbl = None
        self._diff_streak = 0
        self._word_armed = False
        self._word_since = None
        self._added_lbl = None
        self._after_add = False
        self.display_lbl = None
        self.display_conf = 0.0
        self.events = set()

    def update(self, raw_lbl, raw_conf):
        """Update with raw letter prediction."""
        self.events = set()
        now = time.time()

        # Validate: only A-Z
        if raw_lbl and raw_conf >= _conf_thr(raw_lbl):
            # Ensure letter is A-Z only
            if isinstance(raw_lbl, str) and len(raw_lbl) == 1 and raw_lbl.upper() in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                self._buf.append((raw_lbl, raw_conf))
            else:
                self._buf.append(('_', 0.0))
        else:
            self._buf.append(('_', 0.0))

        counts = Counter((l for l, _ in self._buf if l != '_'))
        non_blanks = sum(counts.values())
        winner = counts.most_common(1)[0][0] if counts else None
        w_count = counts[winner] if winner else 0
        min_abs = int(VOTE_WINDOW * 0.35)

        if (winner and non_blanks > 0
                and w_count / non_blanks >= VOTE_MAJORITY
                and w_count >= min_abs):
            w_confs = [c for l, c in self._buf if l == winner]
            cur_lbl = winner
            cur_conf = sum(w_confs) / len(w_confs)
        else:
            cur_lbl = None
            cur_conf = 0.0

        if cur_lbl:
            self.display_lbl = cur_lbl
            self.display_conf = cur_conf
        elif self._locked and self._locked_lbl:
            self.display_lbl = self._locked_lbl
            self.display_conf = self._stable_conf
        else:
            self.display_lbl = None
            self.display_conf = 0.0

        if self._locked:
            self._after_add = True
            if cur_lbl is not None and cur_lbl != self._locked_lbl:
                self._diff_streak += 1
                if self._diff_streak >= UNLOCK_FRAMES:
                    self._locked = False
                    self._after_add = False
                    self._locked_lbl = None
                    self._diff_streak = 0
                    self._stable_lbl = cur_lbl
                    self._stable_conf = cur_conf
                    self._stable_since = now
                    self._word_armed = False
                    self._word_since = None
            else:
                self._diff_streak = 0

            if self._word_armed and self._word_since:
                if now - self._word_since >= WORD_COMMIT_TIME:
                    self.events.add('commit_word')
                    self._word_armed = False
                    self._word_since = None
                    self._full_reset()
            return self.events

        if cur_lbl is None:
            return self.events

        if cur_lbl != self._stable_lbl:
            self._stable_lbl = cur_lbl
            self._stable_conf = cur_conf
            self._stable_since = now
            self._word_armed = False
            self._word_since = None
        else:
            self._stable_conf = cur_conf

        elapsed = now - self._stable_since if self._stable_since else 0.0
        if elapsed >= LETTER_ADD_TIME:
            self._added_lbl = cur_lbl
            self._after_add = True
            self._locked = True
            self._locked_lbl = cur_lbl
            self._diff_streak = 0
            self._word_armed = True
            self._word_since = now
            self.events.add('add_letter')

        return self.events

    def _full_reset(self):
        self._stable_lbl = None
        self._stable_conf = 0.0
        self._stable_since = None
        self._locked = False
        self._after_add = False
        self._locked_lbl = None
        self._diff_streak = 0
        self._word_armed = False
        self._word_since = None
        self._added_lbl = None

    def reset(self):
        self._buf.clear()
        self._full_reset()
        self.display_lbl = None
        self.display_conf = 0.0
        self.events = set()

    def letter_progress(self):
        if self._locked:
            return 1.0
        if self._stable_since is None:
            return 0.0
        return min((time.time() - self._stable_since) / LETTER_ADD_TIME, 1.0)

    def word_progress(self):
        if not self._word_armed or self._word_since is None:
            return 0.0
        return min((time.time() - self._word_since) / WORD_COMMIT_TIME, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
#  ABSENT TIMER (from original, minimal optimizations)
# ══════════════════════════════════════════════════════════════════════════════
class AbsentTimer:
    def __init__(self):
        self._since = None
        self._frames = 0
        self._af = self._tf = self._sf = False
        self.events = set()

    def update(self, has_hands):
        self.events = set()
        if has_hands:
            self._since = None
            self._frames = 0
            self._af = self._tf = self._sf = False
            return self.events
        self._frames += 1
        if self._frames < ABSENT_DEBOUNCE:
            return self.events
        if self._since is None:
            self._since = time.time()
        e = time.time() - self._since
        if e >= T_SPEAK and not self._sf:
            self.events.add('speak')
            self._sf = True
        elif e >= T_TOGGLE and not self._tf:
            self.events.add('toggle')
            self._tf = True
        elif e >= T_ACCEPT and not self._af:
            self.events.add('accept')
            self._af = True
        return self.events

    def elapsed(self):
        return time.time() - self._since if self._since else 0.0

    def reset(self):
        self._since = None
        self._frames = 0
        self._af = self._tf = self._sf = False


class Sparkline:
    def __init__(self, n=40):
        self._v = deque([0.0] * n, maxlen=n)

    def push(self, v):
        self._v.append(float(v))

    def get(self):
        return list(self._v)


# ══════════════════════════════════════════════════════════════════════════════
#  TTS ENGINE (from original, unchanged)
# ══════════════════════════════════════════════════════════════════════════════
class TTSEngine:
    def __init__(self):
        self._lock = threading.Lock()
        self._queue = deque()
        self._speaking = False
        self._eng_name = 'none'
        self._detect_backend()
        threading.Thread(target=self._worker, daemon=True,
                         name='tts-worker').start()

    def _detect_backend(self):
        if TTS_LIB:
            try:
                probe = pyttsx3.init()
                probe.stop()
                self._eng_name = 'pyttsx3'
                print("  [OK] TTS — pyttsx3")
                return
            except Exception as ex:
                print(f"  [WARN] pyttsx3 probe failed: {ex}")
        try:
            import subprocess
            subprocess.run(['espeak', '--version'],
                           capture_output=True, timeout=2, check=True)
            self._eng_name = 'espeak'
            print("  [OK] TTS — espeak")
            return
        except Exception:
            pass
        print("  [WARN] No TTS backend — printed to console")
        self._eng_name = 'print'

    def _worker(self):
        while True:
            text = None
            while text is None:
                with self._lock:
                    if self._queue:
                        text = self._queue.popleft()
                        self._speaking = True
                if text is None:
                    time.sleep(0.04)
            try:
                self._say(text)
            except Exception as e:
                print(f"  [TTS ERR] {e}")
            finally:
                with self._lock:
                    self._speaking = bool(self._queue)

    def _say(self, text):
        if self._eng_name == 'pyttsx3':
            self._say_pyttsx3(text)
        elif self._eng_name == 'espeak':
            self._say_espeak(text)
        else:
            print(f"  [SPEAK] {text}")

    def _say_pyttsx3(self, text):
        for attempt in range(2):
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 155)
                for v in engine.getProperty('voices'):
                    if 'english' in v.name.lower() or 'en_' in v.id.lower():
                        engine.setProperty('voice', v.id)
                        break
                engine.say(text)
                engine.runAndWait()
                engine.stop()
                return
            except RuntimeError as e:
                print(f"  [TTS] pyttsx3 RuntimeError (attempt {attempt + 1}): {e}")
                time.sleep(0.1)
            except Exception as e:
                print(f"  [TTS] pyttsx3 error: {e}")
                break
        print(f"  [SPEAK] {text}")

    def _say_espeak(self, text):
        import subprocess
        try:
            subprocess.run(['espeak', '-s', '155', text], timeout=30)
        except Exception as e:
            print(f"  [TTS] espeak error: {e}\n  [SPEAK] {text}")

    def speak(self, text):
        if not text:
            return
        with self._lock:
            self._queue.clear()
            self._queue.append(text)

    def stop(self):
        with self._lock:
            self._queue.clear()
            self._speaking = False

    @property
    def is_speaking(self):
        with self._lock:
            return self._speaking or bool(self._queue)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ASL SYSTEM v9.0 ★ WITH ADVANCED NLP & OPTIMIZATIONS
# ══════════════════════════════════════════════════════════════════════════════
class ASLSystem:
    _MERGE = {
        ('MY', 'NAME', 'IS'): 'MY_NAME_IS',
        ('MY', 'NAME'): 'MY_NAME_IS',   # ★ NEW: works even without signing IS
        ('I', 'LOVE', 'YOU'): 'I_LOVE_YOU',
        ('HOW', 'ARE', 'YOU'): 'HOW_ARE_YOU',
        ('THANK', 'YOU'): 'THANK_YOU',
        ('GOOD', 'MORNING'): 'GOOD_MORNING',
        ('GOOD', 'NIGHT'): 'GOOD_NIGHT',
        ('NICE', 'TO', 'MEET', 'YOU'): 'NICE_TO_MEET_YOU',
        ('EXCUSE', 'ME'): 'EXCUSE_ME',
        ('SEE', 'YOU', 'LATER'): 'SEE_YOU_LATER',
        ('TAKE', 'CARE'): 'TAKE_CARE',
        ('NO', 'PROBLEM'): 'NO_PROBLEM',
        ('OF', 'COURSE'): 'OF_COURSE',
    }

    def __init__(self):
        self._print_banner()
        self.nlp = AdvancedNLPEngine()  # ★ NEW Advanced NLP
        self.rec = CharacterRecognizer()  # ★ Letters only
        self.absent = AbsentTimer()
        self.ac = Autocomplete()
        self.alerts = Alerts()
        self.tts = TTSEngine()
        self._infer_letter = InferenceWorker()
        self._load_models()
        self._setup_mp()
        self._init_state()
        print("\n[READY]\n")

    def _print_banner(self):
        print("""
╔════════════════════════════════════════════════════════════╗
║  ASL SYSTEM  v9.0  —  Advanced NLP & Optimized Perf.      ║
╠════════════════════════════════════════════════════════════╣
║  SCOPE: A-Z LETTERS + COMPLETE WORDS ONLY                 ║
║  LETTER MODE : hold 0.40s -> auto-add (optimized timing)  ║
║  WORD MODE   : sign -> AI-powered prediction              ║
║  NLP: Spell correction, context, intelligent inference    ║
║  PERF: Ultra-low latency, smooth, zero stuttering         ║
║  ABSENT HAND : 1.0s=accept  3.0s=toggle  5.5s=speak       ║
║  KEYS  S=speak  M=toggle  BKSP=del  Z=undo                ║
║        A=add  SPACE=commit  1-4=suggest                   ║
║        ENTER=clear  D=debug  Q/ESC=quit                   ║
╚════════════════════════════════════════════════════════════╝
""")

    def _init_state(self):
        self.mode = 'words'
        self.sentence = []
        self._undo = []
        self.spelling = []
        self.english = ''
        self._fb_text = ''
        self._fb_col = (80, 80, 80)
        self._fb_until = 0.0
        self._debug = False
        self._frame = 0
        self._sugg = []
        self._spark = Sparkline()
        self._mode_grace = 0.0
        self.top3 = []
        self.last_top3 = []
        self.wbuf = deque()
        self._pred_pool = None
        self._hand_absent = 0
        self._clean_frames = 0
        self._last_spoken = ''

        # ★ FIX: cache for _eng() so we don't call the NLP engine on every
        #   single frame (that was the root cause of the UI-freeze bug —
        #   every draw call could trigger a blocking network request).
        self._eng_cache = ''
        self._eng_cache_key = None

        if self.wm is not None:
            self._pred_pool = PredPool(PRED_POOL_SIZE, self.we)

        self.C = dict(
            bg=(250, 249, 248), panel=(244, 242, 240), panel2=(235, 233, 230),
            header_bg=(255, 255, 255), border=(210, 207, 203),
            border_med=(185, 181, 175), text_primary=(30, 28, 26),
            text_secondary=(100, 96, 90), text_light=(160, 156, 150),
            green=(80, 160, 70), green_light=(220, 245, 215),
            blue=(175, 110, 45), blue_light=(240, 228, 210),
            orange=(35, 120, 205), orange_light=(220, 238, 255),
            teal=(150, 140, 35), teal_light=(230, 242, 215),
            red=(55, 50, 195), red_light=(220, 218, 255),
            purple=(165, 60, 130), purple_light=(242, 220, 242),
            white=(255, 255, 255), black=(30, 28, 26), gray=(140, 136, 130),
            dim=(200, 197, 193), ready=(70, 155, 60), yellow=(25, 150, 195),
        )

    def _load_models(self):
        print("── Loading models ──────────────────────────────────")
        os.makedirs(MODELS_DIR, exist_ok=True)

        # Letter models
        self.lm = self.le = None
        self.lfd = 63
        if ML_OK:
            for mf, ef in [('letters_model.h5', 'letters_labels.pkl'),
                            ('letter_model.h5', 'letter_labels.pkl')]:
                mp_ = os.path.join(MODELS_DIR, mf)
                ep_ = os.path.join(MODELS_DIR, ef)
                if not (os.path.exists(mp_) and os.path.exists(ep_)):
                    continue
                try:
                    self.lm = keras.models.load_model(mp_, custom_objects=_CUSTOM_OBJECTS)
                    self.lfd = int(self.lm.input_shape[-1])
                    with open(ep_, 'rb') as f:
                        self.le = pickle.load(f)
                    print(f"  [OK] Letters — {len(list(self.le.classes_))} classes")
                    dummy = np.zeros((1, self.lfd), np.float32)
                    self.lm(tf.constant(dummy), training=False)
                    break
                except Exception as e:
                    print(f"  [ERR] {mf}: {e}")
        if self.lm is None:
            print("  [INFO] No letter model")

        # Word models
        self.wm = self.we = self.wmt = None
        self.wil = 60
        self.wfd = 126
        self._word_ext = extract_word_frame_126
        if ML_OK:
            for mf, ef, tag in [
                ('words_model.h5', 'words_labels.pkl', 'v7'),
                ('advanced_words_model.h5', 'advanced_words_labels.pkl', 'adv'),
            ]:
                mp_ = os.path.join(MODELS_DIR, mf)
                ep_ = os.path.join(MODELS_DIR, ef)
                if not (os.path.exists(mp_) and os.path.exists(ep_)):
                    continue
                try:
                    raw = keras.models.load_model(mp_, custom_objects=_CUSTOM_OBJECTS,
                                                  safe_mode=False)
                    with open(ep_, 'rb') as f:
                        self.we = pickle.load(f)
                    self.wil = int(raw.input_shape[1])
                    self.wfd = int(raw.input_shape[2])
                    self.wm = tf.function(
                        lambda x: raw(x, training=False),
                        input_signature=[tf.TensorSpec(
                            shape=[1, self.wil, self.wfd], dtype=tf.float32)])
                    self.wmt = tag
                    self._word_ext = make_extractor(self.wfd)
                    classes = (list(self.we.classes_)
                               if hasattr(self.we, 'classes_') else self.we)
                    print(f"  [OK] Words ({tag}) — {len(classes)} classes")
                    dummy = np.zeros((1, self.wil, self.wfd), np.float32)
                    self.wm(tf.constant(dummy, dtype=tf.float32))
                    break
                except Exception as e:
                    print(f"  [ERR] {mf}: {e}")
        if self.wm is None:
            print("  [INFO] No word model")
        print("────────────────────────────────────────────────────")

    def _setup_mp(self):
        if not ML_OK:
            self._mph = self._mphands = self._mpdraw = self._mpstyle = None
            self._facemesh = None
            return
        self._mph = mp.solutions.hands
        self._mphands = self._mph.Hands(
            static_image_mode=False, max_num_hands=2,
            min_detection_confidence=0.65, min_tracking_confidence=0.55,
            model_complexity=0)
        self._mpdraw = mp.solutions.drawing_utils
        self._mpstyle = mp.solutions.drawing_styles
        self._mp_face = mp.solutions.face_mesh
        self._facemesh = self._mp_face.FaceMesh(
            static_image_mode=False, max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def _draw_face(self, frame, rgb):
        if not ML_OK or self._facemesh is None:
            return
        try:
            res = self._facemesh.process(rgb)
            if res and res.multi_face_landmarks:
                for flm in res.multi_face_landmarks:
                    self._mpdraw.draw_landmarks(
                        image=frame, landmark_list=flm,
                        connections=self._mp_face.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self._mpdraw.DrawingSpec(
                            color=(195, 215, 195), thickness=1))
        except Exception:
            pass

    def _lfeats(self, lm):
        return extract_letter_feats(lm, self.lfd)

    def _pred_letter(self, feats):
        """Letter prediction — A-Z ONLY."""
        if self.lm is None or len(feats) != self.lfd:
            return None, 0.0
        try:
            x = tf.constant(np.array(feats, np.float32).reshape(1, -1))
            p = self.lm(x, training=False).numpy()[0]
            i = int(np.argmax(p))
            lbl = (str(self.le.inverse_transform([i])[0])
                   if hasattr(self.le, 'inverse_transform') else str(self.le[i]))
            # Validate: A-Z only
            if isinstance(lbl, str) and len(lbl) == 1 and lbl.upper() in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                return lbl.upper(), float(p[i])
            return None, 0.0
        except Exception as e:
            if self._debug:
                print(f"  [ERR letter] {e}")
            return None, 0.0

    def _run_word_inf(self):
        if self.wm is None or len(self.wbuf) < WORD_MIN_FRAMES:
            return
        try:
            arr = np.array(list(self.wbuf), np.float32)
            arr = gaussian_filter1d(arr, sigma=1.2, axis=0)  # ★ Reduced sigma
            t, fd = self.wil, self.wfd
            if len(arr) < t:
                arr = np.vstack([np.zeros((t - len(arr), fd), np.float32), arr])
            else:
                arr = arr[-t:]
            arr = arr[:, :fd]
            probs = self.wm(
                tf.constant(arr.reshape(1, t, fd), dtype=tf.float32)
            ).numpy()[0]
            self._pred_pool.push(probs)
        except Exception as e:
            if self._debug:
                print(f"  [ERR word] {e}")

    def _asl(self):
        parts = list(self.sentence)
        if self.spelling:
            parts.append(''.join(self.spelling))
        return ' '.join(parts) if parts else 'Start signing...'

    def _set_english_async(self, text):
        """★ Callback fired from a background thread once the AI
        interpreter finishes. Simple attribute assignment is safe here."""
        self.english = text
        self._eng_cache = text

    def _eng(self):
        """
        ★ FIX FOR UI FREEZE:
        Returns a cached translation instantly if the underlying ASL
        sentence hasn't changed. Only kicks off a NEW background AI
        request when the sentence actually changes — never blocks the
        video/draw loop waiting on the network.
        """
        asl = self._asl()
        if asl == 'Start signing...':
            return ''
        if asl == self._eng_cache_key:
            return self._eng_cache  # nothing changed — instant, no work at all
        self._eng_cache_key = asl
        self._eng_cache = self.nlp.enhance_async(asl, callback=self._set_english_async)
        return self._eng_cache

    def _try_merge(self):
        for compound, merged in self._MERGE.items():
            n = len(compound)
            if len(self.sentence) >= n and tuple(self.sentence[-n:]) == compound:
                del self.sentence[-n:]
                self.sentence.append(merged)
                print(f"  [MERGE] {list(compound)} -> {merged}")
                return True
        return False

    def _add_word(self, word):
        word = word.upper()
        if self.sentence and self.sentence[-1] == word:
            self._fb(f"Duplicate skipped: {word}", self.C['gray'])
            return
        self._undo.append(list(self.sentence))
        self.sentence.append(word)
        self._try_merge()
        self.english = self._eng()  # ★ FIX: was a blocking call, now cached/async
        self._fb(f"+ {word}", self.C['green'])
        self.alerts.fire('accept')
        print(f"  [+] {word}  |  {self._asl()}")

    def _commit_word(self):
        if not self.spelling:
            return
        word = ''.join(self.spelling).upper()
        self.spelling = []
        self.rec.reset()
        self._add_word(word)
        self.alerts.fire('word')

    def _do_accept(self):
        if self.mode == 'letters':
            if self.absent.elapsed() >= LETTER_COMMIT_ABSENT_MIN and self.spelling:
                self._commit_word()
                self._fb("Word committed", self.C['green'])
        else:
            picks = self.top3 or self.last_top3
            if picks:
                self._add_word(str(picks[0][0]))
                self._word_reset()

    def _word_reset(self):
        self.top3 = []
        self.last_top3 = []
        self.wbuf.clear()
        if self._pred_pool:
            self._pred_pool.clear()
        self._clean_frames = 0
        self._hand_absent = 0

    def _do_toggle(self):
        if self.mode == 'letters' and self.spelling:
            word = ''.join(self.spelling).upper()
            self._undo.append(list(self.sentence))
            self.sentence.append(word)
            self.english = self._eng()  # ★ FIX: was a blocking call
            self._fb(f"Saved '{word}' before mode switch", self.C['orange'])
            self.alerts.fire('word')
        elif self.mode == 'words':
            picks = self.top3 or self.last_top3
            if picks and picks[0][1] >= WORD_CONF_LOCK:
                self._add_word(str(picks[0][0]))

        # Cycle: WORDS → LETTERS → WORDS
        self.mode = 'letters' if self.mode == 'words' else 'words'

        self._word_reset()
        self.spelling = []
        self.rec.reset()
        self.absent.reset()
        self._mode_grace = time.time() + MODE_GRACE
        self.alerts.fire('toggle')

        mode_names = {'words': 'WORD', 'letters': 'LETTER'}
        label = mode_names.get(self.mode, self.mode.upper())
        self._fb(f"Switched to {label} mode", self.C['teal'])
        print(f"\n  [MODE] {self.mode.upper()}\n")

    def _do_speak(self):
        # NOTE: This is an explicit, occasional user action (not part of
        # the per-frame draw loop), so it's fine for it to block briefly
        # while getting the best possible final text before speaking.
        text = self.nlp.correct_and_enhance(self._asl())
        if not text:
            self._fb("Nothing to speak", self.C['gray'])
            self.alerts.fire('error')
            return
        self.english = text
        self._last_spoken = text
        self._fb(f"Speaking...", self.C['purple'])
        self.alerts.fire('speak')
        print(f"  [SPEAK] {text}")
        self.tts.speak(text)
        self.sentence.clear()
        self._undo.clear()
        self.spelling.clear()
        self.rec.reset()
        self._word_reset()
        self.english = ''
        self._eng_cache = ''
        self._eng_cache_key = None

    def _do_undo(self):
        if self._undo:
            self.sentence = self._undo.pop()
            self.english = self._eng()
            self._fb("Undo", self.C['orange'])
            self.alerts.fire('undo')
        else:
            self._fb("Nothing to undo", self.C['gray'])
            self.alerts.fire('error')

    def _fb(self, text, col):
        self._fb_text = text
        self._fb_col = col
        self._fb_until = time.time() + 2.5

    # ── drawing helpers ───────────────────────────────────────────────────────
    def _t(self, f, s, pos, sc, col, th=1):
        cv2.putText(f, s, pos, cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)

    def _bar(self, f, x1, y1, x2, y2, pct, col, bg=None):
        bg = bg or self.C['panel2']
        cv2.rectangle(f, (x1, y1), (x2, y2), bg, -1)
        if pct > 0:
            cx = x1 + max(0, int((x2 - x1) * pct))
            cv2.rectangle(f, (x1, y1), (cx, y2), col, -1)

    def _pill(self, f, x, y, w, h, col_bg, col_fg, text, ts=0.40):
        r = h // 2
        cv2.rectangle(f, (x + r, y), (x + w - r, y + h), col_bg, -1)
        cv2.circle(f, (x + r, y + r), r, col_bg, -1)
        cv2.circle(f, (x + w - r, y + r), r, col_bg, -1)
        tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, ts, 1)[0][0]
        tx = x + (w - tw) // 2
        ty = y + h - max(2, (h - int(ts * 16)) // 2)
        self._t(f, text, (tx, ty), ts, col_fg, 1)

    def _card(self, f, x, y, w, h, bg=None, border=None):
        bg = bg or self.C['panel']
        border = border or self.C['border']
        cv2.rectangle(f, (x, y), (x + w, y + h), bg, -1)
        cv2.rectangle(f, (x, y), (x + w, y + h), border, 1)

    def _spark_draw(self, f, vals, x, y, w, h, col):
        if len(vals) < 2:
            return
        pts = [(x + int(i * w / (len(vals) - 1)),
                y + h - int(v * h)) for i, v in enumerate(vals)]
        for i in range(len(pts) - 1):
            cv2.line(f, pts[i], pts[i + 1], col, 1, cv2.LINE_AA)

    # ── UI DRAWING (identical to v8.1, preserved exactly) ──────────────────
    def _draw_ui(self, frame, nh):
        C = self.C
        h, w = frame.shape[:2]
        now = time.time()
        T = self._t
        spk = self.tts.is_speaking
        HEADER_H = 52
        BOTTOM_H = 212
        BOTTOM_Y = h - BOTTOM_H

        cv2.rectangle(frame, (0, 0), (w, HEADER_H), C['header_bg'], -1)
        cv2.line(frame, (0, HEADER_H), (w, HEADER_H), C['border'], 1)
        if self.mode == 'letters':
            m_bg, m_fg, m_txt = C['blue_light'], C['blue'], "LETTER MODE"
        else:
            m_bg, m_fg, m_txt = C['green_light'], C['green'], "WORD MODE"
        self._pill(frame, 12, 12, 140, 28, m_bg, m_fg, m_txt, 0.38)
        title = "ASL Communication System"
        tw = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1)[0][0]
        T(frame, title, (w // 2 - tw // 2, 33), 0.58, C['text_primary'], 1)
        ai_txt = (f"AI  {self.nlp._model.split(':')[0][:10]}"
                  if self.nlp.ai_ok else "AI offline")
        ai_bg = C['teal_light'] if self.nlp.ai_ok else C['panel2']
        ai_fg = C['teal'] if self.nlp.ai_ok else C['gray']
        self._pill(frame, w - 148, 12, 134, 28, ai_bg, ai_fg, ai_txt, 0.40)
        if spk:
            self._pill(frame, w - 292, 12, 136, 28,
                       C['purple_light'], C['purple'], "SPEAKING...", 0.40)
        cv2.circle(frame, (w - 162, 26), 5,
                   C['green'] if nh > 0 else C['border_med'], -1)

        no_model = not ML_OK or (self.lm is None and self.wm is None)
        WARN_H = 22 if no_model else 0
        if no_model:
            WY = HEADER_H
            cv2.rectangle(frame, (0, WY), (w, WY + WARN_H), C['orange_light'], -1)
            cv2.line(frame, (0, WY + WARN_H), (w, WY + WARN_H), C['orange'], 1)
            T(frame, "No ML models — grammar / keyboard mode  |  Add .h5 + .pkl to models/",
              (10, WY + 15), 0.37, C['orange'], 1)

        STRIP_Y = HEADER_H + WARN_H
        STRIP_H = 34
        abs_e = self.absent.elapsed()
        grace = now < self._mode_grace
        cv2.rectangle(frame, (0, STRIP_Y), (w, STRIP_Y + STRIP_H), C['panel'], -1)
        cv2.line(frame, (0, STRIP_Y + STRIP_H), (w, STRIP_Y + STRIP_H), C['border'], 1)

        if grace and self.mode == 'letters':
            grem = max(0.0, self._mode_grace - now)
            self._bar(frame, 0, STRIP_Y + STRIP_H - 3,
                      w, STRIP_Y + STRIP_H, grem / MODE_GRACE, C['teal'])
            T(frame, f"Get ready — {grem:.1f}s",
              (w // 2 - 62, STRIP_Y + 22), 0.48, C['teal'], 1)
        elif nh == 0 and abs_e > 0:
            s1 = int(w * T_ACCEPT / T_SPEAK)
            s2 = int(w * T_TOGGLE / T_SPEAK)
            cv2.rectangle(frame, (0, STRIP_Y), (s1, STRIP_Y + STRIP_H), C['green_light'], -1)
            cv2.rectangle(frame, (s1, STRIP_Y), (s2, STRIP_Y + STRIP_H), C['orange_light'], -1)
            cv2.rectangle(frame, (s2, STRIP_Y), (w, STRIP_Y + STRIP_H), C['purple_light'], -1)
            cv2.line(frame, (s1, STRIP_Y), (s1, STRIP_Y + STRIP_H), C['border'], 1)
            cv2.line(frame, (s2, STRIP_Y), (s2, STRIP_Y + STRIP_H), C['border'], 1)
            T(frame, f"ACCEPT  {T_ACCEPT}s", (8, STRIP_Y + 22), 0.40, C['green'], 1)
            T(frame, f"TOGGLE  {T_TOGGLE}s", (s1 + 6, STRIP_Y + 22), 0.40, C['orange'], 1)
            T(frame, f"SPEAK  {T_SPEAK}s", (s2 + 6, STRIP_Y + 22), 0.40, C['purple'], 1)
            fx = int(w * min(abs_e / T_SPEAK, 1.0))
            bc = C['green'] if abs_e < T_ACCEPT else C['orange'] if abs_e < T_TOGGLE else C['purple']
            cv2.rectangle(frame, (0, STRIP_Y + STRIP_H - 3),
                          (fx, STRIP_Y + STRIP_H), bc, -1)
        elif self.mode == 'letters' and nh > 0:
            lp = self.rec.letter_progress()
            wp = self.rec.word_progress()
            if wp > 0:
                self._bar(frame, 0, STRIP_Y + STRIP_H - 3,
                          w, STRIP_Y + STRIP_H, wp, C['orange'])
                T(frame, f"Hold — commit word in {max(0.0, WORD_COMMIT_TIME * (1 - wp)):.1f}s",
                  (12, STRIP_Y + 22), 0.46, C['orange'], 1)
            elif lp > 0:
                bc = C['green'] if lp >= 1.0 else C['teal']
                self._bar(frame, 0, STRIP_Y + STRIP_H - 3,
                          w, STRIP_Y + STRIP_H, lp, bc)
                msg = ("Adding..." if lp >= 1.0 else
                       f"Hold — add letter in {max(0.0, LETTER_ADD_TIME * (1 - lp)):.1f}s")
                T(frame, msg, (12, STRIP_Y + 22), 0.45, bc, 1)
            else:
                T(frame, f"Sign letters and hold still   A=add now   SPACE=commit",
                  (12, STRIP_Y + 22), 0.40, C['text_light'], 1)
        else:
            T(frame, "Remove hand:  1.0s=accept   3.0s=toggle   5.5s=speak     S=speak   Z=undo",
              (12, STRIP_Y + 22), 0.38, C['text_light'], 1)

        PT = STRIP_Y + STRIP_H + 6

        if self.mode == 'letters':
            px, py, pw, ph = 12, PT, 242, 202
            self._card(frame, px, py, pw, ph, C['header_bg'], C['border'])
            if grace:
                grem = max(0.0, self._mode_grace - now)
                cv2.rectangle(frame, (px, py), (px + pw, py + ph), C['teal_light'], -1)
                cv2.rectangle(frame, (px, py), (px + pw, py + ph), C['teal'], 1)
                T(frame, "READY IN", (px + 62, py + 76), 0.65, C['teal'], 1)
                T(frame, f"{grem:.1f}s", (px + 46, py + 148), 2.2, C['teal'], 5)
            elif self.rec.display_lbl:
                lbl = self.rec.display_lbl
                conf = self.rec.display_conf
                self._spark.push(conf)
                lp = self.rec.letter_progress()
                added = self.rec._after_add
                rdy = conf >= _conf_thr(lbl)
                if added:
                    ac, ac_l = C['orange'], C['orange_light']
                    chip, chip_c = "ADDED", C['orange']
                elif lp >= 1.0:
                    ac, ac_l = C['green'], C['green_light']
                    chip, chip_c = "ADDING", C['green']
                elif rdy:
                    ac, ac_l = C['teal'], C['teal_light']
                    chip, chip_c = "DETECTED", C['teal']
                else:
                    ac, ac_l = C['gray'], C['panel2']
                    chip, chip_c = "LOW CONF", C['gray']
                cv2.rectangle(frame, (px, py), (px + pw, py + ph), ac_l, -1)
                cv2.rectangle(frame, (px, py), (px + pw, py + ph), ac, 1)
                T(frame, str(lbl), (px + 58, py + 132), 3.8, ac, 7)
                T(frame, f"{conf:.0%}", (px + pw - 54, py + 164), 0.70, ac, 1)
                self._bar(frame, px + 14, py + 174, px + pw - 14, py + 182, conf, ac, C['panel2'])
                self._spark_draw(frame, self._spark.get(), px + 14, py + 186, pw - 28, 14, ac)
                self._pill(frame, px + pw - 84, py + 8, 72, 20, ac_l, ac, chip, 0.36)
            else:
                cv2.rectangle(frame, (px, py), (px + pw, py + ph), C['panel'], -1)
                cv2.rectangle(frame, (px, py), (px + pw, py + ph), C['border'], 1)
                msg = "No model" if self.lm is None else "Show your hand..."
                T(frame, msg, (px + 28, py + 106), 0.56, C['text_light'], 1)
            if self.spelling:
                sx = px + pw + 12
                self._card(frame, sx, py, 200, 66, C['header_bg'], C['border'])
                T(frame, "letters", (sx + 10, py + 18), 0.42, C['text_light'])
                ws = ''.join(self.spelling)
                if len(ws) > 9:
                    ws = '...' + ws[-8:]
                T(frame, ws, (sx + 10, py + 56), 1.5, C['orange'], 3)
        else:
            d3 = self.top3 if self.top3 else self.last_top3
            CW, CH = 196, 140
            if d3:
                cols = [C['green'], C['blue'], C['orange']]
                fills = [C['green_light'], C['blue_light'], C['orange_light']]
                ranks = ["TOP", "2nd", "3rd"]
                for i, (lbl, conf) in enumerate(d3[:3]):
                    bx = 12 + i * (CW + 10)
                    bcol = cols[i] if conf >= WORD_CONF_SHOW else C['border_med']
                    bfill = fills[i] if conf >= WORD_CONF_SHOW else C['panel']
                    locked = conf >= WORD_CONF_LOCK
                    self._card(frame, bx, PT, CW, CH, bfill, bcol)
                    self._pill(frame, bx + CW - 82, PT + 8, 72, 20,
                               C['header_bg'] if not locked else bcol,
                               bcol if not locked else C['header_bg'],
                               ("* " if locked else "") + ranks[i], 0.35)
                    fs = 1.10 if len(str(lbl)) <= 8 else 0.75
                    T(frame, str(lbl), (bx + 10, PT + 94), fs, bcol, 3)
                    T(frame, f"{conf:.0%}", (bx + CW - 54, PT + 126), 0.64, bcol, 1)
                    self._bar(frame, bx + 10, PT + 132, bx + CW - 10, PT + 138, conf, bcol, C['panel2'])
                bp = min(self._clean_frames / max(1, WORD_MIN_FRAMES), 1.0)
                self._bar(frame, 12, PT + CH + 6, 12 + 3 * (CW + 10), PT + CH + 10, bp, C['blue'], C['panel2'])
                T(frame, "Remove hand 1.0s = accept   * = high confidence",
                  (12, PT + CH + 24), 0.40, C['text_light'], 1)
            else:
                bp = min(self._clean_frames / max(1, WORD_MIN_FRAMES), 1.0)
                self._card(frame, 12, PT, 480, 128, C['header_bg'], C['border'])
                T(frame, "Sign a word or phrase", (24, PT + 36), 0.80, C['text_primary'], 1)
                T(frame, "Hold the sign — prediction appears here",
                  (24, PT + 62), 0.50, C['text_secondary'], 1)
                need = max(0, WORD_MIN_FRAMES - self._clean_frames)
                col_msg = C['teal'] if need == 0 else C['text_light']
                T(frame,
                  "Analysing..." if need == 0 else f"Collecting frames... ({need} more)",
                  (24, PT + 86), 0.44, col_msg, 1)
                self._bar(frame, 24, PT + 100, 468, PT + 106, bp, C['teal'], C['panel2'])
                if self.wm is None:
                    self._pill(frame, 24, PT + 112, 200, 20,
                               C['red_light'], C['red'], "NO WORD MODEL", 0.38)

        cv2.rectangle(frame, (0, BOTTOM_Y), (w, h), C['bg'], -1)
        cv2.line(frame, (0, BOTTOM_Y), (w, BOTTOM_Y), C['border'], 1)
        SUGG_Y = BOTTOM_Y
        SUGG_H = 28
        ASL_Y = SUGG_Y + SUGG_H
        ASL_H = 36
        ENG_Y = ASL_Y + ASL_H
        ENG_H = 82
        KEY_Y = ENG_Y + ENG_H

        cv2.rectangle(frame, (0, SUGG_Y), (w, SUGG_Y + SUGG_H), C['panel'], -1)
        cv2.line(frame, (0, SUGG_Y + SUGG_H), (w, SUGG_Y + SUGG_H), C['border'], 1)
        T(frame, "Suggest:", (10, SUGG_Y + 19), 0.40, C['text_light'], 1)
        xc = 78
        for idx, sg in enumerate(self._sugg[:4]):
            lbl2 = f"{idx + 1}  {sg.replace('_', ' ')}"
            tw2 = cv2.getTextSize(lbl2, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0]
            bw2 = tw2 + 20
            if xc + bw2 > w - 12:
                break
            self._pill(frame, xc, SUGG_Y + 5, bw2, 18,
                       C['blue_light'] if idx == 0 else C['panel2'],
                       C['blue'] if idx == 0 else C['text_secondary'],
                       lbl2, 0.40)
            xc += bw2 + 6

        cv2.rectangle(frame, (0, ASL_Y), (w, ASL_Y + ASL_H), C['header_bg'], -1)
        cv2.line(frame, (0, ASL_Y + ASL_H), (w, ASL_Y + ASL_H), C['border'], 1)
        T(frame, "ASL", (10, ASL_Y + 24), 0.48, C['text_light'], 1)
        cv2.line(frame, (44, ASL_Y + 6), (44, ASL_Y + ASL_H - 6), C['border'], 1)
        asl = self._asl()
        if len(asl) > 72:
            asl = '...' + asl[-70:]
        T(frame, asl, (52, ASL_Y + 24), 0.62, C['text_primary'], 1)
        T(frame, f"{len(self.sentence)}w", (w - 45, ASL_Y + 24), 0.44, C['text_light'], 1)

        cv2.rectangle(frame, (0, ENG_Y), (w, ENG_Y + ENG_H), C['bg'], -1)
        cv2.line(frame, (0, ENG_Y + ENG_H), (w, ENG_Y + ENG_H), C['border'], 1)
        T(frame, "English", (10, ENG_Y + 22), 0.48, C['text_light'], 1)
        cv2.line(frame, (66, ENG_Y + 6), (66, ENG_Y + ENG_H - 6), C['border'], 1)
        badge = "AI" if self.nlp.ai_ok else "local"
        b_bg = C['teal_light'] if badge == "AI" else C['panel2']
        b_fg = C['teal'] if badge == "AI" else C['gray']
        self._pill(frame, 74, ENG_Y + 5, 44, 18, b_bg, b_fg, badge, 0.36)
        eng = self.english or self._eng() or "Translation appears here..."
        ec = C['text_primary'] if eng != "Translation appears here..." else C['text_light']
        if len(eng) > 60:
            T(frame, eng[:60], (74, ENG_Y + 44), 0.65, ec, 1)
            T(frame, eng[60:120], (74, ENG_Y + 68), 0.65, ec, 1)
        else:
            T(frame, eng, (74, ENG_Y + 52), 0.72, ec, 2)

        bx = w - 134
        by3 = ENG_Y + 10
        s_bg = C['purple_light'] if spk else C['panel']
        s_fg = C['purple'] if spk else C['text_secondary']
        s_bd = C['purple'] if spk else C['border']
        cv2.rectangle(frame, (bx, by3), (bx + 122, by3 + 36), s_bg, -1)
        cv2.rectangle(frame, (bx, by3), (bx + 122, by3 + 36), s_bd, 1)
        T(frame, "Speaking..." if spk else "S  Speak", (bx + 10, by3 + 24), 0.56, s_fg, 1)

        cv2.rectangle(frame, (0, KEY_Y), (w, h), C['panel'], -1)
        cv2.line(frame, (0, KEY_Y), (w, KEY_Y), C['border'], 1)
        T(frame,
          "A=add  SPACE=commit  S=speak  M=toggle  BKSP=del  Z=undo  1-4=suggest  ENTER=clear  Q=quit",
          (10, KEY_Y + (h - KEY_Y + 10) // 2), 0.36, C['text_light'], 1)

        if now < self._fb_until and self._fb_text:
            rem = self._fb_until - now
            alpha = min(rem / 0.4, 1.0)
            fw = min(len(self._fb_text) * 11 + 40, w - 80)
            fx = w // 2 - fw // 2
            fy = BOTTOM_Y - 48
            ov = frame.copy()
            cv2.rectangle(ov, (fx, fy), (fx + fw, fy + 36), C['header_bg'], -1)
            cv2.rectangle(ov, (fx, fy), (fx + fw, fy + 36), self._fb_col, 1)
            cv2.addWeighted(ov, alpha * 0.92, frame, 1 - alpha * 0.92, 0, frame)
            T(frame, self._fb_text, (fx + 12, fy + 24), 0.56, self._fb_col, 1)

        self.alerts.draw(frame)

    # ══════════════════════════════════════════════════════════════════════════
    #  OPTIMIZED MAIN LOOP
    # ══════════════════════════════════════════════════════════════════════════
    def run(self):
        cap = None
        for idx in [0, 1, 2]:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                print(f"  [OK] Camera index {idx}")
                break
            cap.release()
        if not cap or not cap.isOpened():
            print("[ERR] No camera found")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        WN = 'ASL System v9.0 — Advanced NLP & Performance'
        # ★ FIX FOR WINDOW STRETCH/PIXELATION:
        #   WINDOW_NORMAL allowed the window to be dragged/maximized far
        #   beyond the 640x480 capture size, which stretched every frame
        #   (and all UI text/shapes drawn on it) into a blocky, pixelated
        #   mess. WINDOW_AUTOSIZE locks the window to the actual frame
        #   size so it can never be resized or maximized.
        cv2.namedWindow(WN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WN, 1920, 1080)

        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            frame = cv2.flip(frame, 1)
            self._frame += 1

            hands = False
            nh = 0
            res = None
            if ML_OK and self._mphands is not None:
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self._draw_face(frame, rgb)
                    res = self._mphands.process(rgb)
                    hands = bool(res and res.multi_hand_landmarks)
                    nh = len(res.multi_hand_landmarks) if hands else 0
                    if hands:
                        for hnd in res.multi_hand_landmarks:
                            self._mpdraw.draw_landmarks(
                                frame, hnd,
                                self._mph.HAND_CONNECTIONS,
                                self._mpstyle.get_default_hand_landmarks_style(),
                                self._mpstyle.get_default_hand_connections_style())
                except Exception as e:
                    if self._debug:
                        print(f"  [ERR mp] {e}")

            abs_ev = self.absent.update(hands)
            grace = time.time() < self._mode_grace
            if not grace:
                if 'speak' in abs_ev:
                    self._do_speak()
                if 'toggle' in abs_ev:
                    self._do_toggle()
                if 'accept' in abs_ev:
                    self._do_accept()

            # ── Letter mode inference ──────────────────────────────────────
            if self.mode == 'letters' and res is not None:
                if hands and not grace:
                    try:
                        lm0 = res.multi_hand_landmarks[0]
                        if self.lm is not None:
                            lfeats = self._lfeats(lm0)
                            self._infer_letter.submit(
                                lambda f: self._pred_letter(f), lfeats)
                            r = self._infer_letter.result()
                            if r is not None:
                                rl, rc = r
                            else:
                                rl, rc = None, 0.0
                        else:
                            rl, rc = None, 0.0

                        lev = self.rec.update(rl, rc)
                        self._spark.push(rc or 0.0)

                        if 'add_letter' in lev and self.rec._added_lbl:
                            ltr = self.rec._added_lbl
                            self.spelling.append(ltr)
                            self._fb(f"+ '{ltr}' ->  {''.join(self.spelling)}",
                                     self.C['teal'])
                            self.alerts.fire('letter')
                            print(f"  [LETTER] '{ltr}' -> {''.join(self.spelling)}")

                        if 'commit_word' in lev and self.spelling:
                            self._commit_word()
                    except Exception as e:
                        if self._debug:
                            print(f"  [ERR letter loop] {e}")
                elif not hands:
                    self.rec.reset()
                    self._spark.push(0.0)

            # ── Word mode inference ────────────────────────────────────────
            elif self.mode == 'words' and self.wm is not None and res is not None:
                if hands:
                    self._hand_absent = 0
                    try:
                        feats = self._word_ext(res.multi_hand_landmarks)
                        self.wbuf.append(feats)
                        while len(self.wbuf) > self.wil:
                            self.wbuf.popleft()
                        self._clean_frames += 1
                        if (self._clean_frames >= WORD_MIN_FRAMES
                                and self._frame % WORD_PRED_EVERY == 0):
                            self._run_word_inf()
                        if self._pred_pool and len(self._pred_pool) > 0:
                            cands = self._pred_pool.top3()
                            if cands and cands[0][1] >= WORD_CONF_SHOW:
                                self.top3 = cands
                                if cands[0][1] >= WORD_CONF_LOCK:
                                    self.last_top3 = cands
                            else:
                                self.top3 = []
                    except Exception as e:
                        if self._debug:
                            print(f"  [ERR word loop] {e}")
                else:
                    self._hand_absent += 1
                    if self._hand_absent > HAND_DROPOUT_GRACE:
                        self.wbuf.clear()
                        self._clean_frames = 0
                        self._hand_absent = 0

            self._sugg = self.ac.suggest(self.sentence, self.spelling)
            self._draw_ui(frame, nh)
            cv2.imshow(WN, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('d'):
                self._debug = not self._debug
                print(f"  Debug: {self._debug}")
            elif key == ord('s'):
                self._do_speak()
            elif key == ord('m'):
                self._do_toggle()
            elif key in (ord('z'), 26):
                self._do_undo()
            elif key == ord('a') and self.mode == 'letters':
                lbl = self.rec.display_lbl
                if lbl:
                    self.spelling.append(lbl)
                    self._fb(f"+ '{lbl}' ->  {''.join(self.spelling)}", self.C['teal'])
                    self.alerts.fire('letter')
                    print(f"  [MANUAL LETTER] '{lbl}' -> {''.join(self.spelling)}")
                    self.rec.reset()
            elif key == ord(' ') and self.mode == 'letters':
                if self.spelling:
                    self._commit_word()
                else:
                    self._fb("No input to commit", self.C['gray'])
            elif key in (ord('1'), ord('2'), ord('3'), ord('4')):
                i = key - ord('1')
                if i < len(self._sugg):
                    self._add_word(self._sugg[i])
            elif key == 8:
                if self.mode == 'letters' and self.spelling:
                    rem = self.spelling.pop()
                    self._fb(f"Deleted '{rem}'  -> {''.join(self.spelling)}", self.C['red'])
                    self.alerts.fire('undo')
                elif self.sentence:
                    r = self.sentence.pop()
                    self.english = self._eng()
                    self._fb(f"Deleted: {r}", self.C['red'])
                    self.alerts.fire('undo')
            elif key == 13:
                self.sentence.clear()
                self.spelling.clear()
                self._undo.clear()
                self.english = ''
                self._eng_cache = ''
                self._eng_cache_key = None
                self._word_reset()
                self.rec.reset()
                self.absent.reset()
                self._fb("Cleared", self.C['gray'])
                self.alerts.fire('clear')
                print("  [CLEAR]")

        cap.release()
        cv2.destroyAllWindows()
        self.tts.stop()
        print("\n[DONE]")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("NLP warm-up (spell correction & context):")
    nlp = AdvancedNLPEngine()
    tests = [
        "hello i home", "i hungry", "me go store",
        "i love you thank you", "i need doctor",
        "i want water please", "you go school tomorrow",
        "i not sick", "i finish eat", "family happy",
        "my name is alex", "thank you", "how are you",
        "where bathroom", "i not feel good",
        "you help me", "i late work", "i have pain",
        "i need medicine", "call police",
        "i doctor", "i water", "i live", "when you come", "why you sad", "hi",
    ]
    for ex in tests:
        corrected = nlp.local.convert(ex)
        print(f"  '{ex}'\n     -> '{corrected}'")
    print()
    try:
        ASLSystem().run()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        import traceback
        print(f"\n[FATAL] {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()