# 🤟 Sign Language Interpretation

A real-time **Sign Language Interpretation System** that recognizes hand gestures through a webcam and converts them into meaningful text. The project uses **Computer Vision**, **Deep Learning**, and **Natural Language Processing (NLP)** to improve communication between sign language users and non-sign language users.

---

## 📌 Features

- 🔤 Real-time alphabet recognition
- 📝 Dynamic word recognition with hand movements
- 📷 Webcam-based gesture detection
- 🧠 AI-assisted sentence generation using Ollama
- ⚡ Fast and accurate predictions
- 🖥️ Easy-to-use interface
- 🤖 Supports local LLM integration for NLP

---

## 🛠️ Technologies Used

- Python
- OpenCV
- MediaPipe
- TensorFlow / Keras
- NumPy
- Matplotlib
- Ollama (Mistral / Qwen or other supported models)

---

## 📂 Project Structure

```
Sign-Language-Interpretation/
│── dataset/
│── models/
│── unified_collector.py
│── ultra_robust_trainer.py
│── advanced_word_collector.py
│── adbvance_word_trainer.py
│── system.py
│── requirements.txt
│── README.md
```

---

# 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Sign-Language-Interpretation.git
```

### 2. Navigate to the Project Folder

```bash
cd Sign-Language-Interpretation
```

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

---

# 📚 How to Train and Run

Follow the steps below in the given order.

## Step 1: Collect Alphabet Dataset

Run the following script to collect sign language alphabet images.

```bash
python unified_collector.py
```

---

## Step 2: Train the Alphabet Model

Train the collected alphabet dataset.

```bash
python ultra_robust_trainer.py
```

---

## Step 3: Collect Word Dataset

Collect dynamic word gestures (including hand movements).

```bash
python advanced_word_collector.py
```

---

## Step 4: Train the Word Recognition Model

Train the collected word dataset.

```bash
python adbvance_word_trainer.py
```

---

## Step 5: Run the Complete System

Launch the final Sign Language Interpretation system.

```bash
python system.py
```

---

# 🤖 AI Integration (Important)

For Natural Language Processing (NLP) and AI-assisted sentence generation, install **Ollama** and download at least one supported Large Language Model.

Recommended models:

- Mistral
- Qwen
- Llama (optional)

Example:

```bash
ollama pull mistral
```

or

```bash
ollama pull qwen
```

The application uses Ollama to improve sentence formation and contextual interpretation of recognized signs.

---

# ⚠️ Important Notes

- Follow the training steps in the exact order.
- Alphabet recognition should be trained before word recognition.
- Word recognition depends on movement-based gesture data.
- **Number recognition is currently under development and is not recommended for training or testing.**
- Ensure your webcam is connected before collecting datasets or running the system.

---

## 📸 Output

You can add screenshots of:

- Home Screen
- Alphabet Detection
- Word Detection
- AI Generated Sentence
- Final Prediction Window

Example output:

```
Detected Sign : HELLO

Generated Sentence :
Hello! How are you today?
```

---

## 📊 Model Performance

Update these values after training your models.

| Model | Accuracy |
|--------|----------|
| Alphabet Recognition | XX% |
| Word Recognition | XX% |

---

## 🎯 Future Improvements

- Number recognition
- Larger vocabulary support
- Continuous sentence recognition
- Speech synthesis (Text-to-Speech)
- Mobile application
- Cloud deployment
- Multi-language translation
- Improved gesture tracking

---

## 👨‍💻 Author

**Rajesh R S**

Bachelor of Computer Applications (BCA)

Skills:
- Python
- SQL
- Machine Learning
- Computer Vision
- Deep Learning

GitHub:
https://github.com/rajeshrs01/Sign-Language-Interpretation


---

## 📄 License

This project is developed for educational, research, and learning purposes.
