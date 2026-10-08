# AI-Powered Adaptive Learning Dashboard

An original learning-resource dashboard demonstrating **runtime UI adaptation** and a lightweight **AI-related NLP recommendation algorithm**. Created for the University of the Cumberlands AI-Based Adaptive Human-Computer Interaction assignment.

## Requirements

- Python 3.10 or later
- A terminal / VS Code
- Internet access for installing the Python packages and opening the example learning-resource websites (no internet needed for recommendation calculations after installation)

## Windows quick start (VS Code CMD terminal)

1. Extract the ZIP, then open the extracted `adaptive_learning_dashboard` folder in VS Code.
2. Open **Terminal → New Terminal**.
3. Run:

```bat
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

4. Open the `Local URL` shown in the terminal, normally `http://localhost:8501`.

On macOS or Linux, activate using `source .venv/bin/activate`.

## Manual demonstration / screenshots

1. **Baseline:** Capture the default dashboard with User Experience selected.
2. **Interest adaptation:** Switch to Artificial Intelligence. Notice the changed recommendation order. Capture the new order.
3. **Learning feedback:** Click `👍 Show me more like this` on any recommendation; notice the adjusted score and logged event. Capture the changed score and feedback count.
4. **Complexity adaptation:** Change **Beginner** to **Advanced**; screenshot the change in descriptions.
5. **Display density:** Set the count to 2 and then to 8; screenshot the different number of items.
6. **Accessibility presentation:** Enable high contrast; screenshot the appearance change.
7. **Reset:** Press **Reset my interaction feedback**; check that interaction history resets.

**Expected result:** changes take effect during the current running session, without changing or reinstalling the code.

## Automated unit tests

From the project folder, after installation:

```bat
python -m unittest discover -s tests -v
```

## How the adaptation works

- The user's chosen interest labels form a temporary text profile.
- `TfidfVectorizer` creates word-weighted vectors from the interest labels and resource descriptions.
- `cosine_similarity` ranks resource relevance.
- Positive/negative feedback adds/subtracts a small transparent amount from the similarity score.
- Streamlit `session_state` remembers feedback for the active session, enabling runtime response to interactions.
- Display changes (text complexity, number of cards, contrast mode) are direct user-controlled adaptations. These controls are **not themselves machine learning**.
- This prototype uses TF-IDF and a feedback-aware content-ranking heuristic, **not an LLM**, neural model, or model trained on personal data.

## Structure

- `app.py`: interactive Streamlit UI and per-session controls.
- `recommender.py`: resource catalog and AI/NLP ranking logic.
- `tests/test_recommender.py`: unit tests.
- `requirements.txt`: Python dependencies.
- `README.md`: installation, testing, and demo instructions.
- `.gitignore`: excludes local environments and caches from GitHub.

## Repository upload

1. Sign in to GitHub and create a repository named `adaptive-learning-dashboard` (public, or private with instructor access).
2. Upload `app.py`, `recommender.py`, `requirements.txt`, `README.md`, `.gitignore`, and the `tests/` folder, or use Git from VS Code.
3. Copy the **actual repository URL** from GitHub into the final report. A repository has **not** been created by this ZIP file.

## Attribution and AI disclosure

The app and tests were generated from scratch with assistance from **OpenAI ChatGPT (GPT-6)** on October 8, 2026, at the student's request. ChatGPT assisted with initial code, structure, explanations, and test cases. The student should run, evaluate, and document the actual results and any modifications before submission.

### Code and technology references

- Streamlit. (n.d.). *Streamlit documentation*. https://docs.streamlit.io/
- scikit-learn developers. (n.d.). *TfidfVectorizer*. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
- scikit-learn developers. (n.d.). *cosine_similarity*. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
- OpenAI. (2026). *ChatGPT* [Large language model]. https://chatgpt.com/

External learning links in `recommender.py` are example reference resources, not scraped content.
