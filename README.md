# Sign Language AI — Streamlit

## Files
- `app.py` — Streamlit app with browser webcam, sign collection, model training, prediction and speech.
- `requirements.txt` — Python dependencies.
- `sign_samples.npz` and `sign_model.joblib` — generated when you collect samples and train the model. Keep backups of these files.

## Run locally
Use Python 3.11 if possible.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown by Streamlit and allow camera access.

## Deploy with GitHub + Streamlit Community Cloud
1. Create a new GitHub repository.
2. Upload `app.py`, `requirements.txt`, and this README.
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Choose **New app**, select your repository and branch, and set the main file to `app.py`.
5. Deploy. Open the HTTPS app URL and allow camera access.

## Important notes
- The app uses `streamlit-webrtc` because `cv2.imshow()` cannot display a webcam window inside a Streamlit web app.
- Browser camera access requires HTTPS on deployed hosting.
- WebRTC may need network access that permits STUN/TURN traffic. Some school/work networks block it.
- Streamlit Community Cloud's local filesystem is not permanent storage. Download/back up `sign_samples.npz` and `sign_model.joblib` after training, and upload them to the repository if you want the model available after a fresh deployment. Do not commit private data.
- gTTS needs an internet connection to generate speech.
- Your original model is trained only on the signs for which you collect samples. Start with a small set (for example A, B, Hello, Yes, No), collect multiple samples per sign, train, then expand.
- This prototype's frame-sequence classifier is not a complete natural-language sign-language translator.
