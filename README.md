:

📌 Letter Segmentation in Palm Leaf Manuscripts

Live Demo ➤ https://letter-segmentation-in-palm-leaf-selva.streamlit.app/

A web application that performs character segmentation on palm leaf manuscript images using OpenCV and Streamlit.
This tool allows you to upload a palm leaf image, preprocess it, and visualize each detected character separately.

🚀 Features

📥 Upload any manuscript image (PNG, JPG, JPEG)

🖤 Grayscale conversion

📊 Adaptive thresholding

🧼 Noise removal

✂️ Character detection & segmentation

🖼️ Display segmented characters in the browser

🛠 Technology Stack
Tool	Purpose
Python	Programming language
OpenCV	Computer Vision
NumPy	Array processing
Streamlit	Web UI
streamlit-webrtc	(Optional) Webcam support if extended
🖥 App Screenshot

(Include a screenshot/GIF here if you want a visual preview in README)

📂 Project Structure
letter-segmentation/
│
├── app.py
├── requirements.txt
├── runtime.txt
└── README.md

⚙️ Quick Setup (Local)
1️⃣ Clone Repository
git clone https://github.com/your-username/letter-segmentation.git
cd letter-segmentation

2️⃣ Create Virtual Environment (Recommended)
python -m venv venv


Activate (Windows):

venv\Scripts\activate


Activate (Mac/Linux):

source venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the App
streamlit run app.py


The app will open in your browser.

🧠 How It Works

Upload a manuscript image

Convert to grayscale

Threshold to binary

Invert for contour detection

Remove small noise

Detect bounding boxes for each character

Show each character separately

📌 Deployment

This app is deployed using Streamlit Cloud.

Live Link:
👉 https://letter-segmentation-in-palm-leaf-selva.streamlit.app/

If you want your own deployment:

Push repo to GitHub

Go to https://streamlit.io/cloud

Connect your GitHub

Deploy

📝 requirements.txt
streamlit
opencv-python-headless
numpy

📌 runtime.txt
python-3.11

📈 Future Improvements

✅ Add bounding box sorting (left-to-right, top-to-bottom)
✅ Save segmented characters as ZIP
✅ Add download button
✅ Line segmentation + character segmentation combined
✅ UI sliders to adjust thresholds

👨‍💻 Author

Selvaganapathy K