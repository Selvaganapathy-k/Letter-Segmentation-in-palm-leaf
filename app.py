import streamlit as st
import cv2
import numpy as np
import os

st.title("🔤 Character Segmentation - Palm Leaf / Tamil Text")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:

    # Read uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Original Image", use_column_width=True)

    # ---------------- Grayscale ----------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    st.image(gray, caption="Grayscale")

    # ---------------- Adaptive Threshold ----------------
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        21,
        11,
    )

    st.image(thresh, caption="Adaptive Threshold")

    # Invert for contour detection
    thresh_inv = cv2.bitwise_not(thresh)

    contours, _ = cv2.findContours(
        thresh_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    thresh_clean = thresh.copy()
    thresh_box = thresh.copy()

    min_area = 50
    max_area = 50000

    segmented_characters = []

    # ---------------- Character Detection ----------------
    for contour in contours:
        area = cv2.contourArea(contour)

        # Remove small noise
        if area < min_area:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(thresh_clean, (x, y), (x + w, y + h), (255, 255, 255), -1)

        # Detect valid characters
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(contour)

            # Draw bounding box
            cv2.rectangle(thresh_box, (x, y), (x + w, y + h), (0, 255, 0), 1)

            # Crop character
            char_img = thresh[y:y+h, x:x+w]
            segmented_characters.append(char_img)

    st.image(thresh_clean, caption="Noise Removed")
    st.image(thresh_box, caption="Detected Characters")

    # ---------------- Display Segmented Characters ----------------
    st.subheader("Extracted Characters")

    for i, char in enumerate(segmented_characters):
        st.image(char, caption=f"Character {i+1}", use_column_width=True)
