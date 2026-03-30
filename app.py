import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

st.title("🤖 AI Object Remover (Auto Person Remove)")

# Load model (downloads automatically first time)
model = YOLO("yolov8n.pt")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image.thumbnail((640, 640))

    img_array = np.array(image)

    st.image(image, caption="Original Image", use_column_width=True)

    if st.button("🚀 Remove Person Automatically"):
        results = model(img_array)

        mask = np.zeros(img_array.shape[:2], dtype=np.uint8)

        for r in results:
            for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
                if int(cls) == 0:  # class 0 = person
                    x1, y1, x2, y2 = map(int, box)

                    # Create mask for detected person
                    mask[y1:y2, x1:x2] = 255

        # Inpaint to remove person
        result = cv2.inpaint(img_array, mask, 5, cv2.INPAINT_TELEA)

        st.image(result, caption="🧼 Person Removed", use_column_width=True)

        # Download
        import io
        buf = io.BytesIO()
        Image.fromarray(result).save(buf, format="PNG")

        st.download_button(
            "📥 Download",
            data=buf.getvalue(),
            file_name="ai_removed.png",
            mime="image/png"
        )