import streamlit as st
from PIL import Image
from model import load_model, predict
 
st.set_page_config(page_title="EfficientNet Image Classifier")
 
@st.cache_resource  # модель завантажується один раз
def get_model():
    return load_model()
 
st.title("EfficientNetB0 — Класифікація зображень")
st.write("Завантажте картинку для класифікації за допомогою EfficientNetB0")
 
with st.spinner("Завантаження моделі..."):
    model = get_model()
 
uploaded_file = st.file_uploader("Завантажте JPG/PNG", type=["jpg", "jpeg", "png"])
 
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Завантажене зображення", width="stretch")
 
    if st.button("Розпізнати", type="primary"):
        with st.spinner("Виконується класифікація..."):
            preds = predict(model, image)
 
        st.subheader("Результати:")
        for _, label, prob in preds:
            st.write(f"**{label}** — {prob*100:.2f}%")
            st.progress(float(prob))
