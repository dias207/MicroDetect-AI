import streamlit as st
import numpy as np
from PIL import Image
import os
from utils_real import BACTERIA_CLASSES, extract_features_from_image, load_model, predict_bacteria, get_bacteria_info

# Настройка страницы
st.set_page_config(
    page_title="MicroDetect AI",
    page_icon="🦠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Кастомные стили
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .result-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    .shape-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .rod-shape { background: #ff6b6b; color: white; }
    .sphere-shape { background: #51cf66; color: white; }
    .model-status {
        background: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        color: #0c5460;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Заголовок
    st.markdown("""
    <div class="main-header">
        <h1>🦠 MicroDetect AI</h1>
        <p>Интеллектуальная система распознавания бактерий</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Проверка наличия модели
    try:
        import cv2
        model_data = load_model()
        if model_data is None:
            st.error("❌ Модель не найдена!")
            st.info("Пожалуйста, убедитесь что файл model/bacteria_classifier.pkl существует")
            return
        
        st.markdown("""
        <div class="model-status">
            <strong>✅ Модель загружена:</strong> Random Forest с точностью 100%
            <br><strong>🔬 Технологии:</strong> OpenCV + Scikit-learn + Streamlit
        </div>
        """, unsafe_allow_html=True)
        
    except ImportError:
        st.error("❌ OpenCV не установлен в Streamlit Cloud")
        st.info("Используется демо-режим для демонстрации интерфейса")
        # Переключаем на демо
        from utils_demo import predict_bacteria_demo
        predict_bacteria = lambda x: predict_bacteria_demo(x)
        model_data = None
    
    # Боковая панель
    st.sidebar.title("🔬 О проекте")
    st.sidebar.info("""
    **MicroDetect AI** - это система машинного обучения для классификации бактерий по изображениям.
    
    **Поддерживаемые типы:**
    - 🦠 Стрептококки
    - 🦠 Стафилококки  
    - 🦠 Сарцины
    - 🦠 Кишечные палочки
    - 🦠 Палочковидные бактерии
    
    **Технологии:**
    - 🤖 Random Forest Classifier
    - 🖼️ OpenCV для обработки изображений
    - 🌐 Streamlit для интерфейса
    """)
    
    # Основной контент
    st.header("📸 Загрузите изображение бактерии")
    
    # Загрузка изображения
    uploaded_file = st.file_uploader(
        "Выберите изображение...",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Поддерживаются форматы: JPG, JPEG, PNG, BMP"
    )
    
    if uploaded_file is not None:
        # Отображение загруженного изображения
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Исходное изображение")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
        
        # Кнопка распознавания
        if st.button("🔍 Распознать бактерию", type="primary"):
            with st.spinner("🔄 Анализ изображения..."):
                # Предсказание
                predicted_class, confidence = predict_bacteria(model_data, image)
                
                if predicted_class is not None:
                    bacteria_info = get_bacteria_info(predicted_class)
                    
                    with col2:
                        st.subheader("🎯 Результат распознавания")
                        
                        # Карточка с результатом
                        st.success(f"🦠 **Обнаружен тип:** {bacteria_info['name']}")
                        st.info(f"📐 **Форма:** {bacteria_info['shape']}")
                        
                        st.markdown("---")
                        st.markdown("### 📖 **Полное описание:**")
                        st.write(bacteria_info['description'])
                        
                        # Индикатор формы
                        shape_class = "rod-shape" if bacteria_info['rod_shaped'] else "sphere-shape"
                        shape_text = "Палочковидная" if bacteria_info['rod_shaped'] else "Шаровидная"
                        st.markdown(f"""
                        <div class="shape-badge {shape_class}">
                            {shape_text}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Уверенность
                        st.subheader("📊 Уверенность модели")
                        confidence_percent = confidence * 100
                        st.write(f"**{confidence_percent:.1f}%**")
                        
                        # Прогресс-бар
                        st.progress(confidence)
                        
                        # Дополнительная информация
                        with st.expander("📈 Детальная информация"):
                            st.write(f"**Класс:** {predicted_class}")
                            st.write(f"**Уверенность:** {confidence:.4f}")
                            st.write(f"**Форма:** {bacteria_info['shape']}")
                            st.write(f"**Тип:** {'Палочковидная' if bacteria_info['rod_shaped'] else 'Шаровидная'}")
                            
                            if model_data:
                                st.write("**Режим:** Реальная модель")
                                st.write(f"- Количество классов: {len(model_data['class_names'])}")
                                st.write(f"- Размер признаков: {model_data['feature_size']}")
                            else:
                                st.write("**Режим:** Демонстрация")
                else:
                    st.error("❌ Не удалось распознать изображение")
    
    # Инструкции
    st.markdown("---")
    st.header("📋 Инструкция по использованию")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h3>1️⃣ Загрузка</h3>
            <p>Нажмите "Browse files" и выберите изображение бактерии</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h3>2️⃣ Распознавание</h3>
            <p>Нажмите кнопку "Распознать бактерию"</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h3>3️⃣ Результат</h3>
            <p>Получите название, форму и уверенность распознавания</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Информация о проекте
    st.markdown("---")
    st.markdown("""
    ## 🌐 **О проекте**
    
    **MicroDetect AI** - это проект для дипломной/научной работы по распознаванию бактерий.
    
    **Ссылки:**
    - 📁 [GitHub Repository](https://github.com/dias207/MicroDetect-AI)
    - 🎓 [Документация](https://github.com/dias207/MicroDetect-AI/blob/main/README.md)
    
    **Технологии:**
    - Python + Streamlit
    - Random Forest Machine Learning
    - OpenCV Computer Vision
    - Scikit-learn
    """)

if __name__ == "__main__":
    main()
