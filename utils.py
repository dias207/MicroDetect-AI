import streamlit as st
import numpy as np
from PIL import Image
import os
import pickle

# Классы бактерий и их характеристики
BACTERIA_CLASSES = {
    0: {
        'name': 'Кишечные палочки',
        'shape': 'палочковидная',
        'rod_shaped': True,
        'description': 'Кишечные палочки — это бактерии палочковидной формы, которые обитают преимущественно в кишечнике человека и животных. Некоторые виды являются нормальной микрофлорой, а некоторые могут вызывать заболевания.'
    },
    1: {
        'name': 'Палочковидные бактерии',
        'shape': 'палочковидная',
        'rod_shaped': True,
        'description': 'Палочковидные бактерии — это микроорганизмы, имеющие вытянутую, продолговатую форму в виде палочки. Они широко распространены в природе и организме человека. Некоторые виды участвуют в нормальной микрофлоре, а другие могут быть возбудителями заболеваний.'
    },
    2: {
        'name': 'Сарцины',
        'shape': 'шаровидная',
        'rod_shaped': False,
        'description': 'Сарцины — это бактерии шаровидной формы, которые располагаются в виде пакетов или групп. Они отличаются характерным расположением клеток и используются как пример особой формы бактерий в микробиологии.'
    },
    3: {
        'name': 'Стафилококки',
        'shape': 'шаровидная',
        'rod_shaped': False,
        'description': 'Стафилококки — это бактерии шаровидной формы, которые чаще всего располагаются в виде скоплений, похожих на гроздья винограда. Они обитают на коже и слизистых оболочках и могут вызывать воспалительные процессы.'
    },
    4: {
        'name': 'Стрептококки',
        'shape': 'шаровидная',
        'rod_shaped': False,
        'description': 'Стрептококки — это бактерии шаровидной формы, которые обычно располагаются в виде цепочек. Они могут встречаться в полости рта, дыхательных путях и организме человека. Некоторые виды способны вызывать заболевания.'
    }
}

def extract_features_from_image(image, target_size=(64, 64)):
    """Извлечение признаков из изображения без OpenCV"""
    try:
        # Конвертация в numpy array
        img_array = np.array(image)
        
        # Изменение размера
        img = Image.fromarray(img_array).resize(target_size)
        
        # Конвертация в grayscale
        if len(img_array.shape) == 3:
            gray = img.convert('L')
        else:
            gray = img
        
        gray_array = np.array(gray)
        
        # Извлечение признаков
        features = []
        
        # 1. Простые статистические признаки
        features.extend([
            np.mean(gray_array),
            np.std(gray_array),
            np.min(gray_array),
            np.max(gray_array),
            np.median(gray_array)
        ])
        
        # 2. Гистограмма (16 бинов)
        hist, _ = np.histogram(gray_array, bins=16, range=(0, 256))
        features.extend(hist.tolist())
        
        # 3. Текстурные признаки (упрощенные)
        # Горизонтальные разницы
        h_diff = np.diff(gray_array, axis=1)
        features.extend([
            np.mean(h_diff),
            np.std(h_diff)
        ])
        
        # Вертикальные разницы
        v_diff = np.diff(gray_array, axis=0)
        features.extend([
            np.mean(v_diff),
            np.std(v_diff)
        ])
        
        # 4. Признаки формы (упрощенные)
        # Бинаризация для поиска контуров
        threshold = np.mean(gray_array)
        binary = gray_array > threshold
        
        # Площадь "бактерий"
        area = np.sum(binary)
        
        # Периметр (упрощенный)
        perimeter = 0
        for i in range(binary.shape[0]):
            for j in range(binary.shape[1]):
                if binary[i, j]:
                    # Проверяем соседей
                    if i > 0 and not binary[i-1, j]:
                        perimeter += 1
                    if i < binary.shape[0]-1 and not binary[i+1, j]:
                        perimeter += 1
                    if j > 0 and not binary[i, j-1]:
                        perimeter += 1
                    if j < binary.shape[1]-1 and not binary[i, j+1]:
                        perimeter += 1
        
        # Отношение сторон bounding box
        white_pixels = np.where(binary)
        if len(white_pixels[0]) > 0:
            h_min, h_max = np.min(white_pixels[0]), np.max(white_pixels[0])
            w_min, w_max = np.min(white_pixels[1]), np.max(white_pixels[1])
            height = h_max - h_min + 1
            width = w_max - w_min + 1
            aspect_ratio = width / height if height > 0 else 0
        else:
            aspect_ratio = 0
        
        features.extend([area, perimeter, aspect_ratio])
        
        return np.array(features)
        
    except Exception as e:
        print(f"Ошибка при обработке изображения: {e}")
        return None

def load_model():
    """Загрузка обученной модели"""
    try:
        model_path = 'model/bacteria_classifier.pkl'
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        return None

def predict_bacteria(model_data, image):
    """Предсказание типа бактерии"""
    try:
        # Извлечение признаков
        features = extract_features_from_image(image)
        if features is None:
            return None, 0.0
        
        # Предсказание
        model = model_data['model']
        predictions = model.predict_proba([features])[0]
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions)
        
        return predicted_class, confidence
        
    except Exception as e:
        print(f"Ошибка при предсказании: {e}")
        return None, 0.0

def get_bacteria_info(class_id):
    """Получение информации о бактерии по классу"""
    return BACTERIA_CLASSES.get(class_id, None)
