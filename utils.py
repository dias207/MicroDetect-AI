import streamlit as st
import numpy as np
from PIL import Image
import cv2
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
    """Извлечение признаков из изображения"""
    try:
        # Конвертация в numpy array если нужно
        if isinstance(image, Image.Image):
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img = img_array
        else:
            img = image
        
        # Изменение размера
        img = cv2.resize(img, target_size)
        
        # Конвертация в grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Извлечение признаков
        features = []
        
        # 1. Гистограмма градиентов
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        
        bins = np.int32(16*ang/(2*np.pi))
        bin_cells = bins[:16,:16], bins[16:,:16], bins[:16,16:], bins[16:,16:]
        mag_cells = mag[:16,:16], mag[16:,:16], mag[:16,16:], mag[16:,16:]
        hists = [np.bincount(b.ravel(), m.ravel(), 16) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)
        features.extend(hist)
        
        # 2. Статистические признаки
        features.extend([
            np.mean(gray),
            np.std(gray),
            np.min(gray),
            np.max(gray),
            np.median(gray)
        ])
        
        # 3. Текстурные признаки
        diff_x = np.diff(gray, axis=1).ravel()
        diff_y = np.diff(gray, axis=0).ravel()
        
        features.extend([
            np.mean(diff_x),
            np.std(diff_x),
            np.mean(diff_y),
            np.std(diff_y)
        ])
        
        # 4. Форма бактерий
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
            else:
                circularity = 0
            
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = w / h if h > 0 else 0
            
            features.extend([area, perimeter, circularity, aspect_ratio])
        else:
            features.extend([0, 0, 0, 0])
        
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
