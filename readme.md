![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white&style=flat)
![YOLOv8](https://img.shields.io/badge/YOLOv8-00FFFF?logo=yolo&logoColor=black&style=flat)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white&style=flat)
![Tkinter](https://img.shields.io/badge/Tkinter-3776AB?logo=python&logoColor=white&style=flat)
![Pillow](https://img.shields.io/badge/Pillow-3776AB?logo=python&logoColor=white&style=flat)

# YOLO Two Wheeled Humans Annotation Tool

Инструмент для автоматической разметки людей на двухколесных транспортных средствах (велосипеды, мотоциклы) в формате YOLO.
Программа использует современные технологии компьютерного зрения для упрощения процесса аннотирования.
Ищет пары человека с двух колесным транспортом и объединяет их.

## 🔥 Особенности

- **Автоматическое обнаружение** пар "человек + транспорт" (велосипед/мотоцикл)
- **Интерактивный интерфейс** для ручной корректировки разметки
- **Пакетная обработка** всех неразмеченных изображений
- **Поддержка формата YOLO** для совместимости с популярными фреймворками
- **Визуализация результатов** с цветовой индикацией

## 🖥️ Использование

### Добавление изображений

```
Нажмите "Добавить изображения" и выберите файлы
```
либо загрузите файлы в папку /dataset/images/

### Автоматическая разметка
```
Выберите изображение → Нажмите "Автоматическая разметка"
```

### Ручная корректировка
```
- Рисуйте прямоугольники мышью
- Редактируйте координаты в поле YOLO формата
- Удаляйте аннотации кнопкой "Удалить выбранную"
```

### Пакетная обработка
```
Нажмите "Разметить все неразмеченные" для массового аннотирования
```

### Сохранение результатов
```
Разметка автоматически сохраняется в YOLO-формате
```

## 🖼 Скриншоты интерфейса

<div align="center">
  <img src="https://github.com/NikitaPolechshuk/YOLO-Two-Wheeled-Human-Annotation-Tool/raw/main/screenshots/scr_01.png" width="45%" alt="Главное окно программы">
  <img src="https://github.com/NikitaPolechshuk/YOLO-Two-Wheeled-Human-Annotation-Tool/raw/main/screenshots/scr_02.png" width="45%" alt="Пример разметки">
</div>

  > желтый прямоугольник - обнаруженный человек
  > синий прямоугольник - двух колесный транспорт
  > красный пямоугольник - их объединение


## 🚀 Установка и запуск

### 1. Клонирование репозитория
```
git clone https://github.com/NikitaPolechshuk/YOLO-Two-Wheeled-Humans-Annotation-Tool.git
cd YOLO-Two-Wheeled-Humans-Annotation-Tool
```

### 2. Cоздать и активировать виртуальное окружение:
```
python3 -m venv venv   # Для Linux/Mac
python -m venv venv   # Для Windows
```

```
source venv/bin/activate   # Для Linux/Mac
venv\Scripts\activate.bat   # Для Windows
```

### 3. Установить зависимости из файла requirements.txt:
```
pip install -r requirements.txt
```

### 6. Запуск программу
```
python two-wheeled-humans_annotation_tool.py
```

## ⚙️ Настройка

Измените в коде:
```
# Пути к данным
IMAGE_DIR = "dataset/images"  # 📁 Папка с исходными изображениями
LABEL_DIR = "dataset/labels"      # 📝 Папка для аннотаций YOLO

# Модель детекции
YOLO_MODEL = "yolov8s.pt"          # 🧠 Модель YOLO (можно заменить на custom)

# Поддерживаемые форматы
SUPPORTED_FORMATS = (".jpg", ".jpeg", ".png")  # 🖼️ Расширения файлов
```


