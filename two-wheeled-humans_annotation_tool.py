import os
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO

# Константы
IMAGE_DIR = "dataset/images"
LABEL_DIR = "dataset/labels"

YOLO_MODEL = "yolov8s.pt"
SUPPORTED_FORMATS = (".jpg", ".jpeg", ".png")


class YOLOTwoWheeledHumansAnnotationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Инструмент для автоматической разметки "
                        "двухколесных людей для YOLO")

        self.canvas_width = 800
        self.canvas_height = 600
        self.image_offset_x = 0
        self.image_offset_y = 0
        self.scale_x = 1.0  # Инициализация scale_x
        self.scale_y = 1.0  # Инициализация scale_y

        # Конфигурация путей
        self.image_dir = IMAGE_DIR
        self.label_dir = LABEL_DIR
        self.supported_formats = SUPPORTED_FORMATS
        self.yolo_model = YOLO(YOLO_MODEL)  # Загрузка модели YOLOv8

        # Создаем папки, если они не существуют
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.label_dir, exist_ok=True)

        # Переменные состояния
        self.image_files = []
        self.current_image = None
        self.current_image_path = None
        self.current_label_path = None
        self.annotations = []
        self.start_x = None
        self.start_y = None
        self.rect = None
        self.image_on_canvas = None
        self.detected_pairs = []  # Для хранения обнаруженных пар
        self.auto_annotation_running = False  # Флаг для авто разметки
        self.current_auto_index = 0  # Текущий индекс при авто разметки

        # Настройка интерфейса
        self.setup_ui()

        # Загрузка списка изображений
        self.load_image_list()

    def setup_ui(self):
        """Настройка пользовательского интерфейса"""
        # Основные фреймы
        left_frame = tk.Frame(self.root, width=200)
        left_frame.pack(side=tk.LEFT,
                        fill=tk.Y,
                        padx=5,
                        pady=5)

        right_frame = tk.Frame(self.root)
        right_frame.pack(side=tk.RIGHT,
                         expand=True,
                         fill=tk.BOTH,
                         padx=5,
                         pady=5)

        # Фрейм списка изображений со скроллбаром
        list_frame = tk.Frame(left_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        # Список изображений
        self.image_listbox = tk.Listbox(list_frame, width=30)
        self.image_listbox.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        self.image_listbox.bind("<<ListboxSelect>>", self.on_image_select)

        # Скроллбар для списка изображений
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.image_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.image_listbox.yview)

        # Фрейм кнопок под списком
        button_frame = tk.Frame(left_frame)
        button_frame.pack(fill=tk.X, pady=5)

        # Кнопки добавления и удаления изображений
        add_btn = tk.Button(
            button_frame, text="Добавить изображения", command=self.add_images
        )
        add_btn.pack(fill=tk.X, pady=2)

        delete_btn = tk.Button(
            button_frame, text="Удалить изображение", command=self.delete_image
        )
        delete_btn.pack(fill=tk.X, pady=2)

        # Кнопка автоматической разметки всех неразмеченных изображений
        auto_all_btn = tk.Button(
            button_frame,
            text="Разметить все неразмеченные",
            command=self.auto_annotate_all_unlabeled,
            bg="lightgreen",
        )
        auto_all_btn.pack(fill=tk.X, pady=2)

        # Холст для отображения изображения
        self.canvas = tk.Canvas(right_frame, bg="gray", cursor="cross")
        self.canvas.pack(expand=True, fill=tk.BOTH)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)

        # Фрейм управления аннотациями
        control_frame = tk.Frame(right_frame)
        control_frame.pack(fill=tk.X, pady=5)

        # Поле ввода класса
        tk.Label(control_frame, text="Класс:").pack(side=tk.LEFT)
        self.class_entry = tk.Entry(control_frame, width=5)
        self.class_entry.pack(side=tk.LEFT, padx=5)
        self.class_entry.insert(0, "0")

        # Поле ввода YOLO формата
        tk.Label(control_frame, text="Формат YOLO:").pack(side=tk.LEFT,
                                                          padx=(10, 0))
        self.yolo_entry = tk.Entry(control_frame, width=30)
        self.yolo_entry.pack(side=tk.LEFT, padx=5)
        self.yolo_entry.bind("<Return>", self.update_annotation_from_entry)

        # Кнопки управления аннотациями
        add_btn = tk.Button(
            control_frame,
            text="Добавить аннотацию",
            command=self.add_annotation_from_entry,
        )
        add_btn.pack(side=tk.LEFT, padx=5)

        delete_btn = tk.Button(
            control_frame,
            text="Удалить выбранную",
            command=self.delete_selected_annotation,
        )
        delete_btn.pack(side=tk.LEFT)

        # Список аннотаций
        self.annotation_listbox = tk.Listbox(right_frame, height=10)
        self.annotation_listbox.pack(fill=tk.X, pady=5)
        self.annotation_listbox.bind("<<ListboxSelect>>",
                                     self.on_annotation_select
                                     )

        # Кнопка автоматической разметки
        auto_btn = tk.Button(
            right_frame,
            text="Автоматическая разметка",
            command=self.auto_annotate_twowheeledhuman,
            bg="lightblue",
        )
        auto_btn.pack(fill=tk.X, pady=5)

        # Кнопка сохранения
        save_btn = tk.Button(
            right_frame,
            text="Сохранить разметку",
            command=self.save_annotations
        )
        save_btn.pack(fill=tk.X, pady=5)

        # Добавляем обработчик изменения размера
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        # Добавляем обработчик нажатия SPACE для
        # прерывания автоматической разметки
        self.root.bind("<space>", self.stop_auto_annotation)

    def stop_auto_annotation(self, event=None):
        """Остановка автоматической разметки по нажатию SPACE"""
        if self.auto_annotation_running:
            self.auto_annotation_running = False
            messagebox.showinfo("Информация",
                                "Автоматическая разметка прервана")

    def auto_annotate_all_unlabeled(self):
        """Автоматическая разметка всех неразмеченных изображений"""
        # Получаем список всех изображений из списка
        all_images = self.image_listbox.get(0, tk.END)

        # Фильтруем неразмеченные изображения (без файлов .txt в labels)
        unlabeled_images = []
        for file in all_images:
            label_file = os.path.splitext(file)[0] + ".txt"
            label_path = os.path.join(self.label_dir, label_file)
            if not os.path.exists(label_path):
                unlabeled_images.append(file)

        if not unlabeled_images:
            messagebox.showinfo("Информация", "Все изображения уже размечены")
            return

        # Подтверждение начала автоматической разметки
        confirm = messagebox.askyesno(
            "Подтверждение",
            f"Найдено {len(unlabeled_images)} неразмеченных изображений."
            f"Начать автоматическую разметку?",
        )

        if not confirm:
            return

        # Запускаем автоматическую разметку
        self.auto_annotation_running = True
        self.current_auto_index = 0
        self.process_next_unlabeled(unlabeled_images)

    def process_next_unlabeled(self, unlabeled_images):
        """Обработка следующего неразмеченного изображения"""
        if not self.auto_annotation_running or self.current_auto_index >= len(
            unlabeled_images
        ):
            self.auto_annotation_running = False
            messagebox.showinfo("Информация",
                                "Автоматическая разметка завершена")
            return

        # Получаем текущее изображение
        filename = unlabeled_images[self.current_auto_index]
        self.current_image_path = os.path.join(self.image_dir, filename)

        # Очищаем все аннотации и временные элементы
        # перед обработкой новой картинки
        self.clear_canvas()
        self.annotations = []
        self.detected_pairs = []
        self.update_annotation_list()

        # Обновляем список файлов и выделяем текущее изображение
        index = self.image_listbox.get(0, tk.END).index(filename)
        self.image_listbox.selection_clear(0, tk.END)
        self.image_listbox.selection_set(index)
        self.image_listbox.see(index)

        # Помечаем текущее изображение желтым цветом
        self.image_listbox.itemconfig(index, {"bg": "yellow"})
        self.root.update()  # Обновляем интерфейс

        # Загружаем и отображаем изображение
        self.display_image()
        self.root.update()  # Обновляем интерфейс

        # Создаем путь для файла разметки
        label_file = os.path.splitext(filename)[0] + ".txt"
        self.current_label_path = os.path.join(self.label_dir, label_file)

        # Выполняем автоматическую разметку
        self.auto_annotate_twowheeledhuman()

        self.root.update()

        # Определяем цвет результата
        index = self.image_listbox.get(0, tk.END).index(filename)
        if self.annotations:  # Если найдены двух колесные люди
            self.image_listbox.itemconfig(index, {"bg": "light green"})
            self.save_annotations()  # Автоматически сохраняем разметку
        else:  # Если не найдены двух колесные люди
            self.image_listbox.itemconfig(index, {"bg": "light blue"})
            # НЕ создаем пустой файл разметки, если аннотаций нет

        # Переходим к следующему изображению
        self.current_auto_index += 1
        self.root.after(100,
                        lambda: self.process_next_unlabeled(unlabeled_images))

    def on_canvas_resize(self, event):
        """Обработчик изменения размера холста"""
        self.canvas_width = event.width
        self.canvas_height = event.height
        self.update_image_position()
        self.draw_annotations()
        # Если есть текущее изображение, перерисовываем его с новыми размерами
        if self.current_image_path:
            self.display_image()

    def update_image_position(self):
        """Обновляет позицию изображения на холсте"""
        if not self.current_image:
            return

        img_width = self.current_image.width * self.scale_x
        img_height = self.current_image.height * self.scale_y

        # Вычисляем смещение для центрирования изображения
        self.image_offset_x = (self.canvas_width - img_width) / 2
        self.image_offset_y = (self.canvas_height - img_height) / 2

    def calculate_area(self, box):
        """Вычисляет площадь прямоугольника"""
        x1, y1, x2, y2 = box
        return (x2 - x1) * (y2 - y1)

    def auto_annotate_twowheeledhuman(self):
        """Автоматическая разметка пар человек-велосипед и человек-мотоцикл"""
        if not self.current_image_path:
            messagebox.showwarning("Предупреждение",
                                   "Сначала выберите изображение")
            return

        try:
            # Загружаем изображение для YOLOv8
            img = cv2.imread(self.current_image_path)
            if img is None:
                raise ValueError("Не удалось загрузить изображение")

            # Получаем предсказания от YOLOv8
            results = self.yolo_model(img)

            # Очищаем предыдущие обнаруженные пары
            self.detected_pairs = []

            # Классы, которые мы ищем (COCO dataset classes)
            # 0 - person, 1 - bicycle, 3 - motorcycle
            person_boxes = []
            bicycle_boxes = []
            motorcycle_boxes = []

            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    coords = box.xyxy[0].tolist()
                    conf = float(box.conf[0])

                    # Фильтр по уверенности (только уверенные предсказания)
                    if conf < 0.5:
                        continue

                    if cls == 0:  # person
                        person_boxes.append(coords)
                    elif cls == 1:  # bicycle
                        bicycle_boxes.append(coords)
                    elif cls == 3:  # motorcycle
                        motorcycle_boxes.append(coords)

            # Ищем пересекающиеся пары
            twowheeledhuman_pairs = []

            # Проверяем пары человек-велосипед
            for p_box in person_boxes:
                for b_box in bicycle_boxes:
                    if self.boxes_intersect(p_box, b_box):
                        combined = self.combine_boxes(p_box, b_box)
                        twowheeledhuman_pairs.append(
                            {
                                "type": "twowheeledhuman",
                                "person_box": p_box,
                                "vehicle_box": b_box,
                                "combined_box": combined,
                                "area": self.calculate_area(combined),
                            }
                        )

            # Проверяем пары человек-мотоцикл
            for p_box in person_boxes:
                for m_box in motorcycle_boxes:
                    if self.boxes_intersect(p_box, m_box):
                        combined = self.combine_boxes(p_box, m_box)
                        twowheeledhuman_pairs.append(
                            {
                                "type": "twowheeledhuman",
                                "person_box": p_box,
                                "vehicle_box": m_box,
                                "combined_box": combined,
                                "area": self.calculate_area(combined),
                            }
                        )

            # Фильтруем дубликаты (пересекающиеся прямоугольники)
            filtered_pairs = []
            used_indices = set()

            for i in range(len(twowheeledhuman_pairs)):
                if i in used_indices:
                    continue

                current_pair = twowheeledhuman_pairs[i]
                max_area_pair = current_pair
                max_area = current_pair["area"]

                for j in range(i + 1, len(twowheeledhuman_pairs)):
                    if j in used_indices:
                        continue

                    other_pair = twowheeledhuman_pairs[j]
                    if self.boxes_intersect(
                        current_pair["combined_box"],
                        other_pair["combined_box"]
                    ):
                        if other_pair["area"] > max_area:
                            max_area = other_pair["area"]
                            max_area_pair = other_pair
                        used_indices.add(j)

                filtered_pairs.append(max_area_pair)
                used_indices.add(i)

            # Сохраняем обнаруженные пары
            self.detected_pairs = filtered_pairs

            # Очищаем текущие аннотации и добавляем только отфильтрованные пары
            self.annotations = []

            for pair in filtered_pairs:
                x1, y1, x2, y2 = pair["combined_box"]

                # Конвертируем в YOLO формат (относительные координаты)
                img_width = self.current_image.width
                img_height = self.current_image.height

                x_center = ((x1 + x2) / 2) / img_width
                y_center = ((y1 + y2) / 2) / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height

                # Все пары сохраняем с классом 0
                self.annotations.append(
                    {
                        "class": "0",  # Класс 0 для всех двух колесных
                        "x_center": x_center,
                        "y_center": y_center,
                        "width": width,
                        "height": height,
                        "type": pair["type"],
                        "person_box": pair["person_box"],
                        "vehicle_box": pair["vehicle_box"],
                    }
                )

            # Обновляем интерфейс
            self.update_annotation_list()
            self.draw_annotations()

        except Exception as e:
            messagebox.showerror("Ошибка",
                                 f"Ошибка автоматической разметки: {str(e)}")

    def boxes_intersect(self, box1, box2):
        """Проверяет, пересекаются ли два прямоугольника"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # Проверяем пересечение по оси X
        x_intersect = (x1_min <= x2_max) and (x1_max >= x2_min)

        # Проверяем пересечение по оси Y
        y_intersect = (y1_min <= y2_max) and (y1_max >= y2_min)

        return x_intersect and y_intersect

    def combine_boxes(self, box1, box2):
        """Объединяет два прямоугольника в один, который их покрывает"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        new_x_min = min(x1_min, x2_min)
        new_y_min = min(y1_min, y2_min)
        new_x_max = max(x1_max, x2_max)
        new_y_max = max(y1_max, y2_max)

        return (new_x_min, new_y_min, new_x_max, new_y_max)

    def on_image_select(self, event):
        """Обработчик выбора изображения из списка"""
        selection = self.image_listbox.curselection()
        if not selection:
            return

        # Полностью очищаем все аннотации
        self.annotations = []
        self.detected_pairs = []
        self.clear_canvas()

        index = selection[0]
        filename = self.image_listbox.get(index)
        self.current_image_path = os.path.join(self.image_dir, filename)

        # Загружаем соответствующую разметку
        label_file = os.path.splitext(filename)[0] + ".txt"
        self.current_label_path = os.path.join(self.label_dir, label_file)

        # Загружаем все аннотации из файла
        if os.path.exists(self.current_label_path):
            with open(self.current_label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        self.annotations.append(
                            {
                                "class": parts[0],
                                "x_center": float(parts[1]),
                                "y_center": float(parts[2]),
                                "width": float(parts[3]),
                                "height": float(parts[4]),
                            }
                        )

        # Отображаем изображение и аннотации
        self.display_image()
        self.update_annotation_list()

    def clear_canvas(self):
        """Полная очистка холста"""
        self.canvas.delete("all")
        self.image_on_canvas = None
        self.rect = None
        self.start_x = None
        self.start_y = None

    def draw_annotations(self):
        """Отрисовка всех аннотаций на холсте с учетом смещения"""
        self.canvas.delete("annotation")

        if not self.current_image:
            return

        # Рисуем все аннотации
        for ann in self.annotations:
            # Получаем координаты в формате YOLO
            x_center = ann["x_center"]
            y_center = ann["y_center"]
            width = ann["width"]
            height = ann["height"]

            # Вычисляем координаты углов
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2

            # Конвертируем в координаты холста с учетом смещения
            canvas_x1, canvas_y1 = self.convert_to_canvas_coords(x1, y1)
            canvas_x2, canvas_y2 = self.convert_to_canvas_coords(x2, y2)

            # Определяем цвет и метку
            if ann["class"] == "0":
                color = "red"
                label = "twowheeledhuman"
            else:
                color = "green"
                label = ann["class"]

            # Рисуем прямоугольник
            self.canvas.create_rectangle(
                canvas_x1,
                canvas_y1,
                canvas_x2,
                canvas_y2,
                outline=color,
                width=2,
                tags="annotation",
            )

            # Подписываем класс
            self.canvas.create_text(
                canvas_x1 + 5,
                canvas_y1 + 5,
                text=label,
                fill=color,
                anchor=tk.NW,
                tags="annotation",
            )

        # Рисуем временные прямоугольники обнаруженных пар
        for pair in self.detected_pairs:
            # Конвертируем координаты с учетом смещения
            p_x1, p_y1 = self.convert_to_canvas_coords(
                pair["person_box"][0] / self.current_image.width,
                pair["person_box"][1] / self.current_image.height,
            )
            p_x2, p_y2 = self.convert_to_canvas_coords(
                pair["person_box"][2] / self.current_image.width,
                pair["person_box"][3] / self.current_image.height,
            )

            v_x1, v_y1 = self.convert_to_canvas_coords(
                pair["vehicle_box"][0] / self.current_image.width,
                pair["vehicle_box"][1] / self.current_image.height,
            )
            v_x2, v_y2 = self.convert_to_canvas_coords(
                pair["vehicle_box"][2] / self.current_image.width,
                pair["vehicle_box"][3] / self.current_image.height,
            )

            c_x1, c_y1 = self.convert_to_canvas_coords(
                pair["combined_box"][0] / self.current_image.width,
                pair["combined_box"][1] / self.current_image.height,
            )
            c_x2, c_y2 = self.convert_to_canvas_coords(
                pair["combined_box"][2] / self.current_image.width,
                pair["combined_box"][3] / self.current_image.height,
            )

            # Рисуем прямоугольники
            self.canvas.create_rectangle(
                p_x1,
                p_y1,
                p_x2,
                p_y2,
                outline="yellow",
                width=1,
                tags="temp_annotation",
            )
            self.canvas.create_rectangle(
                v_x1, v_y1, v_x2, v_y2, outline="blue",
                width=1,
                tags="temp_annotation"
            )
            self.canvas.create_rectangle(
                c_x1, c_y1, c_x2, c_y2, outline="red",
                width=1,
                tags="temp_annotation"
            )

    def load_image_list(self):
        """Загрузка списка изображений из папки"""
        self.image_listbox.delete(0, tk.END)
        self.image_files = []

        # Получаем все файлы изображений из папки
        for file in os.listdir(self.image_dir):
            if file.lower().endswith(self.supported_formats):
                self.image_files.append(file)

        self.image_files.sort()  # Сортируем по имени

        # Добавляем в список, помечаем зеленым размеченные
        for file in self.image_files:
            label_file = os.path.splitext(file)[0] + ".txt"
            label_path = os.path.join(self.label_dir, label_file)

            if os.path.exists(label_path):
                self.image_listbox.insert(tk.END, file)
                self.image_listbox.itemconfig(tk.END, {"bg": "light green"})
            else:
                self.image_listbox.insert(tk.END, file)

    def add_images(self):
        """Добавление новых изображений в папку"""
        files = filedialog.askopenfilenames(
            title="Выберите изображения",
            filetypes=(("Изображения", "*.jpg *.jpeg *.png"),
                       ("Все файлы", "*.*")),
        )

        if files:
            for file in files:
                filename = os.path.basename(file)
                dest = os.path.join(self.image_dir, filename)

                # Обработка дубликатов имен
                counter = 1
                while os.path.exists(dest):
                    name, ext = os.path.splitext(filename)
                    dest = os.path.join(self.image_dir,
                                        f"{name}_{counter}{ext}")
                    counter += 1

                try:
                    # Копируем файл
                    with open(file, "rb") as f_src, open(dest, "wb") as f_dst:
                        f_dst.write(f_src.read())
                except Exception as e:
                    messagebox.showerror(
                        "Ошибка",
                        f"Не удалось скопировать {filename}: {str(e)}"
                    )

            # Обновляем список изображений
            self.load_image_list()

    def delete_image(self):
        """Удаление выбранного изображения и его разметки"""
        selection = self.image_listbox.curselection()
        if not selection:
            return

        index = selection[0]
        filename = self.image_listbox.get(index)

        # Подтверждение удаления
        confirm = messagebox.askyesno(
            "Подтвердите удаление",
            f"Вы уверены, что хотите удалить {filename} и его разметку?",
        )

        if confirm:
            # Удаляем файл изображения
            image_path = os.path.join(self.image_dir, filename)
            if os.path.exists(image_path):
                os.remove(image_path)

            # Удаляем файл разметки
            label_file = os.path.splitext(filename)[0] + ".txt"
            label_path = os.path.join(self.label_dir, label_file)
            if os.path.exists(label_path):
                os.remove(label_path)

            # Обновляем интерфейс
            self.load_image_list()
            self.clear_canvas()

    def display_image(self):
        """Отображение изображения на холсте с учетом изменения размеров"""
        self.clear_canvas()

        try:
            self.current_image = Image.open(self.current_image_path)
            img_width, img_height = self.current_image.size

            # Получаем текущие размеры холста
            self.canvas_width = self.canvas.winfo_width()
            self.canvas_height = self.canvas.winfo_height()

            if self.canvas_width <= 1 or self.canvas_height <= 1:
                self.canvas_width = 800
                self.canvas_height = 600

            # Вычисляем коэффициенты масштабирования
            ratio = min(self.canvas_width / img_width,
                        self.canvas_height / img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)

            # Сохраняем коэффициенты масштабирования
            self.scale_x = new_width / img_width
            self.scale_y = new_height / img_height

            # Обновляем позицию изображения
            self.update_image_position()

            # Масштабируем изображение
            resized_img = self.current_image.resize(
                (new_width, new_height), Image.LANCZOS
            )
            self.tk_image = ImageTk.PhotoImage(resized_img)

            # Отображаем изображение на холсте с учетом смещения
            self.image_on_canvas = self.canvas.create_image(
                self.canvas_width // 2,
                self.canvas_height // 2,
                anchor=tk.CENTER,
                image=self.tk_image,
            )

            # Рисуем аннотации
            self.draw_annotations()

        except Exception as e:
            messagebox.showerror(
                "Ошибка", f"Не удалось загрузить изображение: {str(e)}"
            )

    def convert_to_canvas_coords(self, x, y):
        """Конвертирует координаты изображения в
        координаты холста с учетом смещения"""
        img_x = x * self.current_image.width * self.scale_x
        img_y = y * self.current_image.height * self.scale_y
        return (self.image_offset_x + img_x, self.image_offset_y + img_y)

    def on_mouse_press(self, event):
        """Обработчик нажатия кнопки мыши на холсте"""
        if not self.image_on_canvas:
            return

        # Получаем координаты с учетом смещения изображения
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        # Проверяем, что клик был внутри изображения
        if not (
            self.image_offset_x
            <= canvas_x
            <= self.image_offset_x + self.current_image.width * self.scale_x
            and self.image_offset_y
            <= canvas_y
            <= self.image_offset_y + self.current_image.height * self.scale_y
        ):
            return

        # Запоминаем начальные координаты относительно изображения
        self.start_x = (canvas_x - self.image_offset_x) / self.scale_x
        self.start_y = (canvas_y - self.image_offset_y) / self.scale_y

        # Конвертируем обратно в координаты холста для отрисовки
        canvas_start_x = self.image_offset_x + self.start_x * self.scale_x
        canvas_start_y = self.image_offset_y + self.start_y * self.scale_y

        # Создаем новый прямоугольник
        self.rect = self.canvas.create_rectangle(
            canvas_start_x,
            canvas_start_y,
            canvas_start_x,
            canvas_start_y,
            outline="red",
            width=2,
            tags="rect",
        )

    def on_mouse_drag(self, event):
        """Обработчик перемещения мыши с зажатой кнопкой"""
        if self.rect and self.start_x is not None and self.start_y is not None:
            canvas_x = self.canvas.canvasx(event.x)
            canvas_y = self.canvas.canvasy(event.y)

            # Конвертируем текущие координаты в координаты изображения
            current_x = (canvas_x - self.image_offset_x) / self.scale_x
            current_y = (canvas_y - self.image_offset_y) / self.scale_y

            # Ограничиваем координаты размерами изображения
            current_x = max(0, min(self.current_image.width, current_x))
            current_y = max(0, min(self.current_image.height, current_y))

            # Конвертируем обратно в координаты холста
            canvas_current_x = self.image_offset_x + current_x * self.scale_x
            canvas_current_y = self.image_offset_y + current_y * self.scale_y

            # Конвертируем стартовые координаты обратно в координаты холста
            canvas_start_x = self.image_offset_x + self.start_x * self.scale_x
            canvas_start_y = self.image_offset_y + self.start_y * self.scale_y

            # Обновляем координаты прямоугольника
            self.canvas.coords(
                self.rect,
                canvas_start_x,
                canvas_start_y,
                canvas_current_x,
                canvas_current_y,
            )

    def on_mouse_release(self, event):
        """Обработчик отпускания кнопки мыши"""
        if not self.rect or self.start_x is None or self.start_y is None:
            return

        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        # Конвертируем конечные координаты в координаты изображения
        end_x = (canvas_x - self.image_offset_x) / self.scale_x
        end_y = (canvas_y - self.image_offset_y) / self.scale_y

        # Ограничиваем координаты размерами изображения
        end_x = max(0, min(self.current_image.width, end_x))
        end_y = max(0, min(self.current_image.height, end_y))

        # Проверяем, что прямоугольник имеет достаточный размер
        if abs(end_x - self.start_x) < 5 or abs(end_y - self.start_y) < 5:
            self.canvas.delete(self.rect)
            self.rect = None
            self.start_x = None
            self.start_y = None
            return

        # Вычисляем координаты в формате YOLO (относительные)
        x_center = ((self.start_x + end_x) / 2) / self.current_image.width
        y_center = ((self.start_y + end_y) / 2) / self.current_image.height
        width = abs(end_x - self.start_x) / self.current_image.width
        height = abs(end_y - self.start_y) / self.current_image.height

        # Получаем класс из поля ввода
        class_id = self.class_entry.get().strip()
        if not class_id:
            class_id = "0"

        # Добавляем новую аннотацию
        self.annotations.append(
            {
                "class": class_id,
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height,
            }
        )

        # Обновляем интерфейс
        self.update_annotation_list()
        self.annotation_listbox.selection_clear(0, tk.END)
        self.annotation_listbox.selection_set(len(self.annotations) - 1)
        self.annotation_listbox.see(len(self.annotations) - 1)
        self.update_yolo_entry_from_selection()
        self.draw_annotations()

        # Сбрасываем состояние
        self.rect = None
        self.start_x = None
        self.start_y = None

    def update_annotation_list(self):
        """Обновление списка аннотаций"""
        self.annotation_listbox.delete(0, tk.END)

        for i, ann in enumerate(self.annotations):
            self.annotation_listbox.insert(
                tk.END,
                f"{i}: класс={ann['class']} "
                f"x={ann['x_center']:.4f} y={ann['y_center']:.4f} "
                f"w={ann['width']:.4f} h={ann['height']:.4f}",
            )

    def on_annotation_select(self, event):
        """Обработчик выбора аннотации из списка"""
        selection = self.annotation_listbox.curselection()
        if not selection:
            return

        self.update_yolo_entry_from_selection()
        self.draw_annotations()

    def update_yolo_entry_from_selection(self):
        """Обновление поля YOLO формата по выбранной аннотации"""
        selection = self.annotation_listbox.curselection()
        if not selection:
            self.yolo_entry.delete(0, tk.END)
            return

        index = selection[0]
        if index >= len(self.annotations):
            return

        ann = self.annotations[index]
        yolo_str = f"{ann['class']} {ann['x_center']:.6f} {ann['y_center']:.6f} {ann['width']:.6f} {ann['height']:.6f}"
        self.yolo_entry.delete(0, tk.END)
        self.yolo_entry.insert(0, yolo_str)
        self.class_entry.delete(0, tk.END)
        self.class_entry.insert(0, ann["class"])

    def update_annotation_from_entry(self, event=None):
        """Обновление аннотации из поля YOLO формата"""
        selection = self.annotation_listbox.curselection()
        if not selection:
            return

        index = selection[0]
        if index >= len(self.annotations):
            return

        # Парсим строку YOLO
        yolo_str = self.yolo_entry.get().strip()
        parts = yolo_str.split()

        if len(parts) != 5:
            messagebox.showerror(
                "Ошибка",
                "Неверный формат YOLO. Ожидается: "
                "класс x_center y_center width height",
            )
            return

        try:
            # Валидация и обновление аннотации
            class_id = parts[0]
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            if not (
                0 <= x_center <= 1
                and 0 <= y_center <= 1
                and 0 <= width <= 1
                and 0 <= height <= 1
            ):
                raise ValueError("Координаты должны быть в диапазоне [0, 1]")

            self.annotations[index] = {
                "class": class_id,
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height,
            }

            # Обновляем интерфейс
            self.update_annotation_list()
            self.annotation_listbox.selection_clear(0, tk.END)
            self.annotation_listbox.selection_set(index)
            self.draw_annotations()

        except ValueError as e:
            messagebox.showerror("Ошибка", f"Неверные значения: {str(e)}")

    def add_annotation_from_entry(self):
        """Добавление новой аннотации из поля YOLO формата"""
        yolo_str = self.yolo_entry.get().strip()
        if not yolo_str:
            return

        parts = yolo_str.split()

        if len(parts) != 5:
            messagebox.showerror(
                "Ошибка",
                "Неверный формат YOLO. Ожидается: "
                "класс x_center y_center width height",
            )
            return

        try:
            # Валидация и добавление аннотации
            class_id = parts[0]
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            if not (
                0 <= x_center <= 1
                and 0 <= y_center <= 1
                and 0 <= width <= 1
                and 0 <= height <= 1
            ):
                raise ValueError("Координаты должны быть в диапазоне [0, 1]")

            self.annotations.append(
                {
                    "class": class_id,
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height,
                }
            )

            # Обновляем интерфейс
            self.update_annotation_list()
            self.annotation_listbox.selection_clear(0, tk.END)
            self.annotation_listbox.selection_set(len(self.annotations) - 1)
            self.draw_annotations()

        except ValueError as e:
            messagebox.showerror("Ошибка", f"Неверные значения: {str(e)}")

    def delete_selected_annotation(self):
        """Удаление выбранной аннотации"""
        selection = self.annotation_listbox.curselection()
        if not selection:
            return

        index = selection[0]
        if index >= len(self.annotations):
            return

        # Удаляем аннотацию из списка
        del self.annotations[index]

        # Полностью перерисовываем холст
        self.canvas.delete("all")

        # Восстанавливаем изображение
        if self.image_on_canvas:
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            self.image_on_canvas = self.canvas.create_image(
                canvas_width // 2,
                canvas_height // 2,
                anchor=tk.CENTER,
                image=self.tk_image,
            )

        # Рисуем оставшиеся аннотации
        self.draw_annotations()

        # Очищаем поле ввода
        self.yolo_entry.delete(0, tk.END)

        # Обновляем список аннотаций
        self.update_annotation_list()

    def save_annotations(self):
        """Сохранение аннотаций в файл"""
        if not self.current_image_path or not self.current_label_path:
            return

        # Если аннотаций нет - удаляем файл разметки
        if len(self.annotations) == 0:
            try:
                if os.path.exists(self.current_label_path):
                    os.remove(self.current_label_path)
                    messagebox.showinfo("Успех",
                                        "Файл разметки удален (нет аннотаций)")

                    # Обновляем цвет в списке файлов
                    filename = os.path.basename(self.current_image_path)
                    index = self.image_listbox.get(0, tk.END).index(filename)
                    self.image_listbox.itemconfig(index, {"bg": "white"})
                return
            except Exception as e:
                messagebox.showerror(
                    "Ошибка", f"Не удалось удалить файл разметки: {str(e)}"
                )
                return

        # Сохраняем аннотации в файл
        try:
            with open(self.current_label_path, "w") as f:
                for ann in self.annotations:
                    f.write(
                        f"{ann['class']} {ann['x_center']:.6f} {ann['y_center']:.6f} {ann['width']:.6f} {ann['height']:.6f}\n"
                    )

            # Обновляем цвет в списке файлов
            filename = os.path.basename(self.current_image_path)
            index = self.image_listbox.get(0, tk.END).index(filename)
            self.image_listbox.itemconfig(index, {"bg": "light green"})

            self.root.update()  # Обновляем интерфейс

        except Exception as e:
            messagebox.showerror("Ошибка",
                                 f"Не удалось сохранить разметку: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOTwoWheeledHumansAnnotationApp(root)
    root.geometry("1000x700")
    root.mainloop()
