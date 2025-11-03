"""
Desktop Dashboard для Paint Defect Detection
Современный красивый интерфейс - Tkinter + YOLOv8
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import cv2
import numpy as np
from ultralytics import YOLO
import time
from pathlib import Path
import threading
import math
import json
import pandas as pd
from datetime import datetime
import os

class ModernPaintDefectDashboard:
    def __init__(self, root, model_path='best.pt'):
        self.root = root
        self.root.title("🎨 Paint Defect Detection Dashboard")
        self.root.geometry("1400x900")  # Уменьшил размер для лучшего fit на экране
        self.root.configure(bg='#0a0a0f')
        
        # Установка иконки (если есть)
        try:
            self.root.iconbitmap("paint_icon.ico")
        except:
            pass
        
        # Загрузка модели
        try:
            self.model = YOLO(model_path)
            self.model_loaded = True
            self.model_path = model_path
        except Exception as e:
            self.model_loaded = False
            self.model_path = None
            messagebox.showerror("Error", f"Failed to load model: {e}")
        
        # Переменные
        self.current_image = None
        self.original_image = None
        self.detection_result = None
        self.threshold = tk.DoubleVar(value=0.3)
        self.is_processing = False
        self.current_image_path = None
        
        # Стили
        self.colors = {
            'primary': '#6366f1',
            'primary_dark': '#4f46e5',
            'secondary': '#06d6a0',
            'danger': '#ef4444',
            'warning': '#f59e0b',
            'dark_bg': '#0a0a0f',
            'card_bg': '#1a1a2e',
            'card_bg_light': '#252547',
            'text_primary': '#ffffff',
            'text_secondary': '#94a3b8'
        }
        
        # Создание интерфейса
        self.create_ui()
        
    def create_ui(self):
        """Создание современного компактного интерфейса"""
        
        # ============ HEADER ============
        header_frame = tk.Frame(self.root, bg='#0a0a0f', height=80)  # Уменьшил высоту
        header_frame.pack(fill='x', pady=(0, 5))
        header_frame.pack_propagate(False)
        
        title_container = tk.Frame(header_frame, bg='#0a0a0f')
        title_container.pack(expand=True, fill='both')
        
        title_label = tk.Label(
            title_container,
            text="🎨 Paint Defect Detection",
            font=('Segoe UI', 22, 'bold'),  # Уменьшил шрифт
            fg='white',
            bg='#0a0a0f'
        )
        title_label.pack(pady=(10, 0))
        
        subtitle = tk.Label(
            title_container,
            text="AI-Powered Surface Quality Inspection",
            font=('Segoe UI', 10),  # Уменьшил шрифт
            fg='#94a3b8',
            bg='#0a0a0f'
        )
        subtitle.pack(pady=(0, 10))
        
        # ============ MAIN CONTENT ============
        main_container = tk.Frame(self.root, bg='#0a0a0f')
        main_container.pack(fill='both', expand=True, padx=15, pady=5)  # Уменьшил отступы
        
        # Left Panel - Image Display (65% ширины)
        left_panel = tk.Frame(main_container, bg='#0a0a0f')
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        # Image card
        image_card = tk.Frame(left_panel, bg=self.colors['card_bg'], relief='flat', bd=0)
        image_card.pack(fill='both', expand=True)
        
        # Card header
        card_header = tk.Frame(image_card, bg=self.colors['primary'], height=35)  # Уменьшил высоту
        card_header.pack(fill='x')
        card_header.pack_propagate(False)
        
        tk.Label(
            card_header,
            text="📸 IMAGE PREVIEW",
            font=('Segoe UI', 9, 'bold'),  # Уменьшил шрифт
            fg='white',
            bg=self.colors['primary']
        ).pack(side='left', padx=12, pady=8)
        
        # Image display area
        image_display_frame = tk.Frame(image_card, bg='#0f0f1a', relief='flat', bd=0)
        image_display_frame.pack(fill='both', expand=True, padx=2, pady=2)
        
        self.image_label = tk.Label(
            image_display_frame,
            text="\n\n🖼️\n\nUpload Image to Start Analysis\n\n",
            font=('Segoe UI', 12),  # Уменьшил шрифт
            fg='#475569',
            bg='#0f0f1a',
            justify='center'
        )
        self.image_label.pack(expand=True, fill='both', padx=15, pady=15)
        
        # Right Panel - Controls & Results (35% ширины)
        right_panel = tk.Frame(main_container, bg='#0a0a0f', width=350)  # Уменьшил ширину
        right_panel.pack(side='right', fill='y')
        right_panel.pack_propagate(False)
        
        # Создаем скроллируемую область для правой панели
        right_canvas = tk.Canvas(right_panel, bg='#0a0a0f', highlightthickness=0)
        scrollbar = ttk.Scrollbar(right_panel, orient='vertical', command=right_canvas.yview)
        
        scrollable_frame = tk.Frame(right_canvas, bg='#0a0a0f')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: right_canvas.configure(scrollregion=right_canvas.bbox("all"))
        )
        
        right_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        right_canvas.configure(yscrollcommand=scrollbar.set)
        
        right_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Controls Card
        controls_card = tk.Frame(scrollable_frame, bg=self.colors['card_bg'], relief='flat', bd=0)
        controls_card.pack(fill='x', pady=(0, 10))  # Уменьшил отступ
        
        tk.Label(
            controls_card,
            text="⚙️ CONTROLS",
            font=('Segoe UI', 10, 'bold'),  # Уменьшил шрифт
            fg='white',
            bg=self.colors['primary'],
            anchor='w'
        ).pack(fill='x', padx=0, pady=(0, 10))
        
        controls_content = tk.Frame(controls_card, bg=self.colors['card_bg'])
        controls_content.pack(fill='x', padx=12, pady=(0, 12))  # Уменьшил отступы
        
        # Upload button
        upload_btn = tk.Button(
            controls_content,
            text="📁 UPLOAD IMAGE",
            command=self.upload_image,
            font=('Segoe UI', 10, 'bold'),  # Уменьшил шрифт
            bg=self.colors['primary'],
            fg='white',
            activebackground=self.colors['primary_dark'],
            activeforeground='white',
            cursor='hand2',
            relief='flat',
            bd=0,
            padx=15,
            pady=8,  # Уменьшил padding
            width=16  # Уменьшил ширину
        )
        upload_btn.pack(pady=(0, 10))
        
        # Threshold section
        threshold_header = tk.Label(
            controls_content,
            text="Confidence Threshold:",
            font=('Segoe UI', 9, 'bold'),  # Уменьшил шрифт
            fg=self.colors['text_primary'],
            bg=self.colors['card_bg'],
            anchor='w'
        )
        threshold_header.pack(fill='x', pady=(5, 5))
        
        threshold_value_frame = tk.Frame(controls_content, bg=self.colors['card_bg'])
        threshold_value_frame.pack(fill='x')
        
        self.threshold_value_label = tk.Label(
            threshold_value_frame,
            text=f"{self.threshold.get():.1%}",
            font=('Segoe UI', 18, 'bold'),  # Уменьшил шрифт
            fg=self.colors['primary'],
            bg=self.colors['card_bg']
        )
        self.threshold_value_label.pack()
        
        threshold_slider = tk.Scale(
            threshold_value_frame,
            from_=0.1,
            to=0.9,
            resolution=0.05,
            orient='horizontal',
            variable=self.threshold,
            command=self.on_threshold_change,
            bg=self.colors['card_bg'],
            fg='white',
            troughcolor='#2d2d4d',
            highlightthickness=0,
            length=250,  # Уменьшил длину
            width=10,   # Уменьшил ширину
            sliderrelief='flat',
            sliderlength=18
        )
        threshold_slider.pack(pady=6, fill='x')
        
        # Detect button
        self.detect_btn = tk.Button(
            controls_content,
            text="🔍 DETECT DEFECTS",
            command=self.run_detection,
            font=('Segoe UI', 10, 'bold'),
            bg=self.colors['secondary'],
            fg='black',
            activebackground='#05c290',
            activeforeground='black',
            cursor='hand2',
            relief='flat',
            bd=0,
            padx=15,
            pady=8,
            width=16,
            state='disabled'
        )
        self.detect_btn.pack(pady=8)
        
        # Save button
        self.save_btn = tk.Button(
            controls_content,
            text="💾 SAVE RESULTS",
            command=self.save_results,
            font=('Segoe UI', 10, 'bold'),
            bg='#8b5cf6',
            fg='white',
            activebackground='#7c3aed',
            activeforeground='white',
            cursor='hand2',
            relief='flat',
            bd=0,
            padx=15,
            pady=8,
            width=16,
            state='disabled'
        )
        self.save_btn.pack(pady=(0, 8))
        
        # Status Card
        self.status_card = tk.Frame(scrollable_frame, bg=self.colors['card_bg'], relief='flat', bd=0)
        self.status_card.pack(fill='x', pady=(0, 10))
        
        self.status_content = tk.Frame(self.status_card, bg=self.colors['card_bg'])
        self.status_content.pack(fill='x', padx=12, pady=12)
        
        self.status_icon = tk.Label(
            self.status_content,
            text="⏳",
            font=('Segoe UI', 36),  # Уменьшил шрифт
            bg=self.colors['card_bg']
        )
        self.status_icon.pack(pady=6)
        
        self.status_title = tk.Label(
            self.status_content,
            text="Ready for Analysis",
            font=('Segoe UI', 14, 'bold'),  # Уменьшил шрифт
            fg='white',
            bg=self.colors['card_bg']
        )
        self.status_title.pack()
        
        self.status_text = tk.Label(
            self.status_content,
            text="Upload an image to start defect detection",
            font=('Segoe UI', 9),  # Уменьшил шрифт
            fg=self.colors['text_secondary'],
            bg=self.colors['card_bg']
        )
        self.status_text.pack(pady=6)
        
        # Statistics Card - компактная версия
        stats_card = tk.Frame(scrollable_frame, bg=self.colors['card_bg'], relief='flat', bd=0)
        stats_card.pack(fill='x', pady=(0, 10))
        
        tk.Label(
            stats_card,
            text="📊 STATISTICS",
            font=('Segoe UI', 10, 'bold'),
            fg='white',
            bg=self.colors['primary'],
            anchor='w'
        ).pack(fill='x', padx=0, pady=(0, 10))
        
        stats_content = tk.Frame(stats_card, bg=self.colors['card_bg'])
        stats_content.pack(fill='x', padx=12, pady=(0, 12))
        
        # Используем Frame с Grid для компактного размещения
        self.stats_frame = tk.Frame(stats_content, bg=self.colors['card_bg'])
        self.stats_frame.pack(fill='x')
        
        # Создаем метки для статистики
        self.stat_labels = {}
        stats_items = [
            ('Image Size:', 'image_size'),
            ('Threshold:', 'threshold'),
            ('Defects Found:', 'defects_count'),
            ('Inference Time:', 'inference_time'),
            ('Status:', 'status'),
            ('Quality:', 'quality')
        ]
        
        for i, (label_text, key) in enumerate(stats_items):
            # Label
            label = tk.Label(
                self.stats_frame,
                text=label_text,
                font=('Segoe UI', 8),  # Уменьшил шрифт
                fg=self.colors['text_secondary'],
                bg=self.colors['card_bg'],
                anchor='w'
            )
            label.grid(row=i, column=0, sticky='w', pady=1, padx=(0, 8))
            
            # Value
            value_label = tk.Label(
                self.stats_frame,
                text="--",
                font=('Segoe UI', 8, 'bold'),  # Уменьшил шрифт
                fg='white',
                bg=self.colors['card_bg'],
                anchor='w'
            )
            value_label.grid(row=i, column=1, sticky='w', pady=1)
            
            self.stat_labels[key] = value_label
        
        # Defects Card
        defects_card = tk.Frame(scrollable_frame, bg=self.colors['card_bg'], relief='flat', bd=0)
        defects_card.pack(fill='both', expand=True)
        
        tk.Label(
            defects_card,
            text="🔍 DETECTED DEFECTS",
            font=('Segoe UI', 10, 'bold'),
            fg='white',
            bg=self.colors['primary'],
            anchor='w'
        ).pack(fill='x', padx=0, pady=(0, 10))
        
        # Контейнер для списка дефектов с фиксированной высотой
        defects_container = tk.Frame(defects_card, bg=self.colors['card_bg'], height=250)  # Уменьшил высоту
        defects_container.pack(fill='both', expand=True, padx=2, pady=(0, 2))
        defects_container.pack_propagate(False)
        
        # Создаем canvas и scrollbar для дефектов
        defects_canvas = tk.Canvas(defects_container, bg=self.colors['card_bg'], highlightthickness=0)
        defects_scrollbar = ttk.Scrollbar(defects_container, orient='vertical', command=defects_canvas.yview)
        
        self.defects_frame = tk.Frame(defects_canvas, bg=self.colors['card_bg'])
        
        self.defects_frame.bind(
            '<Configure>',
            lambda e: defects_canvas.configure(scrollregion=defects_canvas.bbox('all'))
        )
        
        defects_canvas.create_window((0, 0), window=self.defects_frame, anchor='nw')
        defects_canvas.configure(yscrollcommand=defects_scrollbar.set)
        
        defects_canvas.pack(side='left', fill='both', expand=True, padx=(12, 0))
        defects_scrollbar.pack(side='right', fill='y', padx=(0, 12), pady=12)
        
        # Status bar
        status_bar = tk.Frame(self.root, bg='#1a1a2e', height=25)  # Уменьшил высоту
        status_bar.pack(side='bottom', fill='x')
        status_bar.pack_propagate(False)
        
        self.status_label = tk.Label(
            status_bar,
            text=f"● Model: {self.model_path if self.model_loaded else 'Not loaded'} | Status: Ready",
            font=('Segoe UI', 8),  # Уменьшил шрифт
            fg=self.colors['text_secondary'],
            bg='#1a1a2e',
            anchor='w'
        )
        self.status_label.pack(side='left', padx=15, pady=4)
        
        # Progress bar (скрыта по умолчанию)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            status_bar, 
            variable=self.progress_var, 
            mode='indeterminate',
            length=120  # Уменьшил длину
        )
        
        # Initial status update
        self.update_status("ready", "Upload an image to start analysis")
        self.update_statistics()  # Инициализация статистики
    
    def update_status(self, status_type, message=""):
        """Обновление статуса с красивыми иконками и цветами"""
        status_configs = {
            "ready": ("⏳", "Ready for Analysis", "#94a3b8"),
            "processing": ("🔄", "Processing...", "#f59e0b"),
            "success": ("✅", "No Defects Found", "#06d6a0"),
            "warning": ("⚠️", "Defects Detected", "#f59e0b"),
            "error": ("❌", "Analysis Failed", "#ef4444")
        }
        
        icon, title, color = status_configs.get(status_type, ("⏳", "Ready", "#94a3b8"))
        
        self.status_icon.config(text=icon)
        self.status_title.config(text=title, fg=color)
        self.status_text.config(text=message)
        
        # Показ/скрытие прогресс-бара
        if status_type == "processing":
            self.progress_bar.pack(side='right', padx=15, pady=4)
            self.progress_bar.start()
        else:
            self.progress_bar.stop()
            self.progress_bar.pack_forget()
    
    def update_statistics(self, result=None):
        """Обновление статистики в компактном формате"""
        if result is None:
            # Сброс статистики
            stats_data = {
                'image_size': '--',
                'threshold': f'{self.threshold.get():.1%}',
                'defects_count': '0',
                'inference_time': '-- ms',
                'status': 'Waiting',
                'quality': '--'
            }
        else:
            # Обновление с результатами
            stats_data = {
                'image_size': f"{result['image_size'][1]}×{result['image_size'][0]}",
                'threshold': f'{self.threshold.get():.1%}',
                'defects_count': str(result['num_defects']),
                'inference_time': f"{result['inference_time_ms']:.1f} ms",
                'status': '❌ REJECT' if result['has_defect'] else '✅ PASS',
                'quality': 'Needs Review' if result['has_defect'] else 'Excellent'
            }
        
        for key, value in stats_data.items():
            if key in self.stat_labels:
                self.stat_labels[key].config(text=value)
    
    def upload_image(self):
        """Загрузка изображения"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.update_status("processing", "Loading image...")
                
                # Загрузка изображения
                img = cv2.imread(file_path)
                if img is None:
                    raise ValueError("Failed to load image")
                    
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.original_image = img_rgb
                self.current_image = img_rgb.copy()
                self.current_image_path = file_path
                
                # Отображение
                self.display_image(img_rgb)
                
                # Активация кнопок
                self.detect_btn.config(state='normal')
                self.save_btn.config(state='disabled')  # Отключаем до детекции
                
                # Обновление статистики
                self.update_statistics()
                
                # Обновление статуса
                self.update_status("ready", "Click 'Detect Defects' to analyze")
                
                self.status_label.config(
                    text=f"● Model: {self.model_path} | Image: {Path(file_path).name} | Status: Ready"
                )
                
            except Exception as e:
                self.update_status("error", f"Failed to load image: {str(e)}")
                messagebox.showerror("Error", f"Failed to load image: {e}")
    
    def display_image(self, img_array, max_width=700, max_height=500):  # Уменьшил максимальные размеры
        """Отображение изображения с красивым оформлением"""
        img = Image.fromarray(img_array)
        
        # Масштабирование с сохранением пропорций
        img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        
        # Добавляем красивую рамку
        border_size = 6  # Уменьшил рамку
        bordered_img = Image.new('RGB', 
                               (img.width + border_size*2, img.height + border_size*2),
                               color='#2d2d4d')
        bordered_img.paste(img, (border_size, border_size))
        
        # Конвертация для Tkinter
        img_tk = ImageTk.PhotoImage(bordered_img)
        
        self.image_label.config(image=img_tk, text="")
        self.image_label.image = img_tk
    
    def run_detection(self):
        """Запуск детекции в отдельном потоке"""
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please upload an image first")
            return
        
        if not self.model_loaded:
            messagebox.showerror("Error", "Model not loaded")
            return
        
        # Блокировка кнопки
        self.detect_btn.config(state='disabled')
        self.save_btn.config(state='disabled')
        self.is_processing = True
        self.update_status("processing", "Analyzing image for defects...")
        
        # Запуск в отдельном потоке
        thread = threading.Thread(target=self.detect_defects)
        thread.daemon = True
        thread.start()
    
    def detect_defects(self):
        """Детекция дефектов"""
        try:
            start_time = time.time()
            
            # Детекция
            results = self.model(
                self.original_image,
                conf=self.threshold.get(),
                verbose=False
            )
            
            inference_time = (time.time() - start_time) * 1000
            
            # Парсинг результатов
            has_defect = False
            defects_list = []
            
            if len(results) > 0:
                result = results[0]
                
                if result.masks is not None and len(result.masks) > 0:
                    has_defect = True
                    
                    masks = result.masks.data.cpu().numpy()
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    class_names = {
                        0: 'Dirt',
                        1: 'Paint Runs',
                        2: 'Scratches',
                        3: 'Water Marks'
                    }
                    
                    for i in range(len(masks)):
                        mask = masks[i]
                        bbox = boxes[i]
                        conf = float(confidences[i])
                        cls_id = int(class_ids[i])
                        
                        area = np.sum(mask > 0.5) / (mask.shape[0] * mask.shape[1])
                        
                        defect_info = {
                            'class': class_names.get(cls_id, f'Class_{cls_id}'),
                            'confidence': conf,
                            'area_percent': area * 100,
                            'bbox': bbox.tolist(),
                            'severity': self.calculate_severity(area, conf)
                        }
                        
                        defects_list.append(defect_info)
                
                # Аннотированное изображение
                annotated = result.plot(conf=True, labels=True, boxes=True, masks=True)
                self.current_image = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            
            # Сортируем дефекты по серьезности
            defects_list.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Результат
            self.detection_result = {
                'has_defect': has_defect,
                'num_defects': len(defects_list),
                'defects': defects_list,
                'inference_time_ms': inference_time,
                'image_size': self.original_image.shape[:2],
                'timestamp': datetime.now().isoformat(),
                'threshold_used': float(self.threshold.get()),
                'model_path': self.model_path
            }
            
            # Обновление UI в главном потоке
            self.root.after(0, self.update_results)
            
        except Exception as e:
            self.root.after(0, lambda: self.update_status("error", f"Detection failed: {str(e)}"))
            self.root.after(0, lambda: self.detect_btn.config(state='normal'))
            self.root.after(0, lambda: messagebox.showerror("Error", f"Detection failed: {e}"))
    
    def calculate_severity(self, area, confidence):
        """Расчет серьезности дефекта"""
        severity_score = (area * 0.7 + confidence * 0.3) * 100
        
        if severity_score > 70:
            return "High"
        elif severity_score > 30:
            return "Medium"
        else:
            return "Low"
    
    def update_results(self):
        """Обновление результатов в UI"""
        result = self.detection_result
        
        # Отображение аннотированного изображения
        self.display_image(self.current_image)
        
        # Обновление статуса
        if result['has_defect']:
            self.update_status(
                "warning",
                f"Found {result['num_defects']} defect(s) requiring attention"
            )
        else:
            self.update_status(
                "success",
                "Paint surface meets quality standards"
            )
        
        # Обновление статистики
        self.update_statistics(result)
        
        # Обновление списка дефектов
        self.update_defects_list(result['defects'])
        
        # Активация кнопок
        self.detect_btn.config(state='normal')
        self.save_btn.config(state='normal')  # Активируем кнопку сохранения
        self.is_processing = False
        
        self.status_label.config(
            text=f"● Model: {self.model_path} | Detection complete ({result['inference_time_ms']:.1f}ms) | Defects: {result['num_defects']}"
        )
    
    def update_defects_list(self, defects):
        """Обновление списка дефектов с красивыми карточками"""
        # Очистка
        for widget in self.defects_frame.winfo_children():
            widget.destroy()
        
        if not defects:
            no_defects_frame = tk.Frame(self.defects_frame, bg=self.colors['card_bg_light'], relief='flat', bd=1)
            no_defects_frame.pack(fill='x', padx=8, pady=4)  # Уменьшил отступы
            
            tk.Label(
                no_defects_frame,
                text="✅ No defects detected",
                font=('Segoe UI', 10, 'bold'),  # Уменьшил шрифт
                fg=self.colors['secondary'],
                bg=self.colors['card_bg_light'],
                pady=12  # Уменьшил padding
            ).pack()
            return
        
        # Отображение дефектов
        severity_colors = {
            'High': '#ef4444',
            'Medium': '#f59e0b',
            'Low': '#3b82f6'
        }
        
        defect_icons = {
            'Dirt': '🟤',
            'Paint Runs': '💧',
            'Scratches': '🔴',
            'Water Marks': '💦'
        }
        
        for i, defect in enumerate(defects, 1):
            # Карточка дефекта
            defect_card = tk.Frame(
                self.defects_frame, 
                bg=self.colors['card_bg_light'], 
                relief='flat', 
                bd=1
            )
            defect_card.pack(fill='x', padx=8, pady=4)  # Уменьшил отступы
            
            # Header с иконкой серьезности
            header_frame = tk.Frame(defect_card, bg=severity_colors[defect['severity']])
            header_frame.pack(fill='x')
            
            header_text = f"{defect_icons.get(defect['class'], '⚫')} {defect['class']}"
            tk.Label(
                header_frame,
                text=header_text,
                font=('Segoe UI', 8, 'bold'),  # Уменьшил шрифт
                fg='white',
                bg=severity_colors[defect['severity']],
                anchor='w',
                padx=6,  # Уменьшил padding
                pady=3   # Уменьшил padding
            ).pack(fill='x')
            
            # Content
            content_frame = tk.Frame(defect_card, bg=self.colors['card_bg_light'])
            content_frame.pack(fill='x', padx=8, pady=6)  # Уменьшил отступы
            
            # Confidence
            conf_percent = defect['confidence'] * 100
            conf_frame = tk.Frame(content_frame, bg=self.colors['card_bg_light'])
            conf_frame.pack(fill='x', pady=1)  # Уменьшил отступы
            
            tk.Label(
                conf_frame,
                text=f"Confidence: {conf_percent:.1f}%",
                font=('Segoe UI', 7, 'bold'),  # Уменьшил шрифт
                fg='white',
                bg=self.colors['card_bg_light'],
                anchor='w'
            ).pack(side='left')
            
            # Severity
            tk.Label(
                conf_frame,
                text=f"Severity: {defect['severity']}",
                font=('Segoe UI', 7, 'bold'),  # Уменьшил шрифт
                fg=severity_colors[defect['severity']],
                bg=self.colors['card_bg_light'],
                anchor='e'
            ).pack(side='right')
            
            # Area information
            area_frame = tk.Frame(content_frame, bg=self.colors['card_bg_light'])
            area_frame.pack(fill='x', pady=1)  # Уменьшил отступы
            
            tk.Label(
                area_frame,
                text=f"Area: {defect['area_percent']:.2f}%",
                font=('Segoe UI', 7),  # Уменьшил шрифт
                fg=self.colors['text_secondary'],
                bg=self.colors['card_bg_light'],
                anchor='w'
            ).pack(side='left')
    
    def save_results(self):
        """Сохранение результатов анализа"""
        if self.detection_result is None:
            messagebox.showwarning("Warning", "No detection results to save")
            return
        
        try:
            # Базовое имя файла
            if self.current_image_path:
                base_name = Path(self.current_image_path).stem
            else:
                base_name = f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Создаем папку для результатов если её нет
            results_dir = Path("model/detection_results_" + base_name)
            results_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 1. Сохранение аннотированного изображения
            if self.current_image is not None:
                img_path = results_dir / f"{base_name}_annotated_{timestamp}.jpg"
                cv2.imwrite(str(img_path), cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR))
            
            # 2. Сохранение JSON с полными результатами
            json_path = results_dir / f"{base_name}_results_{timestamp}.json"
            
            # Подготовка данных для JSON
            save_data = self.detection_result.copy()
            save_data['image_filename'] = self.current_image_path
            save_data['save_timestamp'] = datetime.now().isoformat()
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            # 3. Сохранение CSV с таблицей дефектов
            if save_data['defects']:
                csv_path = results_dir / f"{base_name}_defects_{timestamp}.csv"
                
                # Подготовка данных для CSV
                defects_data = []
                for i, defect in enumerate(save_data['defects'], 1):
                    defect_row = {
                        'defect_id': i,
                        'class': defect['class'],
                        'confidence': f"{defect['confidence']:.3f}",
                        'confidence_percent': f"{defect['confidence'] * 100:.1f}%",
                        'area_percent': f"{defect['area_percent']:.2f}%",
                        'severity': defect['severity'],
                        'bbox': str(defect['bbox'])
                    }
                    defects_data.append(defect_row)
                
                df = pd.DataFrame(defects_data)
                df.to_csv(csv_path, index=False, encoding='utf-8')
            
            # 4. Создание сводного отчета
            report_path = results_dir / f"{base_name}_report_{timestamp}.txt"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 50 + "\n")
                f.write("PAINT DEFECT DETECTION REPORT\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Image: {self.current_image_path or 'Unknown'}\n")
                f.write(f"Model: {self.model_path}\n")
                f.write(f"Threshold: {self.threshold.get():.1%}\n\n")
                
                f.write("SUMMARY:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Image Size: {save_data['image_size'][1]}×{save_data['image_size'][0]}\n")
                f.write(f"Defects Found: {save_data['num_defects']}\n")
                f.write(f"Inference Time: {save_data['inference_time_ms']:.1f} ms\n")
                f.write(f"Quality Status: {'REJECT ❌' if save_data['has_defect'] else 'PASS ✅'}\n\n")
                
                if save_data['defects']:
                    f.write("DETECTED DEFECTS:\n")
                    f.write("-" * 20 + "\n")
                    for i, defect in enumerate(save_data['defects'], 1):
                        f.write(f"{i}. {defect['class']} | "
                               f"Confidence: {defect['confidence']*100:.1f}% | "
                               f"Area: {defect['area_percent']:.2f}% | "
                               f"Severity: {defect['severity']}\n")
                else:
                    f.write("No defects detected - Surface quality is excellent.\n")
            
            # Показать сообщение об успехе
            messagebox.showinfo(
                "Results Saved", 
                f"All results have been saved to:\n{results_dir}\n\n"
                f"• Annotated image\n• JSON data\n• CSV table\n• Text report"
            )
            
            # Обновление статуса
            self.status_label.config(
                text=f"● Results saved to {results_dir} | Files: 4"
            )
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save results: {e}")
    
    def on_threshold_change(self, value):
        """Обработчик изменения порога"""
        # Обновление значения
        self.threshold_value_label.config(text=f"{float(value):.1%}")
        
        # Обновление статистики
        if hasattr(self, 'stat_labels'):
            self.stat_labels['threshold'].config(text=f'{float(value):.1%}')
        
        # Автоматически переобнаруживать если изображение загружено
        if (self.original_image is not None and 
            hasattr(self, 'detection_result') and 
            self.detection_result and
            not self.is_processing):
            self.run_detection()


def main():
    """Запуск приложения"""
    # Настройка стиля
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass
    
    root = tk.Tk()
    
    # Путь к модели
    model_path = 'model/best.pt'
    
    # Проверка существования модели
    if not Path(model_path).exists():
        
        print(f"Warning: Model not found at {model_path}")
        # Спросим пользователя
        model_path = filedialog.askopenfilename(
            title="Select YOLOv8 model file",
            filetypes=[("Model files", "*.pt"), ("All files", "*.*")]
        )
        if not model_path:
            messagebox.showwarning(
                "Warning",
                "No model selected. The app will start but detection won't work.\n\n"
                "Please place your model file in the expected location or select it when prompted."
            )
    
    app = ModernPaintDefectDashboard(root, model_path)
    
    # Центрирование окна
    root.update_idletasks()
    x = (root.winfo_screenwidth() - root.winfo_reqwidth()) // 2
    y = (root.winfo_screenheight() - root.winfo_reqheight()) // 2
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()


if __name__ == "__main__":
    main()