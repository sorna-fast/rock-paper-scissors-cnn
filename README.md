# Rock-Paper-Scissors Image Classification with CNN

A complete deep learning project that implements a Convolutional Neural Network (CNN) to classify images of hand gestures into rock, paper, and scissors categories using TensorFlow and Keras.

![Project Banner](https://img.shields.io/badge/Project-Rock--Paper--Scissors--CNN-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-orange) ![Python](https://img.shields.io/badge/Python-3.9%2B-green)

## Features

- **Custom CNN Architecture**: Implemented with Batch Normalization and Dropout layers for improved performance
- **Data Augmentation Pipeline**: Comprehensive augmentation including random flips and rotations
- **Advanced Training Callbacks**: EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau for optimal training
- **Visualization Tools**: Training history plots and augmented sample visualization
- **Preprocessing Pipeline**: Image normalization and dataset caching for efficient training
- **Model Evaluation**: Complete testing pipeline with confidence scoring


## Important Hardware Configuration Tip

**⚠️ Note: If you are not using Google Colab, your GPU processing settings may be different:**

- **In Google Colab**: Tesla T4 GPU or similar is used by default
- **On local**: Need to manually install CUDA and cuDNN drivers
- **Memory settings**: May need to reduce batch size on systems with less GPU memory

```python
# Check GPU support
import tensorflow as tf
print("Number of available GPUs: ", len(tf.config.list_physical_devices('GPU')))
```

## Project Structure

```
rock-paper-scissors-cnn/
├── model/
│   └── best_model_epoch.h5      # Pre-trained model weights
├── notebooks/
│   ├── rps-cnn-training-en.ipynb  # Complete training notebook (English)
│   └── rps-cnn-training-fa.ipynb  # Complete training notebook (Farsi)
├── dataset/
│   ├── paper/                   # Paper gesture images (≈730 images)
│   ├── rock/                    # Rock gesture images (≈730 images)
│   └── scissors/                # Scissors gesture images (≈730 images)
├── test_samples/                # Custom images for model testing
├── training_plots/
│   ├── augmentation_samples.png # Examples of augmented images
│   └── training_history.png     # Training/validation accuracy and loss plots
├── README.md                    # Project documentation
└── requirements.txt             # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sorna-fast/rock-paper-scissors-cnn.git
cd rock-paper-scissors-cnn
```

2. Create and activate a virtual environment:
```bash
python -m venv myvenv
# On Windows:
myvenv\Scripts\activate
# On macOS/Linux:
source myvenv/bin/activate
```

3. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

### Option 1: Run the Complete Training Pipeline

1. Launch Jupyter Notebook:
```bash
jupyter notebook
```

2. Open either notebook file:
   - `notebooks/rps-cnn-training-en.ipynb` (English version)
   - `notebooks/rps-cnn-training-fa.ipynb` (Farsi version)

3. Execute all cells sequentially to:
   - Load and preprocess the dataset
   - Visualize sample images
   - Apply data augmentation
   - Build and train the CNN model
   - Evaluate model performance
   - Make predictions on test images

### Option 2: Use the Pre-trained Model

1. Load the pre-trained model for predictions:
```python
from tensorflow.keras.models import load_model

model = load_model('model/best_model_epoch.h5')
```

2. Use the prediction function:
```python
# Example prediction
from utils.predict import predict_image

predicted_class, confidence = predict_image(
    model, 
    "rps_dataset/test_samples/your_image.png", 
    class_names=['paper', 'rock', 'scissors']
)
print(f"Predicted: {predicted_class} with {confidence:.2f}% confidence")
```

## Dataset

The project uses the Rock-Paper-Scissors dataset containing approximately 1885 images across three classes . The dataset is organized into separate folders for each class:

- **Paper**: Images showing hand paper gesture (flat hand)
- **Rock**: Images showing hand rock gesture (closed fist)
- **Scissors**: Images showing hand scissors gesture (two extended fingers)

All images are 300×300 pixels in PNG format with transparent backgrounds.

## Model Architecture

The CNN model consists of the following layers:

1.  **Data Augmentation Layer**: Applies random flips and rotations directly within the model.
2.  **Convolutional Layers**: 5 layers with increasing filters (32→256).
3.  **Batch Normalization**: Applied after each convolutional layer to stabilize and accelerate training.
4.  **MaxPooling Layers**: 3 layers for spatial dimensionality reduction.
5.  **Fully Connected Layers**: 2 dense layers with dropout regularization to prevent overfitting.
6.  **Output Layer**: Softmax activation for 3-class classification.

### Model Summary
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ data_augmentation (Sequential)  │ (None, 96, 96, 3)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_50 (Conv2D)              │ (None, 96, 96, 32)     │           896 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_50          │ (None, 96, 96, 32)     │           128 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_30 (MaxPooling2D) │ (None, 48, 48, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_51 (Conv2D)              │ (None, 48, 48, 256)    │        73,984 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_51          │ (None, 48, 48, 256)    │         1,024 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_31 (MaxPooling2D) │ (None, 24, 24, 256)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_52 (Conv2D)              │ (None, 24, 24, 384)    │       885,120 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_52          │ (None, 24, 24, 384)    │         1,536 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_53 (Conv2D)              │ (None, 24, 24, 384)    │     1,327,488 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_53          │ (None, 24, 24, 384)    │         1,536 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_54 (Conv2D)              │ (None, 24, 24, 256)    │       884,992 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_54          │ (None, 24, 24, 256)    │         1,024 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_32 (MaxPooling2D) │ (None, 12, 12, 256)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten_10 (Flatten)            │ (None, 36864)          │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_30 (Dense)                │ (None, 256)            │     9,437,440 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_20 (Dropout)            │ (None, 256)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_31 (Dense)                │ (None, 256)            │        65,792 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_21 (Dropout)            │ (None, 256)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_32 (Dense)                │ (None, 3)              │           771 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 12,681,731 (48.38 MB)
 Trainable params: 12,679,107 (48.37 MB)
 Non-trainable params: 2,624 (10.25 KB)
```

## Training Details

The model was trained with the following parameters:

- **Image Size**: 96×96 pixels
- **Batch Size**: 32
- **Data Augmentation**: Random horizontal/vertical flipping, rotation (±180°)
- **Optimizer**: Adam with default parameters
- **Callbacks**: 
  - EarlyStopping (patience=5, monitor val_accuracy)
  - ModelCheckpoint (save best model only)
  - ReduceLROnPlateau (factor=0.5, patience=2)
- **Validation Split**: 20% of training data

## Training Results

The model achieved excellent performance during training:

  - **Best Epoch**: 19
  - **Best Validation Accuracy**: 99.09%
  - **Training Accuracy at Best Epoch**: 99.06%
  - **Training Details**:
      - Training stopped after 20 epochs.
      - The learning rate was reduced at epoch 16 to `0.0005`.
      - Model weights were restored from the best epoch (19), which had the highest validation accuracy.


## Dependencies

The project requires the following Python packages (exact versions in requirements.txt):

- TensorFlow 2.20.0
- Keras 3.11.3
- OpenCV 4.12.0.88
- Matplotlib 3.10.6
- NumPy 2.2.6
- And other supporting packages

See `requirements.txt` for complete dependency list with exact versions.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

- **GitHub**: [sorna-fast](https://github.com/sorna-fast)
- **Email**: [masudpythongit@gmail.com](mailto:masudpythongit@gmail.com)
- **Telegram**: [@Masoud_Ghasemi_sorna_fast](https://t.me/Masoud_Ghasemi_sorna_fast)

## Acknowledgments

- TensorFlow and Keras teams for the excellent deep learning frameworks
- Contributors to the Rock-Paper-Scissors dataset
- Google Colab for providing computational resources

---

# توضیحات فارسی

## طبقه‌بندی تصاویر سنگ-کاغذ-قیچی با استفاده از شبکه عصبی کانولوشنی

این یک پروژه کامل یادگیری عمیق است که یک شبکه عصبی کانولوشنی (CNN) برای طبقه‌بندی تصاویر حرکات دست به دسته‌های سنگ، کاغذ و قیچی پیاده‌سازی کرده است. این پروژه از TensorFlow و Keras استفاده می‌کند.

## ویژگی‌ها

- **معماری CNN سفارشی**: پیاده‌سازی شده با لایه‌های Batch Normalization و Dropout برای بهبود عملکرد
- **پایپلاین افزایش داده**: افزایش داده جامع شامل چرخش و برعکس کردن تصادفی تصاویر
- **کالبک‌های پیشرفته آموزش**: EarlyStopping، ModelCheckpoint و ReduceLROnPlateau برای آموزش بهینه
- **ابزارهای مصورسازی**: نمودارهای تاریخچه آموزش و نمونه‌های افزایش داده
- **پایپلاین پیش‌پردازش**: نرمال‌سازی تصاویر و کش کردن دیتاست برای آموزش کارآمد
- **ارزیابی مدل**: پایپلاین کامل تست با امتیاز اطمینان
## نکته مهم پیکربندی سخت‌افزار

**⚠️ توجه: اگر از Google Colab استفاده نمی‌کنید، تنظیمات پردازش GPU ممکن است متفاوت باشد:**

- **در Google Colab**: به طور پیش‌فرض از GPU Tesla T4 یا مشابه استفاده می‌شود
- **در محیط محلی**: نیاز به نصب دستی درایورهای CUDA و cuDNN دارید
- **تنظیمات حافظه**: ممکن است نیاز به کاهش اندازه batch در سیستم‌های با حافظه GPU کمتر باشد

```python
# بررسی پشتیبانی از GPU
import tensorflow as tf
print("تعداد GPUهای در دسترس: ", len(tf.config.list_physical_devices('GPU')))
```


## ساختار پروژه

```
rock-paper-scissors-cnn/
├── model/
│   └── best_model_epoch.h5      # وزن‌های مدل از پیش آموزش‌دیده
├── notebooks/
│   ├── rps-cnn-training-en.ipynb  # نوتبوک کامل آموزش (انگلیسی)
│   └── rps-cnn-training-fa.ipynb  # نوتبوک کامل آموزش (فارسی)
├── dataset/
│   ├── paper/                   # تصاویر حرکت کاغذ (حدود ۷۳۰ تصویر)
│   ├── rock/                    # تصاویر حرکت سنگ (حدود ۷۳۰ تصویر)
│   └── scissors/                # تصاویر حرکت قیچی (حدود ۷۳۰ تصویر)
├── test_samples/                # تصاویر سفارشی برای تست مدل
├── training_plots/
│   ├── augmentation_samples.png # نمونه‌هایی از تصاویر افزایش‌یافته
│   └── training_history.png     # نمودارهای دقت و خطای آموزش و اعتبارسنجی
├── README.md                    # مستندات پروژه
└── requirements.txt             # وابستگی‌های پایتون
```

## نصب

1. کلون کردن مخزن:
```bash
git clone https://github.com/sorna-fast/rock-paper-scissors-cnn.git
cd rock-paper-scissors-cnn
```

2. ایجاد و فعال کردن محیط مجازی:
```bash
python -m venv myvenv
# در ویندوز:
myvenv\Scripts\activate
# در macOS/Linux:
source myvenv/bin/activate
```

3. نصب وابستگی‌ها:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## نحوه استفاده

### گزینه ۱: اجرای کامل پایپلاین آموزش

1. اجرای Jupyter Notebook:
```bash
jupyter notebook
```

2. باز کردن یکی از فایل‌های نوتبوک:
   - `notebooks/rps-cnn-training-en.ipynb` (نسخه انگلیسی)
   - `notebooks/rps-cnn-training-fa.ipynb` (نسخه فارسی)

3. اجرای سلول‌ها به ترتیب برای:
   - بارگذاری و پیش‌پردازش دیتاست
   - مصورسازی نمونه تصاویر
   - اعمال افزایش داده
   - ساخت و آموزش مدل CNN
   - ارزیابی عملکرد مدل
   - پیش‌بینی روی تصاویر تست

### گزینه ۲: استفاده از مدل از پیش آموزش‌دیده

1. بارگذاری مدل برای پیش‌بینی:
```python
from tensorflow.keras.models import load_model

model = load_model('model/best_model_epoch.h5')
```

2. استفاده از تابع پیش‌بینی:
```python
# نمونه پیش‌بینی
from utils.predict import predict_image

predicted_class, confidence = predict_image(
    model, 
    "rps_dataset/test_samples/your_image.png", 
    class_names=['paper', 'rock', 'scissors']
)
print(f"Predicted: {predicted_class} with {confidence:.2f}% confidence")
```

## دیتاست

این پروژه از دیتاست سنگ-کاغذ-قیچی استفاده می‌کند که شامل حدود 1885 تصویر در سه کلاس است . دیتاست در پوشه‌های جداگانه برای هر کلاس سازماندهی شده است:

- **کاغذ**: تصاویر حرکت کاغذ (دست باز)
- **سنگ**: تصاویر حرکت سنگ (مشت بسته)
- **قیچی**: تصاویر حرکت قیچی (دو انگشت باز)

همه تصاویر با ابعاد ۳۰۰×۳۰۰ پیکسل و با فرمت PNG و پس‌زمینه شفاف هستند.

## معماری مدل

مدل CNN از لایه‌های زیر تشکیل شده است:

1.  **لایه افزایش داده (Data Augmentation)**: چرخش و برعکس کردن تصادفی تصاویر را مستقیماً درون مدل اعمال می‌کند.
2.  **لایه‌های کانولوشنی**: ۵ لایه با فیلترهای افزایشی (۳۲→۲۵۶).
3.  **Batch Normalization**: بعد از هر لایه کانولوشنی برای پایدارسازی و تسریع آموزش.
4.  **لایه‌های MaxPooling**: ۳ لایه برای کاهش ابعاد فضایی.
5.  **لایه‌های Fully Connected**: ۲ لایه متراکم با رگولاریزیشن Dropout برای جلوگیری از بیش‌برازش.
6.  **لایه خروجی**: فعال‌سازی softmax برای طبقه‌بندی ۳ کلاس.

### خلاصه مدل
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ data_augmentation (Sequential)  │ (None, 96, 96, 3)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_50 (Conv2D)              │ (None, 96, 96, 32)     │           896 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_50          │ (None, 96, 96, 32)     │           128 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_30 (MaxPooling2D) │ (None, 48, 48, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_51 (Conv2D)              │ (None, 48, 48, 256)    │        73,984 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_51          │ (None, 48, 48, 256)    │         1,024 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_31 (MaxPooling2D) │ (None, 24, 24, 256)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_52 (Conv2D)              │ (None, 24, 24, 384)    │       885,120 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_52          │ (None, 24, 24, 384)    │         1,536 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_53 (Conv2D)              │ (None, 24, 24, 384)    │     1,327,488 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_53          │ (None, 24, 24, 384)    │         1,536 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_54 (Conv2D)              │ (None, 24, 24, 256)    │       884,992 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_54          │ (None, 24, 24, 256)    │         1,024 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_32 (MaxPooling2D) │ (None, 12, 12, 256)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten_10 (Flatten)            │ (None, 36864)          │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_30 (Dense)                │ (None, 256)            │     9,437,440 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_20 (Dropout)            │ (None, 256)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_31 (Dense)                │ (None, 256)            │        65,792 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_21 (Dropout)            │ (None, 256)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_32 (Dense)                │ (None, 3)              │           771 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 12,681,731 (48.38 MB)
 Trainable params: 12,679,107 (48.37 MB)
 Non-trainable params: 2,624 (10.25 KB)
```

## جزئیات آموزش

مدل با پارامترهای زیر آموزش داده شده است:

- **اندازه تصویر**: ۹۶×۹۶ پیکسل
- **اندازه بچ**: ۳۲
- **افزایش داده**: برعکس کردن افقی/عمودی تصادفی، چرخش (±۱۸۰ درجه)
- **بهینه‌ساز**: Adam با پارامترهای پیش‌فرض
- **کالبک‌ها**: 
  - EarlyStopping (patience=5, monitor val_accuracy)
  - ModelCheckpoint (ذخیره فقط بهترین مدل)
  - ReduceLROnPlateau (factor=0.5, patience=2)
- **تقسیم اعتبارسنجی**: ۲۰٪ از داده‌های آموزش


## نتایج آموزش

مدل در طول آموزش عملکرد عالی داشت:

  - **بهترین اپوک**: ۱۹
  - **بهترین دقت اعتبارسنجی**: ۹۹.۰۹٪
  - **دقت آموزش در بهترین اپوک**: ۹۹.۰۶٪
  - **جزئیات آموزش**:
      - آموزش پس از اپوک ۲۰ متوقف شد.
      - نرخ یادگیری در اپوک ۱۶ به `۰.۰۰۰۵` کاهش یافت.
      - وزن‌های مدل از بهترین اپوک (۱۹)، که بالاترین دقت اعتبارسنجی را داشت، بازیابی شدند.

## وابستگی‌ها

این پروژه به پکیج‌های پایتون زیر نیاز دارد (نسخه‌های دقیق در فایل requirements.txt):

- TensorFlow 2.20.0
- Keras 3.11.3
- OpenCV 4.12.0.88
- Matplotlib 3.10.6
- NumPy 2.2.6
- و سایر پکیج‌های پشتیبان

برای مشاهده لیست کامل وابستگی‌ها با نسخه‌های دقیق، فایل `requirements.txt` را ببینید.

## مجوز

این پروژه تحت مجوز MIT منتشر شده است. برای جزئیات بیشتر، فایل LICENSE را ببینید.

## تماس

- **GitHub**: [sorna-fast](https://github.com/sorna-fast)
- **ایمیل**: [masudpythongit@gmail.com](mailto:masudpythongit@gmail.com)
- **تلگرام**: [@Masoud_Ghasemi_sorna_fast](https://t.me/Masoud_Ghasemi_sorna_fast)

## تشکر و قدردانی

- از تیم‌های TensorFlow و Keras برای فریم‌ورک‌های عالی یادگیری عمیق
- از مشارکت‌کنندگان در دیتاست سنگ-کاغذ-قیچی
- از Google Colab برای فراهم کردن منابع محاسباتی