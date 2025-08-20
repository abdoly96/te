# YOLOv8 Vision Lab - نظام إدارة البيانات المحسن

## نظرة عامة

تم تطوير هذا النظام ليعمل بشكل متكامل مع YOLOv8، حيث يقوم بإدارة البيانات والتصنيفات بشكل تلقائي ويحفظها في تنسيق YOLO القياسي.

## الميزات الجديدة

### 🎯 إدارة البيانات التلقائية
- **حفظ تلقائي**: يتم حفظ الصور المصنفة تلقائياً في `dataset/train/images/`
- **إنشاء الليبلز**: يتم إنشاء ملفات YOLO labels في `dataset/train/labels/`
- **تحديث data.yaml**: يتم تحديث ملف البيانات تلقائياً مع الفئات الجديدة
- **تجاهل الصور غير المصنفة**: لا يتم نقل الصور التي لا تحتوي على تصنيفات

### 📁 بنية المجلدات
```
dataset/
├── train/
│   ├── images/          # صور التدريب
│   └── labels/          # ملفات التصنيف YOLO
├── val/
│   ├── images/          # صور التحقق (اختيارية)
│   └── labels/          # ملفات التصنيف للتحقق
└── data.yaml            # ملف تكوين البيانات
```

### 🔧 الوظائف الرئيسية

#### 1. حفظ التصنيفات
- حفظ صورة واحدة مع تصنيفاتها
- حفظ مجموعة من الصور في عملية واحدة
- التحقق من صحة التصنيفات قبل الحفظ

#### 2. إدارة الفئات
- إضافة فئات جديدة تلقائياً
- تعيين معرفات فريدة لكل فئة
- تحديث `data.yaml` تلقائياً

#### 3. التحقق من البيانات
- التحقق من تكامل Dataset
- كشف الملفات المفقودة
- تنظيف الملفات الميتة

#### 4. التكامل مع التدريب
- التحقق من صحة Dataset قبل بدء التدريب
- استخدام البيانات مباشرة من مجلدات Dataset
- لا حاجة لإعادة رفع البيانات

## كيفية الاستخدام

### 1. رفع الصور
```javascript
// رفع صورة واحدة أو مجموعة صور
uploadImagesForAutoDetection()

// أو رفع ملف ZIP يحتوي على صور
// سيتم استخراجه تلقائياً
```

### 2. التصنيف
```javascript
// التصنيف اليدوي
// رسم صناديق الإحاطة حول الكائنات

// التصنيف التلقائي
startAutoDetection()
```

### 3. حفظ في Dataset
```javascript
// حفظ جميع التصنيفات في Dataset
saveToDataset()

// سيتم:
// - نسخ الصور إلى dataset/train/images/
// - إنشاء ملفات .txt في dataset/train/labels/
// - تحديث data.yaml
```

### 4. بدء التدريب
```javascript
// النظام سيتحقق تلقائياً من صحة Dataset
// ثم يبدأ التدريب باستخدام البيانات المحفوظة
```

## API Endpoints

### معلومات Dataset
```http
GET /api/dataset/info
```

### التحقق من Dataset
```http
GET /api/dataset/validate
```

### تنظيف Dataset
```http
POST /api/dataset/cleanup
```

### حفظ التصنيفات
```http
POST /api/annotations/save
POST /api/annotations/save_batch
```

### الكشف التلقائي
```http
POST /api/annotations/auto_detect
```

## تنسيق البيانات

### ملفات YOLO Labels
```
class_id x_center y_center width height
```

مثال:
```
0 0.5 0.5 0.3 0.4
1 0.7 0.3 0.2 0.3
```

### ملف data.yaml
```yaml
path: /path/to/dataset
train: train/images
val: val/images
nc: 2
names:
  - palm_tree
  - other_tree
```

## متطلبات النظام

```bash
pip install -r requirements.txt
```

### المكتبات المطلوبة
- Flask==2.3.3
- PyYAML==6.0.1
- opencv-python==4.8.1.78
- ultralytics==8.0.196
- torch==2.0.1

## تشغيل النظام

```bash
python main.py
```

أو

```bash
python app.py
```

## استكشاف الأخطاء

### مشاكل شائعة

1. **خطأ في data.yaml**
   - تأكد من وجود مجلد `dataset/`
   - تحقق من صحة بنية المجلدات

2. **صور بدون ليبلز**
   - استخدم `validateDataset()` للكشف عن المشاكل
   - استخدم `cleanupDataset()` لتنظيف الملفات الميتة

3. **فشل في التدريب**
   - تحقق من وجود صور في `dataset/train/images/`
   - تحقق من وجود ليبلز في `dataset/train/labels/`

## المساهمة

لإضافة ميزات جديدة أو إصلاح الأخطاء:
1. قم بتطوير الميزة في فرع منفصل
2. تأكد من اختبار جميع الوظائف
3. أرسل Pull Request مع وصف واضح للتغييرات

## الترخيص

هذا المشروع مرخص تحت رخصة MIT.
