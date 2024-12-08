

```
violet/
├── __init__.py                        # ملف التهيئة لجعل المجلد حزمة قابلة للاستيراد
├── configuration/                     # مجلد يحتوي على ملفات إعدادات النموذج
│   ├── __init__.py                    # تعريف الوحدة الفرعية
│   └── configuration_violet.py        # ملف تكوين الحزمة
├── tokenizer/                         # مجلد يحتوي على ملفات الترميز
│   ├── __init__.py                    # تعريف الوحدة الفرعية
│   └── tokenization_violet.py         # ملف مسؤول عن الترميز وفك الترميز للنصوص
├── processing/                        # مجلد معالجة الصور والنصوص
│   ├── __init__.py                    # تعريف الوحدة الفرعية
│   └── processing_violet.py           # ملف مسؤول عن معالجة الصور والنصوص
├── modeling/                          # مجلد يحتوي على بنية النموذج
│   ├── __init__.py                    # تعريف الوحدة الفرعية
│   ├── modeling_violet.py             # ملف يحتوي على نموذج Violet الأساسي
│   ├── attention.py                   # وحدة الانتباه متعددة الرؤوس
│   ├── encoder.py                     # وحدة التشفير للنموذج
│   ├── decoder.py                     # وحدة فك التشفير للنموذج
│   └── configuration.py               # إعدادات النموذج
├── training/                          # مجلد يحتوي على ملفات التدريب والتقييم
│   ├── __init__.py                    # تعريف الوحدة الفرعية
│   ├── trainer_violet.py              # ملف مسؤول عن تدريب وتقييم النموذج
│   ├── evaluation.py                  # ملف مسؤول عن تقييم النموذج
│   └── metrics.py                     # تعريف المقاييس المستخدمة في التقييم
├── pipeline/                          # واجهة الاستخدام والتنبؤ
│   ├── __init__.py                    # تعريف الوحدة الفرعية
│   └── pipeline_violet.py             # ملف يتيح واجهة مبسطة للتفاعل مع الحزمة
├── utils/                             # الأدوات المساعدة
│   ├── __init__.py                    # تعريف الوحدة الفرعية
│   └── utils_violet.py                # وظائف مساعدة عامة
├── tests/                             # اختبارات الوحدة
│   ├── __init__.py                    # تعريف الوحدة الفرعية
│   ├── test_model.py                  # اختبارات للنموذج
│   ├── test_tokenizer.py              # اختبارات للمعالج النصي
│   ├── test_pipeline.py               # اختبارات لواجهة التفاعل (Pipeline)
│   ├── test_processing.py             # اختبارات لوحدة المعالجة
│   └── test_training.py               # اختبارات لوحدة التدريب
├── data/                              # مجلد بيانات
│   ├── __init__.py                    # تعريف الوحدة الفرعية
│   ├── dataset_violet.py              # تعريف هيكل البيانات المستخدم في الحزمة
│   └── dataloader_violet.py           # أداة لتحميل البيانات على دفعات
├── resources/                         # موارد إضافية (مثل ملفات التدريب)
│   ├── pretrained/                    # النماذج المدربة مسبقاً
│   └── annotations/                   # ملفات البيانات التوضيحية
├── saved_models/                      # مجلد لحفظ النماذج أثناء وبعد التدريب
│   └── ...                            # نماذج محفوظة
├── docs/                              # وثائق الحزمة
│   ├── api_reference.md               # توثيق واجهة برمجة التطبيقات (API)
│   ├── usage_examples.md              # أمثلة الاستخدام
│   └── installation_guide.md          # دليل التثبيت
├── examples/                          # أمثلة لاستخدام الحزمة
│   ├── train_example.py               # مثال لتدريب النموذج
│   ├── evaluate_example.py            # مثال لتقييم النموذج
│   └── inference_example.py           # مثال لاستخدام النموذج للتنبؤ
├── tools/                             # أدوات إضافية
│   ├── evaluate.py                    # ملف لتقييم النموذج
│   └── train.py                       # ملف لتدريب النموذج
├── setup.py                           # ملف إعداد الحزمة للتثبيت
├── requirements.txt                   # ملف يحتوي على مكتبات بايثون المطلوبة
├── README.md                          # شرح تفصيلي للحزمة
├── LICENSE                            # ملف الترخيص
└── .gitignore                         # استثناء الملفات من النسخة (Git)

```

