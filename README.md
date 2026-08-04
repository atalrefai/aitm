# AI Trading Intelligence System (ATIS)

نظام تداول خوارزمي ذكي معياري — من جلب بيانات MetaTrader 5 إلى التدريب والتحقق والتنفيذ الحي (Paper / Demo)، مع تركيز أساسي على **الذهب XAUUSD**.

> **تنبيه:** النظام أداة احتمالية للبحث واتخاذ القرار. لا يضمن ربحاً. الوضع `live` برأس مال حقيقي **مرفوض** حتى موافقة صريحة منك. إدارة المخاطر إلزامية في كل أمر.

---

## جدول المحتويات

1. [نظرة عامة](#1-نظرة-عامة)
2. [المتطلبات](#2-المتطلبات)
3. [التثبيت والإعداد](#3-التثبيت-والإعداد)
4. [بنية المشروع](#4-بنية-المشروع)
5. [خط الأنابيب (المحركات 1–5)](#5-خط-الأنابيب-المحركات-15)
6. [الواجهة الويب (Gold Desk)](#6-الواجهة-الويب-gold-desk)
7. [التكوين](#7-التكوين)
8. [النماذج والمخرجات](#8-النماذج-والمخرجات)
9. [عرض الأنماط على MT5](#9-عرض-الأنماط-على-mt5)
10. [الاختبارات](#10-الاختبارات)
11. [ملاحظات تشغيل مهمة](#11-ملاحظات-تشغيل-مهمة)
12. [إخلاء المسؤولية](#12-إخلاء-المسؤولية)

---

## 1. نظرة عامة

### ماذا يفعل ATIS؟

| الطبقة | الوظيفة |
|--------|---------|
| **المحرك 1** | جلب شموع OHLCV من MT5 (تفاضلي + backfill) |
| **المحرك 2** | تنظيف البيانات، معالجة الفجوات، تعليم القيم الشاذة |
| **المحرك 3** | مؤشرات فنية + اكتشاف أنماط + ميزات علائقية |
| **المحرك 4** | تدريب Walk-Forward، حواجز ثلاثية، HPO مالي، ترويج Champion/Challenger، وحقن خبرات RL |
| **المحرك 5** | استدلال حي Paper/Demo مع إدارة مخاطر ومخارج ديناميكية وبوابة RL حيّة |
| **التعلّم الحي** | مخزن صفقات تدريب + قاعدة معرفة RL + طابور تغذية راجعة للتدريب |
| **الويب** | لوحة Gold Desk مربوطة بكل المحركات، مراقب RL، ومراكز MT5 الحية |

### الحالة الحالية

- [x] البنية التحتية المشتركة (`shared/`, `config/`, السجلات، السجل)
- [x] المحركات 1–3 لكل الأطر الزمنية المدعومة (عمق التاريخ يعتمد على الوسيط)
- [x] المحرك 4: Research Factory (v16+) مع بوابات جودة وترويج وميزات RL سياقية
- [x] المحرك 5: Paper/Demo + مخارج ديناميكية + نمط مستقل لكل إطار + بوابة سياسة RL
- [x] التركيز على **XAUUSD** في التداول الحي
- [x] واجهة ويب Gold Desk مع مراقبة مراكز حية وإغلاق تلقائي
- [x] تعلّم تعزيزي من نتائج الصفقات الحية مع حفظ الدروس وطابور تدريب
- [x] مخزن صفقات مغلقة للتدريب مع تدقيق شامل لكل الإغلاقات
- [x] عرض الأنماط على شارت MT5 عبر Expert Advisor

### المبادئ التصميمية

1. **عزل الأطر الزمنية:** كل timeframe خط أنابيب مستقل (مبدأ 1.2).
2. **Walk-Forward فقط:** لا تقسيم عشوائي للتدريب.
3. **تكاليف واقعية:** عمولة، سبريد، انزلاق، تأخير تنفيذ.
4. **الجودة أولاً:** بوابات overfitting، استقرار الطيات، PBO، stress، readiness.
5. **لا أمر بدون SL/TP.**
6. **الوضع الحقيقي محظور** حتى موافقة صريحة.

---

## 2. المتطلبات

| المتطلب | التفاصيل |
|---------|----------|
| نظام التشغيل | Windows (موصى به لـ MetaTrader 5) |
| Python | **≥ 3.11** |
| MetaTrader 5 | مثبت مع حساب Demo (مثل Windsor بلاحقة `@`) |
| حساب وسيط | بيانات دخول في `config/secrets.env` |
| ذاكرة/قرص | يفضل ≥ 8 GB RAM؛ بيانات M1 قد تكون كبيرة |

### الاعتماديات الرئيسية

- `pandas`, `numpy`, `pyarrow` — بيانات
- `MetaTrader5` — اتصال الوسيط
- `scikit-learn`, `lightgbm`, `torch`, `joblib` — نماذج
- `fastapi`, `uvicorn` — الواجهة الويب
- `pydantic`, `PyYAML`, `structlog` — إعداد وسجلات
- اختياري للتطوير: `pytest`, `pytest-cov`, `shap`

القائمة الكاملة في `requirements.txt` و `pyproject.toml`.

---

## 3. التثبيت والإعداد

### 3.1 استنساخ وتثبيت الحزمة

من جذر المشروع:

```bash
pip install -e ".[dev]"
```

أو عبر المتطلبات المباشرة:

```bash
pip install -r requirements.txt
pip install -e .
```

### 3.2 إعداد بيانات MT5

انسخ المثال ثم عبّئ بياناتك الحقيقية (الملف **لا يُرفع إلى git**):

```bash
copy config\secrets.env.example config\secrets.env
```

محتوى `config/secrets.env`:

```env
MT5_LOGIN=your_login
MT5_PASSWORD=your_password
MT5_SERVER=your_broker_server
# MT5_PATH=C:\Program Files\MetaTrader 5\terminal64.exe
```

### 3.3 التحقق من الاتصال

```bash
python -m atis.scripts.ping_mt5
```

### 3.4 تشغيل سريع للواجهة

```bash
python -m atis.web.run
# افتح http://127.0.0.1:8787
```

---

## 4. بنية المشروع

```
trading AI/
├── README.md
├── pyproject.toml
├── requirements.txt
├── config/
│   ├── engine_config.yaml      # الإعداد المركزي لكل المحركات
│   ├── symbols.yaml            # الرموز (افتراضياً XAUUSD)
│   ├── timeframes.yaml         # M1 … MN1
│   ├── indicators.yaml         # إعداد المؤشرات
│   ├── secrets.env.example
│   └── secrets.env             # محلي — لا يُرفع
├── atis/
│   ├── config/                 # تحميل YAML + MT5Settings
│   ├── engines/
│   │   ├── engine1_ingestion/
│   │   ├── engine2_cleaning/
│   │   ├── engine3_features/
│   │   ├── engine4_training/   # Research Factory + تشخيص
│   │   └── engine5_live_trading/
│   ├── shared/
│   │   ├── mt5_client/
│   │   ├── data_registry/
│   │   ├── feature_engine/
│   │   ├── pattern_discovery/
│   │   ├── pattern_store/
│   │   ├── pattern_kb/
│   │   ├── rl_learning/            # قاعدة معرفة المكافآت/العقوبات
│   │   ├── winning_trade_store/    # صفقات التدريب المفتوحة/المغلقة
│   │   ├── mt5_pattern_overlay/
│   │   ├── logging_utils/
│   │   └── …
│   ├── scripts/
│   │   ├── ping_mt5.py
│   │   └── run_pipeline_1_3.py
│   └── web/                    # Gold Desk (FastAPI + static + live position watcher)
├── data/
│   ├── raw/                    # شموع خام
│   ├── clean/                  # بعد التنظيف
│   ├── features/               # ميزات جاهزة للتدريب
│   ├── patterns/
│   ├── registry/               # حالة السلاسل والسجل
│   ├── rl_knowledge/           # حلقات RL، الدروس، الحالة، طابور التدريب
│   └── training_trades/        # صفقات مفتوحة + JSONL التدريب + سجل إغلاقات
├── models/
│   ├── XAUUSD/{TF}/            # تجارب + champion / shadow
│   ├── FinalModel/             # نموذج نهائي اختياري
│   └── intelligence/           # research_factory, advisories
├── logs/
│   ├── ingestion/ cleaning/ features/ training/ live/
│   └── live/mt5_overlay/
├── mql5/Experts/               # EA لرسم الأنماط على الشارت
├── scripts/
└── tests/unit/
```

---

## 5. خط الأنابيب (المحركات 1–5)

### مخطط التدفق

```
MT5 ──► Engine 1 (raw) ──► Engine 2 (clean) ──► Engine 3 (features)
                                                      │
                                                      ▼
                                              Engine 4 (train / gates)
                                                      │
                                                      ▼
                         models/{symbol}/{TF}/champion ──► Engine 5 (paper/demo)
                                                      │
                                                      ▼
                                         logs + MT5 overlay + Web UI
```

### 5.1 تشغيل المحركات 1–3 دفعة واحدة

```bash
python -m atis.scripts.run_pipeline_1_3 --symbols XAUUSD --timeframes M15 H1 H4
```

خيارات مفيدة:

| الخيار | المعنى |
|--------|--------|
| `--force-rebuild` | إعادة بناء كاملة بدل التحديث التفاضلي |
| `--skip-ingest` | تخطي المحرك 1 واستخدام البيانات الخام الموجودة |

---

### المحرك 1 — جلب البيانات (Ingestion)

**الغرض:** الاتصال بـ MT5 وجلب OHLCV بشكل تفاضلي مع backfill محدود.

**المدخلات:** `config/symbols.yaml`, `timeframes.yaml`, `engine_config.yaml`, `secrets.env`, سجل الحالة.

**المخرجات:**

- `data/raw/{symbol}/{timeframe}/{symbol}_{timeframe}.parquet`
- تحديثات `data/registry/`
- `logs/ingestion/ingestion_run_report.json`

```bash
python -m atis.engines.engine1_ingestion.run
python -m atis.engines.engine1_ingestion.run --symbols XAUUSD --timeframes H1
python -m atis.engines.engine1_ingestion.run --symbols XAUUSD --timeframes H1 --force-rebuild
```

**ملاحظات:**

- رموز بعض وسطاء Demo تستخدم لاحقة `@` (مثل `XAUUSD@`) وتُحل تلقائياً عبر `trading.broker_symbol_map`.
- عمق التاريخ يعتمد على الوسيط (غالباً أقل من 3 سنوات على M1/M5).

---

### المحرك 2 — تنظيف البيانات (Cleaning)

**الغرض:** تحويل `data/raw` → `data/clean` مع معالجة الفجوات، تعليم الشواذ، توحيد UTC، ومعالجة تفاضلية.

**المخرجات:**

- `data/clean/{symbol}/{timeframe}/…parquet` (يشمل `is_imputed`, `is_outlier`)
- `logs/cleaning/data_quality_report.json`

```bash
python -m atis.engines.engine2_cleaning.run --symbols XAUUSD --timeframes H1
python -m atis.engines.engine2_cleaning.run --symbols XAUUSD --timeframes H1 --force-rebuild
```

إعدادات بارزة في `engine2_cleaning`: استراتيجية الملء (`linear`)، طريقة الشواذ (`iqr`)، وحد أقصى لفجوات الملء.

---

### المحرك 3 — الميزات والأنماط (Features)

**الغرض:** بناء ميزات المؤشرات والأنماط عبر `shared/feature_engine` (نفس المسار المستخدم في التداول الحي).

**المخرجات:**

- `data/features/{symbol}/{timeframe}/features.parquet`
- سجلات في `logs/features/`

```bash
python -m atis.engines.engine3_features.run --symbols XAUUSD --timeframes H1 --force-rebuild
```

يشمل مؤشرات فنية، أنماط شموع/هيكل، وميزات علاقات الأنماط عند التفعيل في التدريب.

---

### المحرك 4 — التدريب والتحقق (Research Factory)

**الغرض:** تدريب نماذج تصنيف/تداول بـ **Walk-Forward** فقط، تسمية **Triple Barrier**، تكاليف واقعية، بوابات جودة، ومقارنة Champion / Challenger / Shadow.

**إصدار الخط تقريباً:** `e4-v16.0-research-factory` وما بعده (تشخيص ذاتي، HPO مالي، factory board).

```bash
python -m atis.engines.engine4_training.run --symbols XAUUSD --timeframes H1
python -m atis.engines.engine4_training.run --symbols XAUUSD --timeframes M15 M30 H1 H4
```

#### ما يفعله المحرك 4 (باختصار)

1. بوابة جودة البيانات (DQ gate)
2. اختيار ميزات مستقرة عبر الطيات
3. Sweep لحساسية الحواجز (عند التفعيل)
4. تنظيف ضوضاء التسميات
5. Nested HPO + Model Zoo (هدف مالي مثل expectancy مقابل التكلفة)
6. تقييم طيات Walk-Forward + holdouts (أزمة / حديث)
7. Stress (سبريد، ضوضاء، فجوات، latency)
8. بوابات: overfitting، استقرار الطيات، PBO، expectancy، readiness
9. حقن ميزات `feat_rl_*` من حلقات التعلّم الحي وإعادة وزن التسميات
10. استهلاك طابور خبرات RL المحفوظة بعد كل دورة تدريب
11. تسجيل Champion أو Shadow Challenger
12. تقارير: metrics، backtest، diagnosis، enterprise dossier، SHAP/explainability
13. Research factory board + نصائح إعادة التدريب (drift)

#### مخرجات نموذجية

| المسار | المحتوى |
|--------|---------|
| `models/{symbol}/{TF}/{run_id}/` | النموذج، التقارير، metadata |
| `models/{symbol}/{TF}/champion.json` | البطل الحالي |
| `models/{symbol}/{TF}/shadow_challenger.json` | المنافس الظلي |
| `models/intelligence/research_factory.json` | لوحة الأبحاث |
| `models/FinalModel/` | نموذج نهائي عند التفعيل |
| `logs/training/training_run_report.json` | ملخص تشغيل |
| `data/rl_knowledge/training_queue.jsonl` | خبرات حيّة محفوظة ستُستهلك في التدريب التالي |

تفاصيل إضافية: `atis/engines/engine4_training/README.md`.

---

### المحرك 5 — التداول الحي Paper / Demo

**الغرض:** استدلال على أحدث البيانات، قرار اتجاهي مع عتبات ثقة، وإدارة مخاطر إلزامية. لا أوامر بدون SL/TP.

```bash
# استدلال فقط (بدون أوامر)
python -m atis.engines.engine5_live_trading.run --symbols XAUUSD --timeframe H1

# أوامر على حساب Demo
python -m atis.engines.engine5_live_trading.run --symbols XAUUSD --timeframe H1 --execute-demo

# حلقة مستمرة
python -m atis.engines.engine5_live_trading.run --symbols XAUUSD --timeframe H1 --loop --interval 60 --max-iterations 5

# رفض النماذج غير المروّجة (champion gated)
python -m atis.engines.engine5_live_trading.run --symbols XAUUSD --timeframe H1 --require-champion
```

#### إدارة المخاطر (إلزامية)

من `engine5_live` في `config/engine_config.yaml`:

- حد خسارة يومي / أسبوعي
- حد أقصى للمراكز المفتوحة والتعرض
- حجم الصفقة حسب الثقة (`confidence_sizing`)
- فلتر سبريد اختياري
- **Kill switch:** `engine5_live.kill_switch: true`

#### المخارج الديناميكية (`dynamic_exits`)

عند التفعيل (الافتراضي):

- SL/TP قريبان من الدخول ومبنيان على العائد المتوقع، الثقة، درجة المخاطر، دعم/مقاومة محلي (~2×ATR)، وATR الحي
- ليست مستويات تاريخية بعيدة ولا إزاحات ثابتة بالـ pips
- لتعطيلها: `dynamic_exits.enabled: false` (عودة لمضاعفات ATR الثابتة)

#### التعلم من نتائج التداول الحي

المحرك 5 صار يرسل كل صفقة مغلقة إلى طبقتين:

1. **`winning_trade_store`** لحفظ سجل الصفقات المفتوحة ثم تثبيت الإغلاق النهائي عند تأكد PnL من الوسيط.
2. **`rl_learning`** لحساب مكافأة/عقوبة مركبة تعتمد على:
   - نتيجة الصفقة الفعلية (PnL)
   - جودة قرار الدخول
   - `RR` المخطط والمحقق
   - التزام قواعد التنفيذ والفلترة

هذا ينتج:

- `data/training_trades/open_trades.json`
- `data/training_trades/winning_trades.jsonl`
- `data/training_trades/closed_trades_audit.jsonl`
- `data/rl_knowledge/episodes.jsonl`
- `data/rl_knowledge/learning_timeline.jsonl`
- `data/rl_knowledge/lessons.json`
- `data/rl_knowledge/rl_state.json`
- `data/rl_knowledge/training_queue.jsonl`

#### بوابة RL الحية

عند تفعيل `engine5_live.reinforcement_learning.live_policy_gate_enabled`، يقرأ ATIS أوزان السياسة المتعلمة من الحلقات السابقة، ويمكنه حظر قرار حي إذا كان متوسط الإشارة أدنى من `live_policy_block_threshold`.

#### تعدد الأطر (AutoTrader)

| الإعداد | السلوك |
|---------|--------|
| `multi_tf_independent: true` (افتراضي) | كل TF يقرر ويستدخل بشكل مستقل |
| `multi_tf_fusion: true` | دمج تصويت إلى أمر واحد (سلوك قديم) |
| `max_open_positions_per_tf` | سقف مراكز ATIS لكل إطار عبر تعليق الصفقة `ATIS\|H1\|…` |

التداول الحي مقصور على الرموز في `trading.allowed_live_symbols` (حالياً **XAUUSD** فقط).

تفاصيل إضافية: `atis/engines/engine5_live_trading/README.md`.

---

## 6. الواجهة الويب (Gold Desk)

```bash
python -m atis.web.run
# http://127.0.0.1:8787
```

الإعداد الافتراضي في `config/engine_config.yaml`:

```yaml
web:
  host: 127.0.0.1
  port: 8787
  title: ATIS Gold Desk
```

### ماذا تغطي الواجهة؟

- حالة اتصال MT5 والسجل (Registry)
- تشغيل/متابعة المحركات 1–5
- التقارير والسجلات والنماذج
- الصفقات والقرارات الحية
- مراقب التعلم التعزيزي: حلقات، دروس، حالات حفظ المعرفة، وطابور التدريب
- مراقبة حيّة للمراكز المفتوحة عبر خيط مستقل + بث SSE للواجهة
- إغلاق يدوي لمركز واحد أو جماعي (`all` / `winners` / `losers`)
- إغلاق تلقائي للربح أو وقف خسارة مصغّر من الواجهة
- إيقاف الطوارئ (Kill switch)
- مساعدة تدريب وتفسيرات الجودة

الواجهة محلية بشكل افتراضي (`127.0.0.1`) — لا تفتحها على شبكة عامة دون حماية.

---

## 7. التكوين

الملف المركزي: **`config/engine_config.yaml`**.

### أقسام رئيسية

| القسم | الدور |
|-------|-------|
| `project` | الاسم، الإصدار، المنطقة الزمنية، البذرة |
| `paths` | مسارات البيانات والنماذج والسجلات |
| `trading` | الرمز الأساسي، خريطة الوسيط، الرموز المسموحة حياً |
| `engine1_ingestion` | backfill، إعادة المحاولة، الرموز/الأطر الافتراضية |
| `engine2_cleaning` | الفجوات، الشواذ، المنطقة الزمنية |
| `engine3_features` | ملف المؤشرات وlookback |
| `engine4_training` | الحواجز، Walk-Forward، LightGBM، البوابات، HPO، stress |
| `engine5_live` | Paper/Demo، المخاطر، المخارج، multi-TF، overlay، RL، winning store |
| `web` | المضيف والمنفذ |
| `logging` | المستوى ومجلد السجلات |

### ملفات مساندة

- `config/symbols.yaml` — قائمة الرموز
- `config/timeframes.yaml` — تعريف كل إطار وجدولة
- `config/indicators.yaml` — إعدادات المؤشرات الفنية
- `config/secrets.env` — بيانات MT5 فقط

### الأطر الزمنية المدعومة

`M1`, `M5`, `M15`, `M30`, `H1`, `H4`, `D1`, `W1`, `MN1`

كل إطار يُعالَج كخط مستقل (بيانات، ميزات، نموذج، قرار).

### أمثلة إعداد شائعة

**إيقاف التداول فوراً:**

```yaml
engine5_live:
  kill_switch: true
```

**تفضيل النموذج النهائي بدل LLModel:**

```yaml
engine5_live:
  prefer_final_model: true
  prefer_llmodel: false
```

**تقليل صرامة بوابة معيّنة أثناء البحث (بحذر):** عدّل الأعلام مثل `fail_on_*` في `engine4_training` — لا تعطّلها للتداول الحي دون فهم الأثر.

**تفعيل/ضبط التعلم التعزيزي الحي:**

```yaml
engine5_live:
  reinforcement_learning:
    enabled: true
    queue_saved_for_training: true
    live_policy_gate_enabled: true
    live_policy_block_threshold: -0.2
```

**تفعيل تخزين الصفقات المغلقة للتدريب:**

```yaml
engine5_live:
  winning_trade_store:
    enabled: true
    winners_only: false
    include_losses: true
    audit_all_closes: true
```

---

## 8. النماذج والمخرجات

### هيكل نموذج مدرب

```
models/XAUUSD/H1/
├── champion.json
├── shadow_challenger.json
├── knowledge_loop.json
└── 20260803T095925Z_xxxxx/
    ├── model.joblib
    ├── metadata.json
    ├── feature_list.json
    ├── metrics_report.json
    ├── backtest_report.json
    ├── diagnosis.json
    ├── evaluation_report.md
    ├── enterprise_dossier.md
    ├── feature_explainability.json
    ├── smart_recommendations.json
    └── training_config.yaml
```

### سجلات التشغيل

| المسار | المحتوى |
|--------|---------|
| `logs/atis.jsonl` | سجل عام منظم |
| `logs/ingestion/` | تقارير الجلب |
| `logs/cleaning/` | جودة التنظيف |
| `logs/features/` | بناء الميزات |
| `logs/training/` | ملخص التدريب |
| `logs/live/` | قرارات، صفقات، تقرير الحي |
| `logs/live/mt5_overlay/` | حالة وأنماط الرسم على الشارت |
| `data/rl_knowledge/` | قاعدة معرفة المكافآت، الحالة، الخط الزمني، الدروس |
| `data/training_trades/` | الصفقات المفتوحة، صفقات التدريب، سجل الإغلاقات |

---

## 9. عرض الأنماط على MT5

واجهة Python لـ MT5 لا ترسم كائنات الشارت. ATIS يكتب حالة الأنماط إلى ملفات، وExpert Advisor يرسمها.

### التثبيت

1. انسخ `mql5/Experts/ATIS_PatternOverlay.mq5` إلى مجلد Experts في MT5.
2. جمّع في MetaEditor (F7).
3. أرفق الـ EA على شارت XAUUSD (نفس الإطار الذي تتداول عليه).
4. فعّل **Allow Algo Trading**.
5. المدخلات المقترحة: `InpUseCommonFiles=true`, `InpPollMs=250`.

### تدفق البيانات

```
Engine 5 / AutoTrader
  → اكتشاف أنماط من الميزات الحية
  → كتابة overlay_state.json
       ├─ المشروع: logs/live/mt5_overlay/
       └─ MT5: Common\Files\ATIS\
  → EA يقرأ الملف ويرسم الأسهم/الخطوط/المستطيلات
```

التفاصيل والألوان: `mql5/Experts/README.md`.

تفعيل الطبقة من الإعداد:

```yaml
engine5_live:
  pattern_overlay:
    enabled: true
    lookback_bars: 8
    max_patterns: 40
```

---

## 10. الاختبارات

```bash
pytest
pytest tests/unit -q
pytest tests/unit/test_engine4_quality_first.py -q
pytest tests/unit/test_rl_learning.py -q
```

تغطي الاختبارات الوحدوية: السجل، الميزات، التنظيف، بوابات المحرك 4، المخارج الديناميكية، نمط overlay، وكذلك اختبارات `RL` الخاصة بتقييم الصفقات، تخزين الحلقات، استهلاك طابور التدريب، وإصلاح الحالات التاريخية.

---

## 11. ملاحظات تشغيل مهمة

1. **الرمز الحي:** مقصور على `XAUUSD` عبر `allowed_live_symbols`.
2. **لاحقة الوسيط:** خريطة مثل `XAUUSD → XAUUSD@` في `trading.broker_symbol_map`.
3. **عمق التاريخ:** الوسيط يحدد كم من M1/M5 متاح؛ لا تفترض 3 سنوات كاملة.
4. **الوضع `live` الحقيقي:** مرفوض في الكود حتى موافقة صريحة.
5. **لا أمر بدون SL/TP** — المخارج الديناميكية أو مضاعفات ATR.
6. **عزل الأطر:** تدريب وقرار كل TF مستقلان ما لم تفعّل الدمج (`multi_tf_fusion`).
7. **الأسرار:** لا ترفع `config/secrets.env` أو مفاتيح الحساب.
8. **الجودة قبل الكمية:** نماذج تفشل البوابات قد تُرفض أو تُبقى في Shadow — راجع `diagnosis.json` و`readiness`.
9. **إعادة التدريب:** يمكن طلبها عبر إعدادات الحي (`retrain_interval_days` / advisory الانجراف).
10. **التعلم الحي لا يعني تعلماً فورياً في النموذج:** خبرات RL تُحفظ أولاً ثم تُستهلك في دورة تدريب Engine4 التالية.
11. **إغلاقات الوسيط الخارجية:** إذا أُغلقت الصفقة عبر SL/TP أو خارج الواجهة، سيحاول `position_watcher` تسويتها من سجل الصفقات في MT5.
12. **النظام احتمالي:** نتائج الماضي (حتى OOS) لا تضمن الأداء المستقبلي.

---

## 12. إخلاء المسؤولية

- ATIS للبحث والتداول التجريبي (Paper/Demo) على مسؤوليتك.
- الأسواق المالية عالية المخاطر؛ قد تخسر رأس المال.
- لا يُعد هذا البرنامج نصيحة استثمارية أو ضمان أداء.
- راجع تقارير التشخيص والـ readiness قبل أي تفعيل Demo، ولا تنتقل لرأس مال حقيقي دون مراجعة صارمة وموافقة صريحة في التصميم.

---

## مرجع أوامر سريع

```bash
# إعداد
pip install -e ".[dev]"
python -m atis.scripts.ping_mt5

# بيانات → ميزات
python -m atis.scripts.run_pipeline_1_3 --symbols XAUUSD --timeframes M15 H1 H4

# تدريب
python -m atis.engines.engine4_training.run --symbols XAUUSD --timeframes H1

# حي (paper ثم demo)
python -m atis.engines.engine5_live_trading.run --symbols XAUUSD --timeframe H1
python -m atis.engines.engine5_live_trading.run --symbols XAUUSD --timeframe H1 --execute-demo --loop

# واجهة
python -m atis.web.run

# اختبارات RL
pytest tests/unit/test_rl_learning.py -q

# اختبارات
pytest -q
```

---

**الإصدار:** انظر `project.version` في `config/engine_config.yaml` وحقل `version` في `pyproject.toml`.  
**التركيز الحالي:** ذهب XAUUSD — ATIS Gold Desk.
