# Customer Typology — Uzcard mijozlar segmentatsiyasi (Homework 6)

Mijozlarni **xulq-atvori va moliyaviy holati** bo'yicha tipologiyaga ajratish bo'yicha
unsupervised loyiha. Clustering `customer_id`, eski `crm_segment`, `turnover_total` va
`net_revenue_uzs` kabi label/leykaj ustunlarisiz quriladi; daromad faqat **klasterdan keyingi
tashqi tekshiruv** uchun ishlatiladi.

Asosiy savol: *mijozlar bazasini shunchaki oborot bo'yicha kvartilga bo'lish o'rniga, xulq
bo'yicha haqiqiy, barqaror va biznes uchun tushunarli segmentlarga ajratish mumkinmi?*

---

## 1. Ma'lumotlar

| Jadval | Qatorlar | Izoh |
|---|---:|---|
| `seg_customers` | 20 000 | mijoz profili (jins, yosh, hudud, ro'yxatdan o'tish sanasi) |
| `seg_transactions` | 12 136 132 | 2024-01-01 → 2025-12-28, kanal (pos/atm/p2p), MCC, summa |
| `seg_balances` | 427 598 | oylik balanslar |
| `seg_products` | 29 983 | mijoz mahsulotlari |
| `seg_revenue_monthly` | 427 598 | oylik sof daromad (`net_revenue_uzs`) |

Tekshirilgan xulosalar:

- `seg_transactions_archive` — `seg_transactions` ning **to'liq dublikati** (2 718 367 qator,
  `customer_id + txn_ts + amount_uzs` bo'yicha 1:1 mos). Ishlatilmaydi.
- `seg_customers_360` va `seg_rfm_old` — tayyor eski segmentatsiya/labellar, ya'ni leykaj.
  Clustering kirishida ishlatilmaydi.
- `seg_temp` — bo'sh jadval.

## 2. Metodologiya (yakuniy pipeline)

1. **Kuzatuv oynasi** — `TRAIN_END = 2025-09-30`, feature'lar 2025-04-01 → 2025-09-30 oralig'idan;
   `FUTURE_START..FUTURE_END = 2025-10-01 → 2025-12-31` faqat tashqi validatsiya uchun.
2. **Leykaj auditi** — identifikator, tayyor biznes label, post-window natija va kelajak
   ma'lumoti bilan hisoblangan ustunlar dasturiy ravishda chiqarib tashlanadi.
3. **Feature bloklari** — activity/recency, frequency, monetary, financial (balans), diversity
   (kanal/MCC entropy, mahsulot soni), tenure. Har bir latent tushunchadan bitta vakil ustun.
4. **Redundancy kontroli** — |Spearman| > 0.95 juftliklar aniqlanadi, bloklangan og'irlik bilan
   masshtablanadi (bitta signal ikki marta "ovoz bermaydi").
5. **Ekstremal mijozlar** — yuqori kvantildagi 271 mijoz `outlier_review` sifatida alohida
   ushlanadi; ular oddiy klaster geometriyasini buzmaydi.
6. **K tanlash** — K = 2..8 uchun inertia, silhouette (sample), Calinski-Harabasz,
   Davies-Bouldin, klaster balansi solishtiriladi. K = 4 tanlandi: K = 2 eng yuqori silhouette
   beradi, lekin 93/7 kabi keskin nomutanosib va biznes uchun bitta "hammasi" klasteriga aylanadi.
7. **Barqarorlik** — turli `random_state`, yarim tanlanma, feature blokini olib tashlash va
   vaqt oynasini surish bo'yicha ARI.
8. **Out-of-time validatsiya** — klasterlar faqat o'tmish bilan qurilgan, keyingi 3 oy
   (faollik, xarajat, sof daromad) ustida tekshiriladi. Kruskal-Wallis: p < 0.001.

## 3. Natijalar (K = 4)

| Klaster | Nom (biznes) | Mijoz | Ulush | Median profil | Keyingi 3 oy |
|---|---|---:|---:|---|---|
| `cluster_3` | Barqaror o'rta segment | 13 926 | 69.6% | oyiga 14 op, faol oyda ~3.5 mln so'm, balans ~1.1 mln, staj 41 oy | 99.9% faol, median sof daromad 86 ming |
| `cluster_0` | Premium faol | 4 082 | 20.4% | oyiga 30 op, faol oyda ~18 mln so'm, balans ~9.6 mln | 100% faol, median xarajat 56.4 mln, sof daromad 634 ming |
| `cluster_2` | Uxlab qolgan / past faollik | 877 | 4.4% | oxirgi op 35 kun oldin, oyiga 1 op, 1 kanal, MCC entropiyasi past | faqat 52% faol, sof daromad 5.8 ming |
| `cluster_1` | Yangi mijozlar | 844 | 4.2% | staj 1.9 oy, oyiga 16 op, ~5.3 mln so'm | 100% faol, sof daromad 146 ming |
| `outlier_review` | VIP / ekstremal qiymat (alohida) | 271 | 1.4% | balans ~27.8 mln, o'rtacha xarajat keskin yuqori | sof daromad median 1.59 mln |

Asosiy xulosalar:

- Klasterlar **kvartil bo'linishi emas**: har bir klaster bir nechta kvartilga tarqaladi, ya'ni
  xulq haqida qo'shimcha ma'lumot beradi.
- Segmentlar keyingi davrdagi xulqni **sezilarli ajratadi** (`future_active_rate`, xarajat va
  sof daromad bo'yicha p < 0.001) — tipologiya amaliy ahamiyatga ega.
- Barqarorlik yuqori: seed bo'yicha ARI ≈ 1.00, yarim tanlanmada 0.97–0.99. Feature blokini
  olib tashlash ARI'ni 0.37–0.45 ga tushiradi — ya'ni har bir blok haqiqatan hissa qo'shadi.
- `outlier_review` oddiy persona sifatida taqdim etilmaydi: ular alohida biznes ko'rib chiqish
  (RM) oqimiga yuboriladi.

## 4. Repo tuzilishi

```
.
├── Homework6_compact_parquet_first.ipynb   # yakuniy, toza pipeline (parquet-first, tavsiya etiladi)
├── Homework6_corrected_segmentation.ipynb  # baseline + leykajga qarshi tuzatilgan qism
├── Homework6.ipynb                         # dastlabki baseline, batafsil izohlar bilan
├── customer_typology_presentation.pdf      # taqdimot (natijalar va tavsiyalar)
├── archive/                                # eski qoralama nusxalar (tarix uchun)
├── seg_*.parquet                           # ma'lumotlar (kichik jadvallar)
├── requirements.txt
└── README.md
```

> **Ma'lumot haqida:** `seg_transactions.parquet` (203 MB) va `seg_transactions_archive.parquet`
> (45 MB) GitHub'ning 100 MB fayl limitidan katta, shuning uchun repoga qo'shilmagan
> (`.gitignore`). Notebook'ni ishga tushirish uchun ularni shu papkaga qo'ying; qolgan jadvallar
> repoda mavjud.

## 5. Qanday ishga tushirish

```bash
pip install -r requirements.txt
jupyter lab Homework6_compact_parquet_first.ipynb
```

Notebook parquet fayllarni o'z papkasidan (yoki ichki papkalaridan) qidiradi, shuning uchun
qo'shimcha sozlash kerak emas. `pathlib`, `numpy`, `pandas`, `pyarrow`, `scikit-learn`, `scipy`,
`matplotlib`, `seaborn` ishlatiladi.

Bazaga ulanish kerak bo'lsa, DSN faqat muhit o'zgaruvchisidan olinadi — kodda parol saqlanmaydi:

```bash
export SEG_DB_DSN="postgresql://user:password@host:port/db"   # Windows: set SEG_DB_DSN=...
```

`SEG_DB_DSN` berilmasa, SQL cell'lari ma'lumotni lokal parquet fayllardan olishni taklif qilib
xato beradi. `.env` fayllari ham repoga qo'shilmaydi.

## 6. Cheklovlar va talqin qoidalari

- Klaster o'rtachasi guruhni tavsiflaydi, **sababni isbotlamaydi**.
- Yuqori classifier accuracy faqat o'sha feature fazosida yaratilgan labelni qayta tiklaydi —
  bu segmentatsiya sifati yoki kelajak daromadining aniqligi emas.
- Katta klaster avtomatik "yomon" emas, lekin ichki geterogenlik tekshirilishi shart.
- Demografiya (yosh, jins, hudud) avval faqat profil uchun ishlatiladi; taqdimotda ular
  segmentatsiyani yaxshilashi isbotlanmaguncha kirishga qo'shilmaydi.
- Natijalar tarixiy oynaga bog'liq: vaqt oynasi surilsa, klaster chegaralari ham siljiydi.
