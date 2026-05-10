# Coding Single Perceptron dengan Python Keras

**Nama Ketua Kelompok :** Ririn Yuliati
**NIM :** 231001004

**Nama Anggota Kelompok :** Rizqikah Amaliyah Syahfa
**NIM :** 231001010

---

## Deskripsi

Proyek ini merupakan implementasi **Single Perceptron** menggunakan Python dan library **Keras (TensorFlow)**. Terdapat dua studi kasus:

1. **Logika AND 2 Input** — Perceptron belajar mengenali pola gerbang logika AND.
2. **Regresi Linear 2 Variabel** — Perceptron belajar menemukan bobot dari persamaan linear `y = w1*x1 + w2*x2`.

---

## Teknologi yang Digunakan

| Library | Kegunaan |
|---|---|
| `keras` | Membangun dan melatih model neural network |
| `numpy` | Pengelolaan array dan data numerik |
| `TensorFlow` | Backend untuk Keras |

---

## Studi Kasus 1 — Logika AND

### Tabel Data Latih

| x1 | x2 | y |
|----|----|---|
| 1  | 1  | 1 |
| 1  | 0  | 0 |
| 0  | 1  | 0 |
| 0  | 0  | 0 |

Output `y = 1` **hanya** jika kedua input bernilai 1.

### Kode & Penjelasan

#### 1. Import Library
```python
import keras
import numpy as np
```
- `keras` — library untuk membangun neural network
- `numpy` — library untuk operasi array numerik

---

#### 2. Membuat Model
```python
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[2])])
```
- `Sequential` — model dengan susunan layer berurutan
- `Dense(units=1)` — satu neuron output (fully connected)
- `input_shape=[2]` — menerima 2 input (x1 dan x2)

Ini adalah arsitektur **Single Perceptron**: 1 neuron dengan 2 bobot (`w1`, `w2`) + 1 bias.

---

#### 3. Kompilasi Model
```python
model.compile(optimizer='sgd', loss='mean_squared_error')
```
- `optimizer='sgd'` — *Stochastic Gradient Descent*, memperbarui bobot setiap iterasi untuk meminimalkan error
- `loss='mean_squared_error'` — fungsi loss: rata-rata kuadrat selisih antara prediksi dan nilai sebenarnya

---

#### 4. Menyiapkan Data
```python
xs = np.array([[1, 1], [1, 0], [0, 1], [0, 0]], dtype=int)
ys = np.array([1, 0, 0, 0], dtype=int)
```
- `xs` — array 4 baris × 2 kolom berisi pasangan input (x1, x2)
- `ys` — array output/label yang sesuai dengan tabel logika AND

---

#### 5. Menampilkan Arsitektur Model
```python
model.summary()
```
Output:
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense (Dense)                   │ (None, 1)              │             3 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
```
- **Output Shape `(None, 1)`** — None berarti jumlah data bisa berapa saja, output 1 nilai
- **Param # 3** — terdiri dari 2 bobot (w1, w2) + 1 bias

---

#### 6. Cek Bobot Awal
```python
weights = model.get_weights()
weights
```
Menampilkan nilai bobot dan bias **sebelum** training (diinisialisasi secara acak oleh Keras).

---

#### 7. Training Model
```python
model.fit(xs, ys, epochs=1000)
```
- `xs, ys` — data input dan output latih
- `epochs=1000` — model belajar dari data sebanyak **1000 kali** putaran penuh

Setiap epoch, bobot diperbarui menggunakan SGD agar nilai loss semakin kecil.

---

#### 8. Prediksi / Testing
```python
data = [1, 1]
answer = model.predict(np.array([data]))
print(answer)
```
- Menguji model dengan input `[1, 1]`
- Hasil prediksi seharusnya mendekati `1.0` (sesuai tabel AND: `1 AND 1 = 1`)

---

#### 9. Cek Bobot Akhir
```python
weights = model.get_weights()
weights
```
Menampilkan nilai bobot dan bias **setelah** training. Bobot sudah disesuaikan agar model dapat memprediksi logika AND dengan benar.

---

## Studi Kasus 2 — Regresi Linear 2 Variabel

### Persamaan Target
```
y = w1*x1 + w2*x2   →   w1 = 2, w2 = 4
```

### Tabel Data Latih

| x1 | x2 | y  |
|----|----|----|
| 2  | 3  | 16 |
| 4  | 1  | 12 |
| 5  | 4  | 28 |
| 7  | 5  | 34 |
| 8  | 2  | 24 |
| 2  | 1  | 8  |
| 4  | 9  | 44 |
| 8  | 2  | 24 |
| 7  | 1  | 18 |
| 6  | 5  | 32 |
| 1  | 1  | 6  |
| 3  | 2  | 14 |

### Kode & Penjelasan

#### 1. Membuat Model Kedua
```python
model2 = keras.Sequential([keras.layers.Dense(units=1, input_shape=[2])])
model2.compile(optimizer='sgd', loss='mean_squared_error')
```
Arsitektur dan kompilasi sama seperti model pertama.

---

#### 2. Menyiapkan Data
```python
xs = np.array([[2,3],[4,1],[5,4],[7,5],[8,2],[2,1],[4,9],[8,2],[7,1],[6,5],[1,1],[3,2]])
ys = np.array([16, 12, 28, 34, 24, 8, 44, 24, 18, 32, 6, 14], dtype=int)
```
- `xs` — 12 pasangan nilai (x1, x2)
- `ys` — hasil dari `2*x1 + 4*x2` untuk setiap pasangan

---

#### 3. Cek Bobot Awal
```python
weights = model2.get_weights()
weights
```
Menampilkan bobot acak sebelum training dimulai.

---

#### 4. Training Model
```python
model2.fit(xs, ys, epochs=1000)
```
Model belajar dari 12 data selama 1000 epoch. Tujuannya: menemukan bobot `w1 ≈ 2` dan `w2 ≈ 4`.

---

#### 5. Prediksi / Testing
```python
data = [1, 3]
answer = model2.predict(np.array([data]))
print(answer)
# seharusnya 1*2 + 3*4 = 14
```
- Menguji dengan input `[1, 3]`
- Nilai yang diharapkan: `1×2 + 3×4 = 2 + 12 = 14`
- Jika training berhasil, prediksi model akan mendekati `14.0`

---

#### 6. Cek Bobot Akhir
```python
weights = model2.get_weights()
weights
```
Menampilkan bobot akhir setelah training. Nilai bobot diharapkan mendekati `[2, 4]` dengan bias mendekati `0`.

---

## Alur Kerja Keseluruhan

```
Data Latih (xs, ys)
        ↓
  Buat Model Sequential
  (Dense 1 unit, 2 input)
        ↓
  Kompilasi (SGD + MSE)
        ↓
  Tampilkan Arsitektur & Bobot Awal
        ↓
  Training (1000 epoch)
        ↓
  Prediksi Data Baru
        ↓
  Tampilkan Bobot Akhir
```

---

## Konsep Utama

| Konsep | Penjelasan |
|---|---|
| **Single Perceptron** | Unit dasar neural network: menerima beberapa input, menghasilkan 1 output |
| **Bobot (Weight)** | Nilai yang menentukan seberapa besar pengaruh setiap input terhadap output |
| **Bias** | Nilai tambahan agar model lebih fleksibel dalam belajar |
| **SGD** | Algoritma optimasi yang memperbarui bobot secara bertahap untuk mengurangi error |
| **MSE** | Fungsi loss: `rata-rata dari (y_prediksi - y_asli)²` |
| **Epoch** | Satu kali iterasi penuh melalui seluruh data latih |
| **`model.fit()`** | Fungsi untuk melatih model dengan data |
| **`model.predict()`** | Fungsi untuk menghasilkan prediksi dari data baru |

---

## Cara Menjalankan

1. Install dependensi:
   ```bash
   pip install tensorflow numpy
   ```

2. Buka notebook:
   ```bash
   jupyter notebook kode_ml_vidio_4.ipynb
   ```

3. Jalankan setiap sel secara berurutan dari atas ke bawah.
