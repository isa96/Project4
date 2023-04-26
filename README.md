# Face Mask Detection 

## Introduction
Projek ini dikembangkan sebagai salah satu freelance project. Deliverables yang diharapkan adalah dapat membangun sebuah model deep learning. 

## Data Summary
Data yang digunakan pada freelance project ini adalah data gambar yang didapat dari salah satu developer. Rincian sebagai berikut:
- `with_mask` : Gambar wajah dengan masker              
- `without_mask` : Gambar wajah tidak memakai masker

## Dependencies
- Tensorflow
- Sklearn
- Matplotlib
- Numpy

## Rubrics
Pada project ini, Tujuannya adalah untuk membuat model yang mana dapat mendeteksi wajah menggunakan masker atau tidak. Beberapa proses yang harus diperhatikan adalah sebagai berikut:

### 1. Setting Repository Github dan Environment 
- Repository 

a. Membuat repository baru di Github

b. Clone repository tersebut ke local dengan git clone
- Environment 

a. Created virtual environment called "nama-project"

Hal pertama yang harus dilakukan adalah melakukan pengaturan environment conda. Untuk menyiapkan conda environment dan kernel, silahkan gunakan command berikut:
```
conda create -n <ENV_NAME> python=3.x
conda activate <ENV_NAME>

conda install ipykernel
python -m ipykernel install --user --name <ENV_NAME>
```

b. Install packages: tensorflow, sklearn, matplotlib, numpy
Untuk melakukan install packages dapat menggunakan perintah berikut:
```
pip install {nama package} --user
```

### 2. Data Preproses 
Pada tahap praproses ini yaitu memodifikasi nilai data, tujuan proses ini adalah memisahkan antara data gambar dan label dalam bentuk list. Pada label yang tadinya adalah tipe kategorikal di transformasi menjadi numeric agar mempermudah proses train nantinya. Dari tipe data list yang ada pada data gambar dan label dirubah menjadi data array matrix. Tahap akhir data array matrix dipisah menjadi 2 bagian yaitu data train dan test.

### 3. Architecture Model
Pada tahap architecture model ini adalah pembuatan alur yang nanti dilewati oleh data ketika sedang proses train. Disini architecture model yang digunakan adalah MobileNetV2.

### 4. Train Process 
- Data train yang sudah disiapkan sebelumnya di masukan dalam proses augmentasi. Tujuan augmentasi adalah model akan belajar dengan berbagai macam perspektif sesuai dengan parameter yang dicantumkan dalam alur proses augmentasi.
- Data train yang sudah di augmentasi dapat di proses langsung menggunakan architecture model yang sudah dibuat.

### 5. Evaluasi Model 
- Proses yang menggunakan data test sebagai validasi model, dimana untuk melihat model sudah berjalan dengan baik atau tidak. Adapun beberapa parameter yang digunakan untuk evaluasi adalah accuracy, precision, recall, dan f1score.

### 6. Test model 
Untuk menggunakan hasil model bisa menggunakan file `detect_mask_video.py`.
Untuk menjalankan file tersebut dapat menggunakan perintah berikut:
```
python detect_mask_video.py
```



