# pip install transformers numpy

import fileinput
import json
import os
import random
import sys

import numpy as np

# Menghubungkan jalur internal repositori RWKV
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Menggunakan pustaka Hugging Face untuk memuat tokenizer.json lokal
from transformers import PreTrainedTokenizerFast  # noqa: E402
from src.binidx import MMapIndexedDataset  # noqa: E402

"""
Cara Penggunaan di Terminal:
python make_data.py <nama_dataset>.jsonl <jumlah_epoch> <panjang_konteks>

Contoh:
python make_data.py dataset.jsonl 1 4096
"""

########################################################################################################
# 1. INISIALISASI TOKENIZER LOKAL
########################################################################################################
TOKENIZER_PATH = "tokenizer.json"  # Sesuaikan jika file Anda berada di folder lain

if not os.path.exists(TOKENIZER_PATH):
    print(f"ERROR: File '{TOKENIZER_PATH}' tidak ditemukan!")
    print("Pastikan Anda sudah menjalankan skrip pelatihan tokenizer terlebih dahulu.")
    sys.exit(1)

print(f"### Memuat Tokenizer Lokal dari '{TOKENIZER_PATH}'...")
tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH)
print(f"### Ukuran Kosakata Tokenizer: {tokenizer.vocab_size} token")

if tokenizer.vocab_size > 65536:
    print("\nWARNING" * 5)
    print(f"Ukuran vocab Anda ({tokenizer.vocab_size}) melebihi batas maksimal uint16 (65536)!")
    print("Hal ini akan menyebabkan eror saat pembentukan file biner. Mohon batasi vocab ke 65536.")
    sys.exit(1)

########################################################################################################
# 2. STRUKTUR DATA MMAP INDEXED DATASET BUILDER
########################################################################################################
def index_file_path(prefix_path):
    return prefix_path + ".idx"


def data_file_path(prefix_path):
    return prefix_path + ".bin"


class MMapIndexedDatasetBuilder(object):
    def __init__(self, out_file, dtype=np.uint16):
        self._data_file = open(out_file, "wb")
        self._dtype = dtype
        self._sizes = []
        self._doc_idx = [0]

    def add_item(self, np_array):
        assert np_array.dtype == self._dtype
        self._data_file.write(np_array.tobytes(order="C"))
        self._sizes.append(np_array.size)

    def end_document(self):
        self._doc_idx.append(len(self._sizes))

    def finalize(self, index_file):
        self._data_file.close()
        with MMapIndexedDataset.Index.writer(index_file, self._dtype) as index:
            index.write(self._sizes, self._doc_idx)


cnt = 0


def add_raw(raw):
    global builder, cnt
    # Menggunakan pengodean tekstual dari tokenizer.json Anda
    out = tokenizer.encode(raw)
    
    # [0] didefinisikan secara keras sebagai end_of_doc (EOS) oleh arsitektur RWKV
    out.append(0)  
    
    builder.add_item(np.array(out, dtype=np.uint16))
    builder.end_document()
    if cnt % 500 == 0:
        print(cnt, end=" ", flush=True)
    cnt += 1


def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

########################################################################################################
# 3. PROSES DATASET & PEMBUATAN EPOCH PERULANGAN
########################################################################################################
if len(sys.argv) < 4:
    print("ERROR: Argumen kurang lengkap!")
    print("Format: python make_data.py <file.jsonl> <epoch> <ctx_len>")
    sys.exit(1)

N_EPOCH = int(sys.argv[2].strip())
IN_FILE = sys.argv[1].strip()
OUT_NAME = os.path.splitext(os.path.basename(IN_FILE))[0]
CTX_LEN = int(sys.argv[3].strip())
TEMP_FILE = "make_data_temp.jsonl"

print(f"### Mengonversi {IN_FILE} menjadi {OUT_NAME}.bin/.idx...")

with open(IN_FILE, "r", encoding="utf-8") as file:
    non_empty_lines = [line.strip() for line in file if line.strip()]

print(f"### Berhasil mendeteksi {len(non_empty_lines)} baris teks di dalam {IN_FILE}")

# Membuat replikasi data dan pengacakan sesuai jumlah epoch yang ditargetkan
file = open(TEMP_FILE, "w", encoding="utf-8")
for i in range(N_EPOCH):
    print(f"Proses Pengacakan (Shuffle): {i+1} dari {N_EPOCH}")
    random.shuffle(non_empty_lines)
    for entry in non_empty_lines:
        file.write(entry + "\n")
file.close()

########################################################################################################
# 4. EKSEKUSI PEMBENTUKAN FILE BINER (.BIN/.IDX)
########################################################################################################
print("### Membangun berkas kompresi binidx...")

builder = MMapIndexedDatasetBuilder(f"{OUT_NAME}.bin")
with fileinput.input(TEMP_FILE, encoding="utf-8") as ffff:
    for line in ffff:
        try:
            x = json.loads(line)["text"]
            add_raw(x)
        except (json.JSONDecodeError, KeyError):
            continue  # Lewati baris jika format jsonl rusak atau kolom 'text' tidak ada
            
builder.finalize((f"{OUT_NAME}.idx"))
print("selesai")

# Menghapus file sampah penampung sementara setelah konversi sukses
if os.path.exists(TEMP_FILE):
    os.remove(TEMP_FILE)

########################################################################################################
# 5. VERIFIKASI AKHIR & PENENTUAN PARAMETER MAGIC_PRIME RWKV
########################################################################################################
print("### Memulai proses verifikasi integritas data...")
data = MMapIndexedDataset(OUT_NAME)
data_len = len(data)
data_size = len(data._bin_buffer) // data._index._dtype_size

TODO = [0, data_len - 1] if data_len > 1 else [0]
PREVIEW_LIMIT = 100

for idx in TODO:
    ptr, size = data._index[idx]
    dix = data.get(idx=idx, offset=0, length=size).astype(int)
    print("-" * 70 + f" [{OUT_NAME} indeks ke-{idx} ukuran {size} token]")
    assert dix[-1] == 0
    dix = dix[:-1]
    if len(dix) > PREVIEW_LIMIT:
        try:
            print(tokenizer.decode(dix[:PREVIEW_LIMIT]))
        except BaseException:
            print(tokenizer.decode(dix[:PREVIEW_LIMIT + 1], skip_special_tokens=True))
        print(" · " * 15)
        try:
            print(tokenizer.decode(dix[-PREVIEW_LIMIT:]))
        except BaseException:
            print(tokenizer.decode(dix[-PREVIEW_LIMIT - 1:], skip_special_tokens=True))
    else:
        print(tokenizer.decode(dix))

print(f"{'-'*80}\n### BERHASIL! Berkas final {OUT_NAME}.bin/.idx memiliki total {data_size} token dari {data_len} dokumen.")
print(f"### Tipe Data Biner: {data._index.dtype}")

# Menghitung parameter magic_prime wajib untuk peluncuran training RWKV
if data_size >= CTX_LEN * 3:
    n_chunk = int(data_size // CTX_LEN) - 1
    for i in range(n_chunk, 0, -1):
        if i % 3 == 2:
            if is_prime(i):
                print(f"\n### PARAMETER TRAINING ANDA UNTUK DI-COPY:")
                print(f'--my_exit_tokens {data_size} --magic_prime {i} --ctx_len {CTX_LEN}\n')
                exit(0)
else:
    print("\n[PERINGATAN]: Jumlah total token terlalu sedikit untuk panjang konteks ini.")
    print("Tambahkan isi data pada berkas jsonl Anda agar parameter kalkulasi multi-GPU aman.")
