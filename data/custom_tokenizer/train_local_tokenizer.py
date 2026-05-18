# pip install tokenizers

import json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Sequence, Lowercase, StripAccents

def batch_iterator(jsonl_path, text_key="text", batch_size=1000):
    """
    Generator untuk membaca dataset.jsonl dalam bentuk batch (potongan kecil)
    agar hemat memori RAM saat membaca file berukuran gigabyte.
    """
    batch = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                if text_key in data and data[text_key].strip():
                    batch.append(data[text_key])
                
                if len(batch) == batch_size:
                    yield batch
                    batch = []
            except json.JSONDecodeError:
                continue  # Lewati baris jika ada json yang rusak
        if batch:
            yield batch

def main():
    # ---- 1. KONFIGURASI ----
    INPUT_JSONL = "dataset.jsonl"       # Nama file dataset Anda
    TEXT_COLUMN = "text"                # Nama kolom teks di dalam JSONL
    OUTPUT_JSON = "tokenizer.json"      # Nama file hasil akhir
    VOCAB_SIZE = 65536                  # Batas keras ukuran token untuk RWKV-v7
    
    print("Inisialisasi Tokenizer BPE Modern...")
    
    # ---- 2. DESAIN ARSITEKTUR TOKENIZER ----
    # Inisialisasi model dasar menggunakan algoritma BPE
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    
    # Normalisasi teks: Mengubah huruf menjadi kecil (lowercase) dan menghapus aksen aneh
    tokenizer.normalizer = Sequence([Lowercase(), StripAccents()])
    
    # Pre-tokenizer: Memotong teks berdasarkan spasi aturan dasar sebelum diproses BPE
    tokenizer.pre_tokenizer = Whitespace()
    
    # ---- 3. KONFIGURASI TRAINER ----
    # Menentukan token khusus yang wajib ada di dalam model RWKV/LLM
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=special_tokens,
        min_frequency=2,          # Kata/karakter harus muncul minimal 2 kali agar tidak mengotori vocab
        show_progress=True
    )
    
    # ---- 4. PROSES PELATIHAN (TRAINING) ----
    print(f"Memulai pemindaian data dari '{INPUT_JSONL}'...")
    data_iterator = batch_iterator(INPUT_JSONL, text_key=TEXT_COLUMN)
    
    tokenizer.train_from_iterator(data_iterator, trainer)
    
    # ---- 5. PENYIMPANAN ----
    tokenizer.save(OUTPUT_JSON)
    print("\n" + "="*50)
    print(f"SUKSES! Tokenizer baru berhasil dilatih.")
    print(f"File disimpan di: {OUTPUT_JSON}")
    print(f"Total ukuran kosakata: {tokenizer.get_vocab_size()} token (Batas maks: {VOCAB_SIZE})")
    print("="*50)

if __name__ == "__main__":
    main()
