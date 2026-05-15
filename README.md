<div align="center">

# RWKV-LM-V7 (Google Colab & Drive Edition)

</div>

## Pengantar

Repositori ini merupakan *fork* dari [RWKV-Vibe/RWKV-LM-V7](https://github.com/RWKV-Vibe/RWKV-LM-V7). Semua kode dasar bersumber dari proyek orisinal RWKV-LM: https://github.com/BlinkDL/RWKV-LM.

**Tujuan Repositori Ini:**
Fokus utama dari *fork* ini adalah untuk mempermudah pembuatan, pelatihan, dan eksperimen model **RWKV-v7** menggunakan **Google Colab**. 

Karena Google Colab bersifat sementara (hilang saat di- *restart*), repositori ini dirancang dengan alur kerja khusus yang memanfaatkan **Google Drive** sebagai penyimpanan persisten. Kami mengemas *Conda Environment* (Python 3.12) dan menyimpannya di Google Drive agar Anda tidak perlu mengulang proses instalasi yang memakan waktu setiap kali memulai sesi Colab baru.

## Cara Memulai di Google Colab

Untuk memulai, pastikan Anda sudah memiliki akun Google dan ruang kosong di Google Drive Anda.

1. Buka file notebook `.ipynb` yang tersedia di repositori ini (misalnya `setup_rwkv_v7.ipynb`).
2. Klik tombol **Open in Colab**.
3. Jalankan sel di dalam notebook secara berurutan. Notebook tersebut akan secara otomatis:
   - Menginstal `condacolab`.
   - Melakukan *mounting* ke Google Drive Anda.
   - **Logika Cerdas:** Mengecek apakah *backup environment* sudah ada di Drive Anda. Jika belum, sistem akan membangun dari awal dan menyimpannya (`.tar.gz`) ke Drive. Jika sudah ada, sistem hanya akan mengekstraknya dalam 1-2 menit.

## Kredit & Referensi

- Proyek Asli RWKV-LM: [https://github.com/BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM).
- Base Fork: [RWKV-Vibe/RWKV-LM-V7](https://github.com/RWKV-Vibe/RWKV-LM-V7).
