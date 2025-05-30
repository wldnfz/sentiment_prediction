from flask import Flask, request, jsonify, send_from_directory
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary

import librosa
import pandas as pd
import pickle
import re

stop_words = StopWordRemoverFactory().get_stop_words()
new_array = ArrayDictionary(stop_words)
stop_words_remover_new = StopWordRemover(new_array)

# Initialize Flask app
app = Flask(__name__)

# Load model & vectorizer
with open('artifacts/model.pkl', 'rb') as f:
    clf = pickle.load(f)
with open('artifacts/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

norm = {
    'bsa': 'bisa',
    'gk': 'tidak',
    'ga': 'tidak',
    'nggak': 'tidak',
    'ngak': 'tidak',
    'gak': 'tidak',
    'tdk': 'tidak',
    'udh': 'sudah',
    'udh': 'sudah',
    'sdh': 'sudah',
    'blm': 'belum',
    'aja': 'saja',
    'bgt': 'banget',
    'authentificator': 'authenticator',
    'ny': 'nya',
    'banget': 'sekali',
    'bener': 'benar',
    'dr': 'dari',
    'krn': 'karena',
    'tp': 'tapi',
    'kl': 'kalau',
    'klo': 'kalau',
    'dpt': 'dapat',
    'trus': 'terus',
    'trs': 'terus',
    'mf': 'maaf',
    'sbnrnya': 'sebenarnya',
    'daftar': 'registrasi',
    'regis': 'registrasi',
    'akun': 'account',
    'log': 'login',
    'pw': 'password',
    'otp': 'otp',
    'ota': 'otp',
    'nik': 'nomor induk kependudukan',
    'hp': 'handphone',
    'emailnya': 'email',
    'akunny': 'akun',
    'gaada': 'tidak ada',
    'gaada': 'tidak ada',
    'ngetiknya': 'mengetik',
    'mohon': 'harap',
    'nya': '',
    'yg': 'yang',
    'mau': 'ingin',
    'mnt': 'menit',
    'mt': 'maintenance',
    'ttd': 'tanda tangan digital',
    'tandatangan': 'tanda tangan',
    'tte': 'tanda tangan elektronik',
    'mempermudah': 'memudahkan',
    'membantu': 'menolong',
    'banget': 'sekali',
    'ok': 'baik',
    'oke': 'baik',
    'buat': 'membuat',
    'bnyak': 'banyak',
    'gitu': 'begitu',
    'mksdnya': 'maksudnya',
    'sblmnya': 'sebelumnya',
    'dtg': 'datang',
    'sgt': 'sangat',
    'punyq': 'punya',
    'sejenis': 'jenis',
    'khususnya': 'terutama',
    'blm': 'belum',
    'udah': 'sudah',
    'nyoba': 'mencoba',
    'coba': 'mencoba',
    'maaf': 'mohon maaf',
    'min': 'admin',
    'app': 'aplikasi',
    'aplikasonya': 'aplikasinya',
    'aplikasinya': 'aplikasi',
    'diupdate': 'diperbarui',
    'versi': 'versi',
    'ke': 'ke',
    'lumayan': 'cukup',
    'bgs': 'bagus',
    'sekaligus': 'sekaligus',
    'dmn': 'dimana',
    'lbh': 'lebih',
    'jd': 'jadi',
    'smoga': 'semoga',
    'bgtu': 'begitu',
    'spt': 'seperti',
    'sya': 'saya',
    'sy': 'saya',
    'gausah': 'tidak usah',
    'malah': 'bahkan',
    'dlm': 'dalam',
    'mrk': 'mereka',
    'udh': 'sudah',
    'aja': 'saja',
    'smg': 'semoga',
    'gimana': 'bagaimana',
    'dan lain lain': '',
    'dll': '',
    'lah': '',
    'dong': '',
    'deh': '',
    'mah': '',
    'kok': '',
    'tok': 'saja',
    'punya': 'memiliki'
}

def normalisasi(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    corrected_words = [norm.get(word, word) for word in words]
    text = " ".join(corrected_words)
    text = stop_words_remover_new.remove(text)
    return text

def predict_review(text):
    # preprocessing
    tokens = normalisasi(text)
    # Transform ke vektor TF-IDF
    vec = vectorizer.transform([tokens])
    # Prediksi sentimen
    pred = clf.predict(vec)
    return pred[0]

# Handle styling and javascript file
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('public', path)

# Homepage
@app.route('/')
def index():
    return send_from_directory('public', 'index.html')

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict_sentiment():
    try:
        prediction = ''
        if request.method == 'POST':
            review = request.form['review']
            prepro = normalisasi(review)
            vec = vectorizer.transform([prepro])
            prediction = clf.predict(vec)[0]

        return jsonify({'sentiment': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)