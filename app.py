# ============================
# KONFIGURASI DAN PERSIAPAN
# ============================

# Import library yang dibutuhkan
import pandas as pd
import matplotlib.pyplot as plt
import base64
import pickle
import os
import uuid
import io
from flask import Flask, render_template, request, send_file, redirect, url_for, session
from utils.preprocessing import full_preprocessing # Fungsi untuk preprocessing teks
from wordcloud import WordCloud
from io import BytesIO

# Inisialisasi Flask app
app = Flask(__name__)
app.secret_key = "s3cr3t_@spekSentimen2025" # Kunci rahasia untuk session

# Simpan progress dummy berdasarkan session
progress_data = {}

# Load model dan vectorizer hanya sekali saat aplikasi dijalankan
with open("model/model9_svm_rbf_nosmote.pkl", "rb") as f:
    loaded_model = pickle.load(f)

vectorizer = loaded_model["vectorizer"]
models = loaded_model["models"]

# ============================
# ROUTES
# ============================

# Halaman Utama
@app.route('/')
def index():
    return render_template('index.html')

# Halaman prediksi aspek dan sentimen dari teks/csv
@app.route('/predict-aspect-sentiment', methods=["GET", "POST"])
def predict_aspect_sentiment():
    if request.method == "POST":
        input_text = request.form.get("input_text", "").strip()
        csv_file   = request.files.get("csv_file")

        # CASE 1: Prediksi dari input teks
        if input_text:
            processed = full_preprocessing(input_text)
            X_new     = vectorizer.transform([processed]).toarray()

            label_map   = {1: "Positif", 2: "Negatif"}
            predictions = {}

            # Lakukan prediksi untuk setiap aspek
            for label_name, model in models.items():
                y_pred = model.predict(X_new)[0]
                if y_pred in label_map:
                    predictions[label_name] = label_map[y_pred]
            
            # Simpan hasil ke session
            session["hasil_prediksi"] = predictions
            session["original"] = input_text

            return redirect(url_for("predict_aspect_sentiment"))

        # CASE 2: Prediksi dari file CSV
        elif csv_file and csv_file.filename.endswith('.csv'):
            try:
                session_id = str(uuid.uuid4())
                progress_data[session_id] = 0 # Dummy untuk tracking progres

                df = pd.read_csv(csv_file)
                if 'review' not in df.columns:
                    raise ValueError("File CSV harus memiliki kolom 'review'.")

                label_map = {1: "Positif", 2: "Negatif"}
                aspek_list = list(models.keys())
                chart_data = {}
                pie_chart_data = {}

                # Prediksi setiap review untuk tiap aspek
                for label_name, model in models.items():
                    df[label_name] = df['review'].apply(
                        lambda x: label_map.get(
                            model.predict(
                              vectorizer.transform([full_preprocessing(str(x))]).toarray()
                            )[0],
                            "-"
                        )
                    )
                
                # Siapkan data chart (bar dan pie)
                for aspek in aspek_list:
                    count_positif = (df[aspek] == "Positif").sum()
                    count_negatif = (df[aspek] == "Negatif").sum()
                    total = count_positif + count_negatif

                    percent_positif = (count_positif / total * 100) if total else 0
                    percent_negatif = (count_negatif / total * 100) if total else 0

                    # Buat deskripsi untuk setiap aspek
                    # description = f"Dari {total} data, terdapat {count_positif} sentimen positif ({percent_positif:.1f}%) dan {count_negatif} sentimen negatif ({percent_negatif:.1f}%)"
                    dominant = "positif" if count_positif >= count_negatif else "negatif"
                    # Buat deskripsi untuk setiap aspek
                    if dominant == "positif":
                        description = (
                            f"Dari total {total} ulasan, mayoritas menunjukkan sentimen positif "
                            f"sebanyak {count_positif} ({percent_positif:.1f}%), dibandingkan {count_negatif} "
                            f"({percent_negatif:.1f}%) yang negatif."
                        )
                    else:
                        description = (
                            f"Dari total {total} ulasan, mayoritas menunjukkan sentimen negatif "
                            f"sebanyak {count_negatif} ({percent_negatif:.1f}%), dibandingkan {count_positif} "
                            f"({percent_positif:.1f}%) yang positif."
                        )

                    chart_data[aspek] = {
                        "Positif": int(count_positif),
                        "Negatif": int(count_negatif)
                    }
                    pie_chart_data[aspek] = {
                        "labels": ["Positif", "Negatif"],
                        "values": [int(count_positif), int(count_negatif)],
                        "description": description,
                        "dominant": dominant
                    }

                # Nama file hasil prediksi
                file_nama_asli = os.path.splitext(csv_file.filename)[0]
                file_nama_predict = f"{file_nama_asli}_predict.csv"

                return render_template(
                    "predict.html",
                    hasil_tabel=df.to_dict(orient='records'),
                    session_id=session_id,
                    nama_file=csv_file.filename,
                    file_nama_asli = os.path.splitext(csv_file.filename)[0],
                    nama_file_predict = f"{os.path.splitext(csv_file.filename)[0]}_predict.csv",
                    chart_data=chart_data,
                    pie_chart_data=pie_chart_data
                )

            except Exception as e:
                return render_template(
                    "predict.html",
                    error=f"⚠️ Gagal membaca/olah CSV: {str(e)}"
                )

        # CASE 3: Tidak ada input
        else:
            return render_template(
                "predict.html",
                error="⚠️ Silakan masukkan teks atau unggah file CSV."
            )

    # GET method: Ambil hasil dari session (jika ada)
    hasil_prediksi = session.pop("hasil_prediksi", None)
    original = session.pop("original", None)
    return render_template("predict.html", hasil_prediksi=hasil_prediksi, original=original)

# Endpoint untuk download hasil prediksi sebagai file CSV
@app.route("/download-csv")
def download_csv():
    csv_content = request.args.get("data")
    if not csv_content:
        return "Tidak ada data untuk diunduh.", 400

    buffer = io.BytesIO()
    buffer.write(csv_content.encode('utf-8'))
    buffer.seek(0)

    return send_file(
        buffer,
        mimetype='text/csv',
        as_attachment=True,
        download_name='predict_review_results.csv'
    )

# ============================
# MAIN APP RUNNER
# ============================
if __name__ == '__main__':
    print("🚀 Flask berjalan di http://127.0.0.1:5000")
    app.run(debug=True)
