# Import Library yang dibutuhkan
from flask import Flask, render_template, request, jsonify, redirect, url_for
import pickle
import re
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from sqlalchemy import create_engine, text
import os

app = Flask(__name__)

# Menerapkan Lemmatizer
nltk.download('wordnet', quiet=True)
lemmatizer = WordNetLemmatizer()

# Mengambil kedua model pickle
with open('ModelNBC.pkl', 'rb') as f:
    model_hybrid = pickle.load(f)
print("Model Hybrid (ModelNBC.pkl) berhasil diload")

with open('Model-NBC-Multiclass.pkl', 'rb') as f:
    model_one_vs_rest = pickle.load(f)
print("Model One-vs-Rest (Model-NBC-Multiclass.pkl) berhasil diload")

# Simpan metrik akurasi jika tersedia (prioritaskan hybrid sebagai default)
model_accuracy = getattr(model_hybrid, 'accuracy', None)
# Check apakah model dalam pickle sudah dilatih
try:
    check_is_fitted(model_hybrid)
    print("Model sudah fit")
except Exception as e:
    print("Model belum fit:", e)

# Menghubungkan ke database db_siredo
engine = create_engine("mysql+pymysql://root:@localhost/db_siredo")

# Ambil data dosen lengkap (nama, kode, department, expertise, nip, nidn, email)
df_lecturer = pd.read_sql("SELECT code_lec, name, departments_id, expertise, nip, nidn, email FROM lecturers", engine)

# Mapping code_lec ke nama dosen
code_lec_to_name = dict(zip(df_lecturer['code_lec'], df_lecturer['name']))

# Reverse mapping nama (lowercased/trimmed) ke code_lec untuk lookup by name
name_to_code = {
    (name or '').strip().lower(): code
    for code, name in code_lec_to_name.items()
}

# Ambil data department
# (Tetap perlu ambil nama department dari tabel departments)
df_departments = pd.read_sql("SELECT id, name_dept FROM departments", engine)
dept_id_to_name = dict(zip(df_departments['id'], df_departments['name_dept']))
code_lec_to_dept = dict(
    (row['code_lec'], dept_id_to_name.get(row['departments_id'], '-'))
    for _, row in df_lecturer.iterrows()
)

# Mapping code_lec ke expertise
df_lecturer['expertise'] = df_lecturer['expertise'].fillna('-')
code_lec_to_expertise = dict(zip(df_lecturer['code_lec'], df_lecturer['expertise']))

# pilih tabel lecturer_keywords sebagai bahan rekomendasi
query = "SELECT * FROM lecturer_keywords"
df = pd.read_sql(query, engine)

# Pivot table
pivot_df = df.pivot_table(index='keyword', columns='code_lec', values='freq', fill_value=0)

# Tahapan rekomendasi dengan Hybrid
def sistemrekomendasi_hybrid(keyword_list, df, model, pivot_df):
    try:
        # Membersihkan kata-kata kapital dan mengubahnya menjadi kata pada aslinya
        clean_word = re.sub(r'[^\w\s,\.&]', '', keyword_list.lower())
        
        # Split berdasarkan koma, titik, dan ampersand untuk mendukung multi-topik
        # Setiap topik bisa terdiri dari beberapa kata yang harus tetap bersama
        topics = []
        for topic in clean_word.replace('.', ',').replace('&', ',').split(','):
            topic = topic.strip()  # Hapus spasi di awal dan akhir
            if topic:  # Hanya tambahkan jika tidak kosong
                topics.append(topic)
        
        # Untuk setiap topik, split berdasarkan spasi dan lakukan lemmatization
        word_list_lemma = []
        for topic in topics:
            topic_words = topic.split()
            topic_lemma = [lemmatizer.lemmatize(word) for word in topic_words]
            word_list_lemma.extend(topic_lemma)

        # Filter dosen yang memiliki SEMUA kata kunci (multi-keyword)
        lecturer_keywords = {}
        
        # Ambil semua dosen yang memiliki kata kunci yang dimasukkan
        for keyword in word_list_lemma:
            keyword_data = df[df['keyword'] == keyword]
            for _, row in keyword_data.iterrows():
                lecturer_code = row['code_lec']
                if lecturer_code not in lecturer_keywords:
                    lecturer_keywords[lecturer_code] = set()
                lecturer_keywords[lecturer_code].add(keyword)
        
        # Filter hanya dosen yang memiliki SEMUA kata kunci
        matching_lecturers = []
        for lecturer_code, keywords in lecturer_keywords.items():
            if len(keywords) == len(word_list_lemma):  # Harus memiliki semua kata kunci
                matching_lecturers.append(lecturer_code)
        
        if not matching_lecturers:
            return {'message': f"Tidak Ada Dosen yang Memiliki Semua Kata Kunci: {keyword_list}", 'results': []}

        # Ambil data frekuensi untuk dosen yang cocok
        line = df[df['code_lec'].isin(matching_lecturers) & df['keyword'].isin(word_list_lemma)]
        if line.empty:
            return {'message': f"Tidak Ada Dosen yang Relevan Berdasarkan Kata Kunci yaitu {keyword_list}", 'results': []}

        frequency = line.groupby('code_lec')['freq'].sum()
        frequency_dict = frequency.to_dict()

        fitur_df = frequency.reindex(pivot_df.columns, fill_value=0).values.reshape(1, -1)
        
        # Prediksi probabilitas menggunakan model yang dipilih
        proba = model.predict_proba(fitur_df)[0]
        kelas = model.classes_

        # Urutkan dosen berdasarkan frekuensi descending supaya hasil konsisten
        sort_lec = frequency.sort_values(ascending=False).index.tolist()

        results = []
        for lecturer in sort_lec:
            if lecturer in kelas:
                idx = list(kelas).index(lecturer)
                score = proba[idx]
                results.append((lecturer, score))

        if not results:
            return {'message': "Tidak ada dosen yang relevan berdasarkan kata kunci tersebut.", 'results': []}

        # Urutkan hasil berdasarkan skor
        sort_results = sorted(results, key=lambda x: x[1], reverse=True)

        # Format hasil sebagai list dict agar mudah dikirim via JSON
        results_formatted = [
            {
                'lecture': code_lec_to_name.get(d, d),
                'code_lec': d,
                'score': round(s, 4),
                'department': code_lec_to_dept.get(d, '-'),
                'expertise': code_lec_to_expertise.get(d, '-')
            }
            for d, s in sort_results
        ]

        return {
            'message': f"Rekomendasi untuk kata kunci: {' '.join(word_list_lemma)} - Menampilkan dosen yang memiliki SEMUA kata kunci",
            'results': results_formatted
        }

    except NotFittedError:
        return {'message': "Model belum dilatih (fit). Silakan latih model terlebih dahulu sebelum membuat rekomendasi.", 'results': []}
    except Exception as e:
        return {'message': f"Terjadi error lain: {str(e)}", 'results': []}

@app.route('/rekomendasi', methods=['POST'])
def api_rekomendasi():
    data = request.json
    keyword = data.get('kata_kunci', '')
    approach = (data.get('approach') or 'hybrid').lower()
    
    if not keyword:
        return jsonify({'error': 'Masukkan kata kunci yang valid'}), 400

    # Pilih model berdasarkan pendekatan
    if approach in ['onevsrest', 'ovr', 'one-vs-rest']:
        chosen_model = model_one_vs_rest
    else:
        chosen_model = model_hybrid

    hasil = sistemrekomendasi_hybrid(keyword, df, chosen_model, pivot_df)
    return jsonify(hasil)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/profile/<identifier>')
def view_profile(identifier):
    # identifier bisa berupa code_lec langsung atau nama dosen
    lecturer_code = None

    # 1) Jika identifier adalah code_lec yang valid
    if identifier in code_lec_to_name:
        lecturer_code = identifier
    else:
        # 2) Coba cocokkan sebagai nama dosen (case-insensitive)
        normalized_name = (identifier or '').strip().lower()
        lecturer_code = name_to_code.get(normalized_name)
    
    if lecturer_code is None:
        return "Lecturer not found", 404

    lecturer_name = code_lec_to_name.get(lecturer_code, identifier)
    print(f"Lecturer Code for {lecturer_name}: {lecturer_code}")

    # Get lecturer's complete data from database
    query = text("""
        SELECT l.*, d.name_dept
        FROM lecturers l
        LEFT JOIN departments d ON l.departments_id = d.id
        WHERE l.code_lec = :code_lec
    """)
    
    with engine.connect() as connection:
        result = connection.execute(query, {"code_lec": lecturer_code})
        lecturer_data = result.fetchone()
        
        # Convert to dict and add the name from code_lec_to_name
        lecturer = {
            'name': lecturer_name,  # Pakai dari df(code_lec_to_name) => bukan dari database
            'code_lec': lecturer_code,
            'nip': lecturer_data.nip if lecturer_data else None,
            'nidn': lecturer_data.nidn if lecturer_data else None,
            'email': lecturer_data.email if lecturer_data else None,
            'name_dept': lecturer_data.name_dept if lecturer_data else None,
            'expertise': lecturer_data.expertise if lecturer_data else None
        }

    # Check for LDA visualization files
    static_folder = app.static_folder # This gives the absolute path to the static folder
    
    lda_id_path = f"LDAVis_id/{lecturer_code}.html"
    full_lda_id_path = os.path.join(static_folder, lda_id_path)
    lda_id_exists = os.path.exists(full_lda_id_path)
    print(f"Checking ID path: {full_lda_id_path}, Exists: {lda_id_exists}")

    lda_en_path = f"LDAVis_en/{lecturer_code}.html"
    full_lda_en_path = os.path.join(static_folder, lda_en_path)
    lda_en_exists = os.path.exists(full_lda_en_path)
    print(f"Checking EN path: {full_lda_en_path}, Exists: {lda_en_exists}")

    # Get lecturer's publications
    query_publications = text("""
        SELECT title, linkURL
        FROM publication
        WHERE code_lec = :code_lec
        ORDER BY id DESC
    """)
    with engine.connect() as connection:
        publications_result = connection.execute(query_publications, {"code_lec": lecturer_code})
        publications = [row._asdict() for row in publications_result.fetchall()]
    
    return render_template('viewprofile.html', 
                         lecturer=lecturer,
                         lda_id_path=lda_id_path if lda_id_exists else None,
                         lda_en_path=lda_en_path if lda_en_exists else None,
                         publications=publications,
                         model_accuracy=model_accuracy
                         )

@app.route('/lecturers')
def lecturers_page():
    # Get all departments for filter
    query_departments = text("SELECT id, name_dept FROM departments ORDER BY name_dept")
    with engine.connect() as connection:
        departments_result = connection.execute(query_departments)
        departments = [row._asdict() for row in departments_result.fetchall()]
    
    # Get all lecturers with their department info
    query_lecturers = text("""
        SELECT l.code_lec, l.name, l.expertise, l.nip, l.nidn, l.email, d.name_dept
        FROM lecturers l
        LEFT JOIN departments d ON l.departments_id = d.id
        ORDER BY l.name
    """)
    
    with engine.connect() as connection:
        lecturers_result = connection.execute(query_lecturers)
        lecturers = [row._asdict() for row in lecturers_result.fetchall()]
    
    return render_template('lecturers.html', 
                         lecturers=lecturers,
                         departments=departments)

@app.route('/api/lecturers')
def api_lecturers():
    department_filter = request.args.get('department', '')
    
    # Build query based on filter
    if department_filter:
        query = text("""
            SELECT l.code_lec, l.name, l.expertise, l.nip, l.nidn, l.email, d.name_dept
            FROM lecturers l
            LEFT JOIN departments d ON l.departments_id = d.id
            WHERE d.name_dept = :department
            ORDER BY l.name
        """)
        params = {"department": department_filter}
    else:
        query = text("""
            SELECT l.code_lec, l.name, l.expertise, l.nip, l.nidn, l.email, d.name_dept
            FROM lecturers l
            LEFT JOIN departments d ON l.departments_id = d.id
            ORDER BY l.name
        """)
        params = {}
    
    with engine.connect() as connection:
        lecturers_result = connection.execute(query, params)
        lecturers = [row._asdict() for row in lecturers_result.fetchall()]
    
    return jsonify(lecturers)

if __name__ == '__main__':
    app.run(debug=True)