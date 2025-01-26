import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
import pandas as pd
import plotly.express as px
import hydralit_components as hc

# Inisialisasi MediaPipe untuk deteksi wajah
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Fungsi untuk mendeteksi landmark wajah
def detect_face_landmarks(image):
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return results

# Fungsi untuk menggambar landmark pada gambar
def draw_landmarks(image, landmarks):
    annotated_image = image.copy()
    if landmarks.multi_face_landmarks:
        for face_landmarks in landmarks.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
    return annotated_image

# Fungsi untuk menghitung jarak berdasarkan mask
def calculate_distances(landmarks, image_shape, mask_pairs):
    distances = {}
    h, w, _ = image_shape
    for feature_name, (start_idx, end_idx) in mask_pairs.items():
        pt1 = landmarks[start_idx]
        pt2 = landmarks[end_idx]
        x1, y1 = int(pt1.x * w), int(pt1.y * h)
        x2, y2 = int(pt2.x * w), int(pt2.y * h)
        distances[feature_name] = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distances

# Fungsi untuk analisis fitur wajah menggunakan mask
def analyze_face_features(landmarks1, landmarks2, image1, image2):
    if not (landmarks1.multi_face_landmarks and landmarks2.multi_face_landmarks):
        return None, "Wajah tidak terdeteksi pada salah satu gambar"
    
    # Ambil landmark wajah dari gambar pertama dan kedua
    face1 = landmarks1.multi_face_landmarks[0].landmark
    face2 = landmarks2.multi_face_landmarks[0].landmark

    # Definisikan mask (indeks landmark yang dianalisis)
    mask_pairs = {
        "Bibir Vertikal": (13, 14),
        "Bibir Horizontal": (61, 291),
        "Pipi": (234, 454),
        "Mata Vertikal": (159, 145),
        "Jarak Antar Mata": (33, 133),
        "Dahi": (10, 7),
        "Alis": (70, 300),
        "Dagu": (152, 10)
    }

    # Hitung jarak pada gambar pertama dan kedua menggunakan mask
    distances1 = calculate_distances(face1, image1.shape, mask_pairs)
    distances2 = calculate_distances(face2, image2.shape, mask_pairs)

    # Hitung perubahan (%)
    changes = {}
    for feature in distances1.keys():
        changes[feature] = ((distances2[feature] - distances1[feature]) / distances1[feature]) * 100

    # Analisis Ekspresi
    smile_threshold = 20
    laughter_threshold = 50
    shock_threshold = 10  # Misalnya, threshold untuk ekspresi kaget
    sad_threshold = 5     # Misalnya, threshold untuk ekspresi sedih
    anger_threshold = 30  # Misalnya, threshold untuk ekspresi marah

    smile_expression = "Ekspresi Netral"

    # Menambahkan ekspresi berdasarkan perubahan Bibir Vertikal
    if changes["Bibir Vertikal"] > laughter_threshold:
        smile_expression = "Tertawa"
    elif changes["Bibir Vertikal"] > smile_threshold:
        smile_expression = "Senyum"
    elif changes["Bibir Vertikal"] < -shock_threshold:
        smile_expression = "Kaget"  # Jika Bibir Vertikal sangat menurun, bisa menandakan ekspresi kaget
    elif changes["Bibir Vertikal"] < -sad_threshold:
        smile_expression = "Sedih"  # Jika Bibir Vertikal menurun sedikit, bisa menandakan ekspresi sedih
    elif changes["Bibir Vertikal"] > anger_threshold:
        smile_expression = "Marah"  # Jika Bibir Vertikal sangat meningkat, bisa menandakan ekspresi marah
    else:
        smile_expression = "Ekspresi Netral"  # Default ekspresi jika tidak ada perubahan signifikan

   # Return hasil analisis
    analysis = {
        "Fitur": list(mask_pairs.keys()) + ["Ekspresi"],
        "Sebelum Ekspresi (px)": [f"{distances1[feature]:.2f}" for feature in mask_pairs.keys()] + ["Netral"],
        "Setelah Ekspresi (px)": [f"{distances2[feature]:.2f}" for feature in mask_pairs.keys()] + [smile_expression],
        "Perubahan (%)": [f"{changes[feature]:.2f}%" for feature in mask_pairs.keys()] + [""]
    }

    return analysis, changes, smile_expression

# Fungsi visualisasi perubahan fitur wajah menggunakan Plotly
def plot_interactive_changes(changes):
    # Buat DataFrame dari dictionary perubahan
    df = pd.DataFrame({"Fitur": list(changes.keys()), "Perubahan (%)": list(changes.values())})
    
    # Buat bar chart interaktif dengan Plotly
    fig = px.bar(
        df, 
        x="Fitur", 
        y="Perubahan (%)", 
        title="Perubahan Fitur Wajah", 
        text="Perubahan (%)", 
        labels={"Perubahan (%)": "Perubahan (%)"},
        color="Perubahan (%)"
    )
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', height=400)
    
    # Tampilkan plot di Streamlit
    st.plotly_chart(fig)

# Fungsi halaman unggah gambar
def upload_page():
    st.title("Unggah Gambar")
    st.write("Unggah dua gambar wajah: sebelum dan setelah ekspresi.")
    
    # Upload file
    uploaded_file1 = st.file_uploader("Unggah gambar sebelum ekspresi", type=["jpg", "png", "jpeg"], key="file1")
    uploaded_file2 = st.file_uploader("Unggah gambar setelah ekspresi", type=["jpg", "png", "jpeg"], key="file2")
    
    if uploaded_file1 and uploaded_file2:
        # Simpan gambar ke session state
        st.session_state.image1 = cv2.imdecode(np.frombuffer(uploaded_file1.read(), np.uint8), cv2.IMREAD_COLOR)
        st.session_state.image2 = cv2.imdecode(np.frombuffer(uploaded_file2.read(), np.uint8), cv2.IMREAD_COLOR)
        st.success("Gambar berhasil diunggah!")

# Fungsi halaman analisis
def analysis_page():
    if "image1" not in st.session_state or "image2" not in st.session_state:
        st.warning("Silakan unggah gambar terlebih dahulu di halaman 'Unggah Gambar'.")
        return
    
    st.title("Analisis Fitur Wajah")
    image1 = st.session_state.image1
    image2 = st.session_state.image2

    # Deteksi landmark
    landmarks1 = detect_face_landmarks(image1)
    landmarks2 = detect_face_landmarks(image2)

    # Analisis fitur wajah
    analysis, changes, smile_expression = analyze_face_features(landmarks1, landmarks2, image1, image2)
    
    if not analysis:
        st.error("Gagal mendeteksi wajah pada gambar.")
    else:
        st.subheader("Perbandingan Fitur Wajah")
        df = pd.DataFrame(analysis)
        st.table(df)
        
        # Gambarkan landmark pada gambar
        annotated_image1 = draw_landmarks(image1, landmarks1)
        annotated_image2 = draw_landmarks(image2, landmarks2)
        
        # Tampilkan gambar dengan landmark
        st.subheader("Gambar dengan Landmark")
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(annotated_image1, cv2.COLOR_BGR2RGB), caption="Sebelum Ekspresi", use_container_width=True)
        with col2:
            st.image(cv2.cvtColor(annotated_image2, cv2.COLOR_BGR2RGB), caption="Setelah Ekspresi", use_container_width=True)

        # Plot perubahan jarak fitur wajah
        st.subheader("Visualisasi Perubahan Fitur Wajah")
        plot_interactive_changes(changes)

# Fungsi halaman About
def about_page():
    st.title("Tentang Aplikasi")
    st.write("Aplikasi ini menggunakan MediaPipe dan Streamlit untuk menganalisis fitur wajah.")
    st.write("**Fitur Utama:**")
    st.write("- Mengunggah gambar sebelum dan sesudah ekspresi.")
    st.write("- Analisis fitur wajah dengan perbandingan detail.")
    st.write("- Visualisasi landmark pada wajah.")
    st.write("- Visualisasi interaktif dengan Plotly.")
    st.write("**Dikembangkan oleh:**")
    st.write("- Nama: Irvan Fernanda")
    st.write("- Universitas Dharma Wacana")
    st.write("- Jurusan: Teknik Informatika")

# Fungsi untuk menambahkan CSS kustom
def add_css():
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            background-color: #2E4053; /* Warna latar belakang sidebar */
            height: 100vh; /* Membuat tinggi sidebar penuh hingga bawah */
            display: flex;
            flex-direction: column;
            justify-content: space-between; /* Gambar tetap di bagian bawah */
        }
        .nav-footer {
            text-align: center;
            margin: 10px 0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Fungsi utama
def main():
    # Menambahkan CSS
    add_css()

    # Definisi menu navigasi
    navbar = [
        {'icon': "fas fa-upload", 'label': "Unggah Gambar", 'id': "upload"},
        {'icon': "fas fa-chart-bar", 'label': "Analisis Fitur", 'id': "analysis"},
        {'icon': "fas fa-info-circle", 'label': "Tentang Aplikasi", 'id': "about"}
    ]

    # Sidebar untuk navigasi
    with st.sidebar:
        # Tema dengan latar belakang biru
        over_theme = {
            'menu_background': '#2E4053',
            'txc_active': '#0066CC',
        }

        # Navigasi utama
        menu_id = hc.nav_bar(menu_definition=navbar, home_name='Home', override_theme=over_theme)

        # Bagian bawah untuk gambar
        st.markdown('<div class="nav-footer">', unsafe_allow_html=True)
        st.image("C:/Users/LENOVO/Downloads/buatkan gambar ai tentang analisis wajah.png", caption="Gambar di bawah", use_container_width=True)  
        st.markdown('</div>', unsafe_allow_html=True)

    # Halaman Home (berisi logo kampus dan deskripsi aplikasi)
    if menu_id == "Home":
        st.title("Selamat Datang di Aplikasi Analisis Ekspresi Wajah")
        
        # Menampilkan logo kampus
        try:
            st.image("C:/Users/LENOVO/Downloads/logo udw.png", caption="Logo Universitas Dharma Wacana", use_container_width=True)
        except Exception as e:
            st.error(f"Error membuka gambar logo: {e}")
        
        # Deskripsi aplikasi
        st.subheader("Deskripsi Aplikasi")
        st.write("""
            Aplikasi ini dibuat sebagai bagian dari tugas Ujian Akhir Semester (UAS) untuk mata kuliah Teknik Informatika. 
            Tujuan dari aplikasi ini adalah untuk menganalisis ekspresi wajah pengguna dengan menggunakan teknik deteksi landmark wajah.
            Dengan aplikasi ini, pengguna dapat mengunggah dua gambar wajah dan membandingkan ekspresi mereka sebelum dan sesudah 
            ekspresi tertentu, seperti senyum, tertawa, kaget, atau marah.
        """)
        
        st.write("**Fitur Utama Aplikasi:**")
        st.write("- Deteksi wajah menggunakan MediaPipe.")
        st.write("- Perbandingan ekspresi wajah sebelum dan sesudah perubahan.")
        st.write("- Visualisasi interaktif menggunakan Plotly.")

    # Navigasi antar halaman
    if menu_id == "upload":
        upload_page()
    elif menu_id == "analysis":
        analysis_page()
    elif menu_id == "about":
        about_page()

if __name__ == "__main__":
    main()

