import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Konfigurasi awal agar sidebar langsung terbuka
st.set_page_config(page_title="Deteksi Telur", layout="centered", initial_sidebar_state="expanded")

# Load model YOLO
model = YOLO("best.pt")  # Ganti path sesuai lokasi model kamu

# Menu navigasi
menu = st.sidebar.radio("Pilih Menu", ("Definisi", "Kriteria Foto Telur", "Deteksi Telur"))

if menu == "Definisi":
    st.title("📘 Definisi Telur Fertil dan Infertil")

    st.markdown("""
    <div style='text-align: justify; font-size: 16px;'>
        <strong>Telur Fertil</strong> adalah telur yang telah dibuahi oleh sperma ayam jantan. Jika dierami dalam kondisi yang tepat,
        telur ini bisa berkembang menjadi embrio dan menetas menjadi anak ayam. Ciri khas telur fertil mulai tampak dengan munculnya
        garis pembuluh darah pada hari ke-3 hingga ke-5 setelah inkubasi. Telur ini cocok untuk proses pembibitan dan penetasan.
        <br><br>
        <strong>Telur Infertil</strong> adalah telur yang tidak dibuahi dan tidak akan berkembang menjadi embrio. Biasanya dihasilkan oleh ayam
        betina tanpa proses kawin. Saat diteropong (candling), telur infertil tampak bening tanpa adanya jaringan atau titik embrio. Telur ini
        aman dikonsumsi dan biasanya digunakan sebagai telur konsumsi sehari-hari.
        <br><br>
        <em>Mengetahui perbedaan antara telur fertil dan infertil sangat penting bagi peternak untuk mengoptimalkan hasil produksi,
        meningkatkan efisiensi proses penetasan, serta mengurangi risiko kerugian akibat kegagalan penetasan.</em>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Contoh Gambar Telur Fertil:**", unsafe_allow_html=True)
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.image("181_Fertil.jpg", caption="Telur Fertil - Terlihat garis pembuluh darah", width=250)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("""
        **Ciri-ciri Telur Fertil:**
        - Terdapat bercak embrio atau titik putih berkembang.
        - Tampak jaring pembuluh darah saat candling.
        - Warna isi telur tidak terlalu bening.
        - Bisa berkembang menjadi anak ayam jika dierami.
        """)

    with col2:
        st.markdown("**Contoh Gambar Telur Infertil:**", unsafe_allow_html=True)
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.image("36_Non Fertil.jpg", caption="Telur Infertil - Tampak bening tanpa jaringan", width=250)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("""
        **Ciri-ciri Telur Infertil:**
        - Tidak ada titik embrio atau jaringan darah.
        - Tampak bening saat candling.
        - Warna isi telur cenderung jernih.
        - Tidak bisa menetas.
        """)

elif menu == "Kriteria Foto Telur":
    st.title("🔍 Kriteria Foto Telur yang Layak Dideteksi")
    st.markdown("""
    Agar proses deteksi telur fertil dan infertil menggunakan model YOLO berjalan optimal, berikut beberapa **kriteria penting** untuk foto telur yang akan diunggah:

    ### ✅ Foto yang Disarankan:
    - **Penerangan cukup**: Foto terang, tidak terlalu gelap atau overexposed.
    - **Latar belakang netral**: Usahakan latar belakang polos (hitam/putih/abu) untuk mempermudah deteksi objek.
    - **Fokus pada telur**: Foto tidak blur dan bagian telur terlihat jelas.
    - **1 telur per foto** *(jika memungkinkan)*: Model lebih akurat mendeteksi objek tunggal.

    ### ❌ Hindari Foto Berikut:
    - Terlalu banyak bayangan atau pantulan cahaya.
    - Foto dari jarak terlalu jauh atau terlalu dekat.
    - Foto dengan banyak objek lain di sekitarnya.
    - Gambar hasil crop kasar atau blur.

    ### Contoh Foto yang Disarankan:
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.image("181_Fertil.jpg", caption="Contoh Foto Telur Fertil", width=250)
    with col2:
        st.image("36_Non Fertil.jpg", caption="Contoh Foto Telur Infertil", width=250)

    st.info("Dengan memenuhi kriteria di atas, proses deteksi menggunakan YOLO akan lebih akurat dan dapat memberikan rekomendasi yang lebih tepat untuk peternak.")

elif menu == "Deteksi Telur":
    st.title("🖼️ Deteksi Telur Fertil dan Infertil")

    uploaded_file = st.file_uploader("Upload gambar telur", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Gambar Diupload", width=400)

        with st.spinner("Mendeteksi..."):
            result = model.predict(img)[0]
            st.image(result.plot(), caption="Hasil Deteksi", width=400)

            st.subheader("Deteksi:")
            if result.boxes:
                for box in result.boxes:
                    label_index = int(box.cls)
                    conf = box.conf.item()

                    if label_index == 1:
                        label = "Telur Fertil"
                        rekomendasi = "Tempatkan di inkubator dengan suhu dan kelembaban optimal. Periksa kembali dengan candling pada hari ke-3 dan ke-7."
                        penjelasan = "Telur fertil telah dibuahi dan memiliki potensi berkembang menjadi embrio hingga menetas menjadi anak ayam."
                    elif label_index == 0:
                        label = "Telur Tidak Fertil"
                        rekomendasi = "Pisahkan dari inkubator. Telur ini bisa digunakan sebagai konsumsi atau dipasarkan."
                        penjelasan = "Telur tidak fertil adalah telur yang tidak dibuahi oleh sperma ayam jantan. Tidak memiliki potensi menetas."
                    else:
                        label = "Label tidak dikenal"
                        rekomendasi = "-"
                        penjelasan = "-"

                    st.write(f"**{label}** (Confidence: {conf:.2f})")
                    st.markdown(f"**Penjelasan:** {penjelasan}")
                    st.markdown(f"**Rekomendasi:** {rekomendasi}")
            else:
                st.warning("Objek telur tidak terdeteksi dalam gambar. Coba lagi dengan gambar yang lebih jelas.")
