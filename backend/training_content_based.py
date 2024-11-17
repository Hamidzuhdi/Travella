import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split

# === 1. Load preprocessed dataset ===
tourism = pd.read_csv("data/preprocessed_tourism.csv")
ratings = pd.read_csv("data/tourism_rating.csv")
users = pd.read_csv("data/preprocessed_users.csv")

# === 2. Pisahkan Fitur untuk Rekomendasi ===
exclude_columns = ['Place_Id', 'Place_Name', 'Description', 'Category', 'City', 'Coordinate', 'Lat', 'Long']
feature_columns = [col for col in tourism.columns if col not in exclude_columns]
final_feature_matrix = tourism[feature_columns].values

# === 3. Hitung Kesamaan ===
similarity_matrix = cosine_similarity(final_feature_matrix)

# Simpan hasil kesamaan dalam DataFrame
similarity_df = pd.DataFrame(similarity_matrix, index=tourism['Place_Id'], columns=tourism['Place_Id'])

# === 4. Train-Test Split ===
train_ratings, test_ratings = train_test_split(ratings, test_size=0.2, random_state=42)

# === 5. Sistem Rekomendasi ===
def recommend_place(user_id, top_n=5):
    # Ambil tempat wisata yang disukai pengguna berdasarkan rating
    liked_places = train_ratings[train_ratings['User_Id'] == user_id].sort_values(by='Place_Ratings', ascending=False)
    
    if liked_places.empty:
        return "Pengguna belum memberi rating tempat wisata."
    
    # Ambil ID tempat wisata dengan rating tertinggi
    top_place_id = liked_places.iloc[0]['Place_Id']
    
    # Cari tempat wisata yang mirip berdasarkan kesamaan
    similar_places = similarity_df[top_place_id].sort_values(ascending=False).head(top_n + 1)
    recommended_ids = similar_places.index[1:]  # Skip tempat wisata yang sama
    
    # Ambil nilai cosine similarity
    similarity_values = similar_places[1:]  # Ambil nilai similarity yang sesuai dengan tempat yang direkomendasikan
    
    # Tampilkan rekomendasi dengan cosine similarity
    recommendations = tourism[tourism['Place_Id'].isin(recommended_ids)].copy()  # Membuat salinan dari DataFrame
    
    # Menambahkan nilai cosine similarity dengan .loc untuk menghindari peringatan
    recommendations.loc[:, 'Cosine_Similarity'] = similarity_values.values  # Menambahkan nilai cosine similarity
    recommendations.loc[:, 'Cosine_Similarity'] = recommendations['Cosine_Similarity'] * 100  # Mengubah ke persentase
    
    return recommendations[['Place_Id', 'Place_Name', 'Category', 'City', 'Rating', 'Cosine_Similarity']]

# === 6. Precision at K dan Recall at K ===
def precision_at_k(recommended_places, actual_places, k=5):
    recommended_at_k = recommended_places[:k]
    relevant_recommendations = sum([1 for place in recommended_at_k if place in actual_places])
    return relevant_recommendations / k

def recall_at_k(recommended_places, actual_places, k=5):
    recommended_at_k = recommended_places[:k]
    relevant_recommendations = sum([1 for place in recommended_at_k if place in actual_places])
    return relevant_recommendations / len(actual_places) if len(actual_places) > 0 else 0

# === 7. Evaluasi Menggunakan Test Set ===
def evaluate_recommendations(user_id, k=5):
    # Ambil rekomendasi berdasarkan data train
    recommendations = recommend_place(user_id, top_n=k)
    
    if isinstance(recommendations, str):
        print(recommendations)  # Jika pengguna tidak memberikan rating
        return
    
    # Ambil data rating pengguna di test set
    test_data = test_ratings[test_ratings['User_Id'] == user_id]
    
    # Cek apakah tempat wisata yang direkomendasikan ada di data test
    recommended_places = recommendations['Place_Id'].values
    actual_places = test_data['Place_Id'].values
    
    # Menyesuaikan jumlah rekomendasi dengan data pengujian
    if len(recommended_places) > len(actual_places):
        recommended_places = recommended_places[:len(actual_places)]  # Potong rekomendasi
    elif len(recommended_places) < len(actual_places):
        # Jika rekomendasi kurang, tambahkan nilai default (misalnya -1 atau tempat wisata tidak dikenal)
        recommended_places = list(recommended_places) + [-1] * (len(actual_places) - len(recommended_places))
    
    # Buat confusion matrix
    cm = confusion_matrix(actual_places, recommended_places)
    
    # Menghitung precision dan recall
    precision = precision_score(actual_places, recommended_places, average='micro')
    recall = recall_score(actual_places, recommended_places, average='micro')
    
    # Menghitung Precision at K dan Recall at K
    precision_k = precision_at_k(recommended_places, actual_places, k)
    recall_k = recall_at_k(recommended_places, actual_places, k)
    
    print(f"Confusion Matrix:\n{cm}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"Precision at {k}: {precision_k:.2f}")
    print(f"Recall at {k}: {recall_k:.2f}")


# === 8. Contoh Evaluasi untuk Pengguna ===
user_id = 2  # Gantilah dengan ID pengguna yang ingin diuji
evaluate_recommendations(user_id, k=5)
