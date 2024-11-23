import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity

tourism = pd.read_csv("data/preprocessed_tourism.csv")
ratings = pd.read_csv("data/preprocessed_ratings.csv")
users = pd.read_csv("data/preprocessed_users.csv")

exclude_columns = ['Place_Id', 'Place_Name', 'Description', 'Category', 'City', 'Coordinate', 'Lat', 'Long']
feature_column = [col for col in tourism.columns if col not in exclude_columns]
final_feature_matrix = tourism[feature_column].values


similarity_matrix = cosine_similarity(final_feature_matrix)

similarity_df = pd.DataFrame(similarity_matrix, index=tourism['Place_Id'], columns=tourism['Place_Id'])

# === 4. Sistem Rekomendasi ===
def recommend_place(user_id, top_n=50, category=None, city=None):
    # Ambil tempat wisata yang disukai pengguna berdasarkan rating
    liked_places = ratings[ratings['User_Id'] == user_id].sort_values(by='Place_Ratings', ascending=False)

    # Jika pengguna belum memberi rating, fallback langsung
    if liked_places.empty:
        print("Pengguna belum memberi rating. Menggunakan fallback berbasis kesamaan.")
        recommendations = tourism.copy()
        if category:
            recommendations = recommendations[recommendations['Category'] == category]
        if city:
            recommendations = recommendations[recommendations['City'] == city]
        
        # Hitung cosine similarity sebagai fallback
        subset_ids = recommendations['Place_Id']
        filtered_matrix = similarity_df.loc[subset_ids, subset_ids]
        recommendations['Cosine_Similarity'] = filtered_matrix.mean(axis=1)
        recommendations = recommendations.sort_values(by=['Cosine_Similarity', 'Rating'], ascending=[False, False]).head(top_n)
        return recommendations[['Place_Id', 'Place_Name', 'Category', 'City', 'Rating', 'Cosine_Similarity']]
    
    # Ambil ID tempat wisata dengan rating tertinggi
    top_place_id = liked_places.iloc[0]['Place_Id']
    top_place_name = tourism[tourism['Place_Id'] == top_place_id]['Place_Name'].values[0]
    top_place_category = tourism[tourism['Place_Id'] == top_place_id]['Category'].values[0]
    top_place_city = tourism[tourism['Place_Id'] == top_place_id]['City'].values[0]
    top_place_rating = tourism[tourism['Place_Id'] == top_place_id]['Rating'].values[0]
    
    print(f"Tempat wisata yang disukai: {top_place_name}")
    print(f"Kategori: {top_place_category}, Kota: {top_place_city}, Rating: {top_place_rating}")
    
    # Cari tempat wisata yang mirip berdasarkan kesamaan
    similar_places = similarity_df[top_place_id].sort_values(ascending=False).head(top_n + 1)
    recommended_ids = similar_places.index[1:]  # Skip tempat wisata yang sama
    
    # Ambil nilai cosine similarity
    similarity_values = similar_places[1:]  # Ambil nilai similarity yang sesuai dengan tempat yang direkomendasikan
    
    # Tampilkan rekomendasi dengan cosine similarity
    recommendations = tourism[tourism['Place_Id'].isin(recommended_ids)].copy()
    
    # Menambahkan nilai cosine similarity
    recommendations.loc[:, 'Cosine_Similarity'] = similarity_values.values
    recommendations.loc[:, 'Cosine_Similarity'] = recommendations['Cosine_Similarity'] * 100  # Mengubah ke persentase
    
    # Filter berdasarkan kategori
    if category:
        recommendations = recommendations[recommendations['Category'] == category]
    
    # Filter berdasarkan kota
    if city:
        recommendations = recommendations[recommendations['City'] == city]
    
    # Jika tidak ada hasil setelah filter, gunakan fallback global
    if recommendations.empty:
        print("Tidak ada hasil yang cocok dengan filter. Menggunakan fallback berbasis kesamaan.")
        recommendations = tourism.copy()
        if category:
            recommendations = recommendations[recommendations['Category'] == category]
        if city:
            recommendations = recommendations[recommendations['City'] == city]
        subset_ids = recommendations['Place_Id']
        filtered_matrix = similarity_df.loc[subset_ids, subset_ids]
        recommendations['Cosine_Similarity'] = filtered_matrix.mean(axis=1)
        recommendations = recommendations.sort_values(by=['Cosine_Similarity', 'Rating'], ascending=[False, False]).head(top_n)
    
    return recommendations[['Place_Id', 'Place_Name', 'Category', 'City', 'Rating', 'Cosine_Similarity']]


