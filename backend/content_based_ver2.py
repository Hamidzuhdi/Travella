import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# === 1. Load preprocessed dataset ===
tourism = pd.read_csv("data/preprocessed_tourismnew.csv")
ratings = pd.read_csv("data/preprocessed_ratingsnew.csv")
#ratings = pd.read_csv("data/tourism_rating.csv")
users = pd.read_csv("data/preprocessed_usersnew.csv")

# === 2. Pisahkan Fitur untuk Rekomendasi ===
# Ambil semua kolom fitur kecuali kolom ID dan kolom non-fitur dengan fitur2 dari pengolahan tf idf dan encoding kategori dan kota
exclude_columns = ['Place_Id', 'Place_Name', 'Description', 'Category', 'City', 'Coordinate', 'Lat', 'Long', 'image_url']
feature_columns = [col for col in tourism.columns if col not in exclude_columns]

# category_columns_encode = [col for col in feature_columns if col.startswith('Category_')] 
# city_columns_encode = [col for col in feature_columns if col.startswith('City_')]

# tourism[category_columns_encode] *= 2
# tourism[city_columns_encode] *= 1.5


#print(feature_columns)
final_feature_matrix = tourism[feature_columns].values

# === 3. Hitung Kesamaan ===
similarity_matrix = cosine_similarity(final_feature_matrix)

# Simpan hasil kesamaan dalam DataFrame
similarity_df = pd.DataFrame(similarity_matrix, index=tourism['Place_Id'], columns=tourism['Place_Id'])
#print(similarity_df)

# === 4. Sistem Rekomendasi ===
def recommend_place(user_id, top_n=50, category=None, city=None):
    # Ambil tempat wisata yang disukai pengguna berdasarkan rating
    liked_places = ratings[ratings['User_Id'] == user_id].sort_values(by='Place_Ratings', ascending=False)

    # Jika pengguna belum memberi rating, fallback langsung
    if liked_places.empty:
        print("Pengguna belum memberi rating. Menggunakan fallback berbasis kesamaan.")
        recommendations = tourism.copy()
        if category:
            category_col = f"Category_{category.replace(' ', '_')}"
            #recommendations = recommendations[recommendations['Category'] == category]
            recommendations = recommendations[recommendations[category_col] == 1]
        if city:
            city_col = f"City_{city.replace(' ', '_')}"
            #recommendations = recommendations[recommendations['City'] == city]
            recommendations = recommendations[recommendations[city_col] == 1]
        
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
    
    top_place_info = tourism[tourism['Place_Id'] == top_place_id].copy()
    
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
        category_col = f"Category_{category.replace(' ', '_')}"
        recommendations = recommendations[recommendations[category_col] == 1]
        #recommendations = recommendations[recommendations['Category'] == category]
    
    # Filter berdasarkan kota
    if city:
        city_col = f"City_{city.replace(' ', '_')}"
        recommendations = recommendations[recommendations[city_col] == 1]
        #recommendations = recommendations[recommendations['City'] == city]
    
    # Jika tidak ada hasil setelah filter, gunakan fallback global
    if recommendations.empty:
        print("Tidak ada hasil yang cocok dengan filter. Menggunakan fallback berbasis kesamaan.")
        recommendations = tourism.copy()
        if category:
            category_col = f"Category_{category.replace(' ', '_')}"
            recommendations = recommendations[recommendations[category_col] == 1]
            #recommendations = recommendations[recommendations['Category'] == category]
        if city:
            city_col = f"City_{city.replace(' ', '_')}"
            recommendations = recommendations[recommendations[city_col] == 1]
            #recommendations = recommendations[recommendations['City'] == city]
        subset_ids = recommendations['Place_Id']
        filtered_matrix = similarity_df.loc[subset_ids, subset_ids]
        recommendations['Cosine_Similarity'] = filtered_matrix.mean(axis=1)
        recommendations = recommendations.sort_values(by=['Cosine_Similarity', 'Rating'], ascending=[False, False]).head(top_n)
    
     # tempat wisata yang disukai di posisi awal
    top_place_info['Cosine_Similarity'] = 100  
    recommendations = pd.concat([top_place_info, recommendations], ignore_index=1)
    
    # Hapus duplikasi 
    recommendations = recommendations.drop_duplicates(subset='Place_Id', keep='first')
    
    # Sort ulang untuk memastikan tempat yang disukai tetap di posisi pertama
    recommendations = recommendations.sort_values(by=['Cosine_Similarity', 'Rating'], ascending=[False, False]).head(top_n)
    
    return recommendations[['Place_Id', 'Place_Name', 'Category', 'City', 'Rating', 'Cosine_Similarity','image_url']]




def recomendation_by_category(user_id, top_n=50, category=None, city=None):
    # Ambil tempat wisata yang disukai pengguna berdasarkan rating
    liked_places = ratings[ratings['User_Id'] == user_id].sort_values(by='Place_Ratings', ascending=False)

    # Jika pengguna belum memberi rating, fallback langsung
    if liked_places.empty:
        
        print("Pengguna belum memberi rating. Menggunakan fallback berbasis kesamaan.")
        recommendations = tourism.copy()
        if category:
            category_col = f"Category_{category.replace(' ', '_')}"
            recommendations = recommendations[recommendations[category_col] == 1]
        if city:
            city_col = f"City_{city.replace(' ', '_')}"
            recommendations = recommendations[recommendations[city_col] == 1]
        
        # Hitung cosine similarity sebagai fallback
        subset_ids = recommendations['Place_Id']
        filtered_matrix = similarity_df.loc[subset_ids, subset_ids]
        recommendations['Cosine_Similarity'] = filtered_matrix.mean(axis=1)
        recommendations = recommendations.sort_values(by=['Cosine_Similarity', 'Rating'], ascending=[False, False]).head(top_n)
        return recommendations[['Place_Id', 'Place_Name', 'Category', 'City', 'Rating', 'Cosine_Similarity']]
    
    # Ambil ID tempat wisata dengan rating tertinggi
    top_place_id = liked_places.iloc[0]['Place_Id']
    
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
        category_col = f"Category_{category.replace(' ', '_')}"
        recommendations = recommendations[recommendations[category_col] == 1]
    
    # Filter berdasarkan kota
    if city:
        city_col = f"City_{city.replace(' ', '_')}"
        recommendations = recommendations[recommendations[city_col] == 1]
    
    # Jika tidak ada hasil setelah filter, gunakan fallback global
    if recommendations.empty:
        print("Tidak ada hasil yang cocok dengan filter. Menggunakan fallback berbasis kesamaan.")
        recommendations = tourism.copy()
        if category:
            category_col = f"Category_{category.replace(' ', '_')}"
            recommendations = recommendations[recommendations[category_col] == 1]
        if city:
            city_col = f"City_{city.replace(' ', '_')}"
            recommendations = recommendations[recommendations[city_col] == 1]
        subset_ids = recommendations['Place_Id']
        filtered_matrix = similarity_df.loc[subset_ids, subset_ids]
        recommendations['Cosine_Similarity'] = filtered_matrix.mean(axis=1)
        recommendations = recommendations.sort_values(by=['Cosine_Similarity', 'Rating'], ascending=[False, False]).head(top_n)
    
    # Hapus duplikasi untuk menghindari redundansi
    recommendations = recommendations.drop_duplicates(subset='Place_Id', keep='first')
    
    # Sort ulang berdasarkan kesamaan dan rating
    recommendations = recommendations.sort_values(by=['Cosine_Similarity', 'Rating'], ascending=[False, False]).head(top_n)
    
    return recommendations[['Place_Id', 'Place_Name', 'Category', 'City', 'Rating', 'Cosine_Similarity', 'image_url']]




# === 5. Contoh Rekomendasi untuk Pengguna ===
user_id = 3  # Gantilah dengan ID pengguna yang ingin diuji
recommendations = recommend_place(user_id)
#recommendations = recomendation_by_category(user_id=2, category='Taman Hiburan')
print(recommendations)
