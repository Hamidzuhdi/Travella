import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import string
import random

# === 1. Load datasets ===
tourism = pd.read_csv("data/tourism_with_id.csv")
ratings = pd.read_csv("data/tourism_rating.csv")
users = pd.read_csv("data/user.csv")

# Hapus kolom tidak diperlukan di tourism
tourism.drop(columns=['Unnamed: 11', 'Unnamed: 12'], inplace=True, errors='ignore')

tourism.info()
ratings.info()
users.info()

print(tourism.isnull().sum())
print(ratings.isnull().sum())
print(users.isnull().sum())
# === 2. Periksa dan Tangani Null ===

# a. Periksa nilai null di dataset tourism
if tourism.isnull().sum().sum() > 0:
    print("Nilai null ditemukan di dataset tourism.")
    # Isi null di kolom 'Time_Minutes' dengan rata-rata
    tourism['Time_Minutes'] = tourism['Time_Minutes'].fillna(tourism['Time_Minutes'].mean())
    # Jika ada null di kolom lain, hapus barisnya
    tourism.dropna(inplace=True)  
    
    
# b. Periksa nilai null di dataset ratings
if ratings.isnull().sum().sum() > 0:
    print("Nilai null ditemukan di dataset ratings.")
    ratings.dropna(inplace=True)  # Hapus baris dengan nilai null

# c. Periksa nilai null di dataset users
if users.isnull().sum().sum() > 0:
    print("Nilai null ditemukan di dataset users.")
    users.dropna(inplace=True)  # Hapus baris dengan nilai null
    
 
# ==== Create Data Dummies User ==== 

username_random = [
    "rangga", "hamid","levina", "john", "betty", "jane", "mike", "susan", "alex", "david", "linda",
    "robert", "karen", "paul", "laura", "kevin", "amy", "steve", "sarah"
]


def generate_username(row):
    users = random.choice(username_random)
    unique_number = random.randrange(1000, 9999)
    return f"{users}{unique_number}"
    
def generate_email(username):
    return f"{username}@gmail.com"

def generate_password(username):
    return f"{username}123"

def generate_password(length=8):
    characters = string.ascii_letters + string.digits + string.punctuation
    return ''.join(random.choice(characters) for i in range(length))

users["Username"] = users.apply(generate_username, axis=1)
users["Email"] = users["Username"].apply(generate_email)
users["Password"] = [generate_password() for _ in range(len(users))]


# === CREATE IMAGE DUMMMY URL ======= 

image_urls = {
    "Budaya": [
        "https://images.pexels.com/photos/6513506/pexels-photo-6513506.jpeg?auto=compress&cs=tinysrgb&w=800",
        "https://images.pexels.com/photos/18889680/pexels-photo-18889680/free-photo-of-air-cairan-agama-hindu.jpeg?auto=compress&cs=tinysrgb&w=800",
        "https://images.pexels.com/photos/3646848/pexels-photo-3646848.jpeg?auto=compress&cs=tinysrgb&w=800",
        "https://images.pexels.com/photos/14479741/pexels-photo-14479741.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1",
        "https://images.pexels.com/photos/13005422/pexels-photo-13005422.jpeg?auto=compress&cs=tinysrgb&w=800"
    ],

    "Taman Hiburan": [
        "https://images.pexels.com/photos/22092310/pexels-photo-22092310/free-photo-of-lumba-lumba-di-akuarium.jpeg?auto=compress&cs=tinysrgb&w=800",
        "https://images.pexels.com/photos/5692430/pexels-photo-5692430.jpeg?auto=compress&cs=tinysrgb&w=800&lazy=load",
        "https://images.pexels.com/photos/5692435/pexels-photo-5692435.jpeg?auto=compress&cs=tinysrgb&w=800",
        "https://images.pexels.com/photos/22046745/pexels-photo-22046745/free-photo-of-hitam-dan-putih-hitam-putih-hitam-putih-hitam-putih.jpeg?auto=compress&cs=tinysrgb&w=800",
        "https://images.pexels.com/photos/29510962/pexels-photo-29510962.png?auto=compress&cs=tinysrgb&w=800" 
    ],

    "Cagar Alam": [
        "https://images.pexels.com/photos/16228243/pexels-photo-16228243/free-photo-of-scenery.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=800",
        "https://images.pexels.com/photos/27528414/pexels-photo-27528414/free-photo-of-mangrove.jpeg?auto=compress&cs=tinysrgb&w=800",
        "https://images.pexels.com/photos/238631/pexels-photo-238631.jpeg?auto=compress&cs=tinysrgb&w=600",
        "https://images.pexels.com/photos/29421651/pexels-photo-29421651/free-photo-of-monkey-perched-on-ornate-stone-statue-in-jungle.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=800",
        "https://images.pexels.com/photos/28908035/pexels-photo-28908035/free-photo-of-serene-coastal-mangrove-forest-landscape.jpeg?auto=compress&cs=tinysrgb&w=800"
    ],

    "Tempat Ibadah": [
        "https://images.pexels.com/photos/10634440/pexels-photo-10634440.jpeg?auto=compress&cs=tinysrgb&w=800",
        "https://images.pexels.com/photos/3155276/pexels-photo-3155276.jpeg?auto=compress&cs=tinysrgb&w=800",
        "https://images.pexels.com/photos/28154007/pexels-photo-28154007/free-photo-of-gate-of-heaven-bali.jpeg?auto=compress&cs=tinysrgb&w=900",
        "https://images.pexels.com/photos/13688470/pexels-photo-13688470.jpeg?auto=compress&cs=tinysrgb&w=800",
        "https://images.pexels.com/photos/22065599/pexels-photo-22065599/free-photo-of-tomb-of-hazrat-shah-rukn-e-alam.jpeg?auto=compress&cs=tinysrgb&w=800"
    ],

    "Bahari": [
        "https://images.pexels.com/photos/29116365/pexels-photo-29116365/free-photo-of-scenic-pier-over-turquoise-waters-in-marsa-alam.jpeg?auto=compress&cs=tinysrgb&w=800",
        "https://images.pexels.com/photos/18785841/pexels-photo-18785841/free-photo-of-a-sign-attached-to-a-palm-tree-on-the-beach.jpeg?auto=compress&cs=tinysrgb&w=800",
        "https://images.pexels.com/photos/29355304/pexels-photo-29355304/free-photo-of-stunning-sunset-over-pulau-aman-beach-in-penang.jpeg?auto=compress&cs=tinysrgb&w=800",
        "https://images.pexels.com/photos/68767/divers-underwater-ocean-swim-68767.jpeg?auto=compress&cs=tinysrgb&w=800",
        "https://images.pexels.com/photos/11447571/pexels-photo-11447571.jpeg?auto=compress&cs=tinysrgb&w=800"
    ],

    "Pusat Perbelanjaan": [
        "https://res.cloudinary.com/wegowordpress/image/upload/h_531,w_750/v1492760459/IMG_20170418_095509_eefdyi.jpg",
        "https://img.inews.co.id/media/822/files/inews_new/2022/03/04/pasar_glodok_ist.jpg",
        "https://images.halfhalftravel.com/myanmar/what-to-do-in-yangon/DAN05726.jpg?width=700",
        "https://pergiyuk.com/wp-content/uploads/2019/10/plaza_indonesia.jpg",
        "https://dtravelsround.com/wp-content/uploads/2014/07/photo-5.jpg"
    ]
}

def get_random_image_url(category):
    if category in image_urls:
        return random.choice(image_urls[category])
    return None

tourism['image_url'] = tourism['Category'].apply(get_random_image_url)


# === 3. Preprocessing Tourism Data ===

# a. Normalisasi fitur numerik
numeric_features = ['Price', 'Rating', 'Time_Minutes']
scaler = MinMaxScaler()
tourism[numeric_features] = scaler.fit_transform(tourism[numeric_features])

# b. Encoding fitur kategorikal
tourism['Category'] = tourism['Category'].str.replace(' ', '_')
tourism['City'] = tourism['City'].str.replace(' ', '_')
category_encoded = pd.get_dummies(tourism['Category'], prefix='Category', dtype=int)
city_encoded = pd.get_dummies(tourism['City'], prefix='City', dtype=int)

# Gabungkan encoding ke dataframe utama
tourism = pd.concat([tourism, category_encoded, city_encoded], axis=1)

# c. Vectorization fitur teks
tfidf = TfidfVectorizer(stop_words='english', max_features=500)
description_matrix = tfidf.fit_transform(tourism['Description'].fillna(''))  # Jika ada null, isi dengan string kosong

# Simpan TF-IDF matrix ke dalam DataFrame dengan kolom numerik
tfidf_df = pd.DataFrame(description_matrix.toarray(), columns=[f"TFIDF_{i}" for i in range(description_matrix.shape[1])])
tourism = pd.concat([tourism.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)


print("========= AFTER PREPROCESSING =========")
print(tourism.isnull().sum())
print(ratings.isnull().sum())
print(users.isnull().sum())


# === 4. Ekspor hasil preprocessing ===
# Simpan hasil preprocessing ke dalam file terpisah
tourism.to_csv("data/preprocessed_tourismnew.csv", index=False)
ratings.to_csv("data/preprocessed_ratingsnew.csv", index=False)
users.to_csv("data/preprocessed_usersnew.csv", index=False)

print("Preprocessing selesai:")
print("- tourism: preprocessed_tourism.csv")
print("- ratings: preprocessed_ratings.csv")
print("- users: preprocessed_users.csv")
