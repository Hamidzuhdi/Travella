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



# === 3. Preprocessing Tourism Data ===

# a. Normalisasi fitur numerik
numeric_features = ['Price', 'Rating', 'Time_Minutes']
scaler = MinMaxScaler()
tourism[numeric_features] = scaler.fit_transform(tourism[numeric_features])

# b. Encoding fitur kategorikal
category_encoded = pd.get_dummies(tourism['Category'], prefix='Category')
city_encoded = pd.get_dummies(tourism['City'], prefix='City')

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
tourism.to_csv("data/preprocessed_tourism.csv", index=False)
ratings.to_csv("data/preprocessed_ratings.csv", index=False)
users.to_csv("data/preprocessed_users.csv", index=False)

print("Preprocessing selesai:")
print("- tourism: preprocessed_tourism.csv")
print("- ratings: preprocessed_ratings.csv")
print("- users: preprocessed_users.csv")
