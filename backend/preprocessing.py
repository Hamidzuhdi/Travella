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
        "https://img.freepik.com/free-photo/public-square-with-empty-road-floor-downtown_1127-3691.jpg",
        "https://e1.pxfuel.com/desktop-wallpaper/469/445/desktop-wallpaper-kota-tua-jakarta.jpg",
        "https://c1.staticflickr.com/5/4143/4904374767_1a45e48fce_b.jpg",
        "https://thumbs.dreamstime.com/b/jakarta-indonesia-february-th-tourist-seen-fatahillah-museum-well-known-as-kota-tua-old-town-central-jakarta-indonesia-172466103.jpg",
        "https://c0.wallpaperflare.com/preview/247/664/874/indonesia-museum-bank-mandiri.jpg"
    ],
    "Taman Hiburan": [
        "https://png.pngtree.com/thumb_back/fw800/background/20230308/pngtree-taman-mini-indonesia-indah-is-a-culture-based-recreational-area-located-photo-image_1863506.jpg",
        "https://3.bp.blogspot.com/-DHJNf4sHyRY/TuxGSbntmAI/AAAAAAAAAGc/D7eK3m9lcyI/s1600/Ancol.jpg",
        "https://2.bp.blogspot.com/-eetDHEnCxKA/WRkkaT_7x-I/AAAAAAAAZh0/61Jk58D-w9YHpgrZnxcarLhJ6W5X5So7gCLcB/s1600/eco1.jpg",
        "https://live.staticflickr.com/1273/1093244703_3312cec7d5_b.jpg",
        "https://www.tiketmasuk.com/wp-content/uploads/2023/03/Taman-Menteng-jakarta-665x374.jpg"
    ],
    "Cagar Alam": [
        "https://statik.tempo.co/data/2021/06/25/id_1030566/1030566_720.jpg",
        "https://img.okezone.com/content/2017/06/19/406/1719832/uncover-indonesia-menelusuri-keindahan-hutan-mangrove-bali-dGYNoGcgfZ.JPG",
        "https://www.hargatiket.net/wp-content/uploads/2018/09/Gembira-Loka-Zoo.jpg",
        "https://visitingjogja.jogjaprov.go.id/wp-content/uploads/2021/10/gl7.jpg",
        "https://i1.wp.com/www.syakirurohman.net/wp-content/uploads/2015/10/air-terjun-2-warna.jpg?fit=1024%2C768&ssl=1"
    ],
    "Tempat Ibadah": [
        "https://images.pexels.com/photos/10634440/pexels-photo-10634440.jpeg?auto=compress&cs=tinysrgb&w=800",
        "https://images.pexels.com/photos/3155276/pexels-photo-3155276.jpeg?auto=compress&cs=tinysrgb&w=800",
        "https://images.pexels.com/photos/28154007/pexels-photo-28154007/free-photo-of-gate-of-heaven-bali.jpeg?auto=compress&cs=tinysrgb&w=900",
        "https://images.pexels.com/photos/13688470/pexels-photo-13688470.jpeg?auto=compress&cs=tinysrgb&w=800",
        "https://images.pexels.com/photos/22065599/pexels-photo-22065599/free-photo-of-tomb-of-hazrat-shah-rukn-e-alam.jpeg?auto=compress&cs=tinysrgb&w=800"
    ],
    "Bahari": [
        "https://wallpapercave.com/wp/wp3218393.jpg",
        "https://indonesiakaya.com/wp-content/uploads/2020/10/pantai-goa-cina-1200.jpg",
        "https://cdn.pixabay.com/photo/2023/02/11/23/53/goa-7783922_1280.jpg",
        "https://4.bp.blogspot.com/-uMJDlNnjdo4/UaeMChakRMI/AAAAAAAAAyA/ghXUeAQdgyM/s1600/tumblr_mjjsu4FSUB1qlz3jlo7_1280.jpg",
        "https://www.tempatwisata.pro/users_media/3092/Pantai-Goa-Cina-Cover.jpg"
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
