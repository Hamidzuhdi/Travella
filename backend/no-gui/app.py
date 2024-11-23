from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import re
import string

app = Flask(__name__)

# Load datasets
tourism_rating = pd.read_csv('../data/tourism_rating.csv')
tourism_with_id = pd.read_csv('../data/tourism_with_id.csv')
user_data = pd.read_csv('../data/user.csv')  # Load user data

# Preprocessing
def preprocess_data():
    # Merge tourism_rating with tourism_with_id to get place_name
    merged_data = pd.merge(tourism_rating, tourism_with_id, on='Place_Id')
    
    # Group by user_id and place_id to calculate average ratings
    grouped_data = merged_data.groupby(['User_Id', 'Place_Id'], as_index=False)['Place_Ratings'].mean()
    
    # Calculate the overall average rating per place
    place_avg_rating = grouped_data.groupby('Place_Id', as_index=False)['Place_Ratings'].mean()
    place_avg_rating = place_avg_rating.rename(columns={'Place_Ratings': 'avg_rating'})
    
    # Join with tourism_with_id to add place_name
    place_avg_rating = pd.merge(place_avg_rating, tourism_with_id[['Place_Id', 'Place_Name']], on='Place_Id')
    
    return merged_data, grouped_data, place_avg_rating

merged_data, grouped_data, place_avg_rating = preprocess_data()

# Split data into train and test sets (80-20 split)
train_data, test_data = train_test_split(grouped_data, test_size=0.2, random_state=42)

# Text preprocessing function
def preprocess_text(text):
    # Case folding
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenization
    tokens = text.split()
    # Remove stop words (elimination and filtering)
    stop_words = set(['dan', 'di', 'ke', 'dari', 'yang', 'untuk', 'pada', 'dengan', 'sebagai', 'adalah', 'ini', 'itu'])
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Preprocess descriptions
tourism_with_id['processed_description'] = tourism_with_id['Description'].apply(preprocess_text)

# Convert category names to lowercase
tourism_with_id['Category'] = tourism_with_id['Category'].str.lower()

# Calculate TF-IDF matrix
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(tourism_with_id['processed_description'])

# Calculate similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('User_Id', type=int)
    
    if user_id is None:
        return jsonify({"error": "User_Id is required"}), 400
    
    # Filter user data to get age
    user_info = user_data[user_data['User_Id'] == user_id]
    if user_info.empty:
        return jsonify({"error": "User_Id not found"}), 404
    
    # Convert age to Python int
    user_age = int(user_info.iloc[0]['Age'])
    
    # Filter training data for the given user_id
    user_ratings = merged_data[merged_data['User_Id'] == user_id]
    
    if user_ratings.empty:
        return jsonify({"error": "No data found for this user"}), 404
    
    # Filter places visited by the user with a rating of 5
    user_ratings = user_ratings[user_ratings['Place_Ratings'] == 5]
    
    if user_ratings.empty:
        return jsonify({"error": "No places with a rating of 5 found for this user"}), 404
    
    # Sort places visited by the user by rating (highest to lowest)
    sorted_ratings = user_ratings.sort_values(by='Place_Ratings', ascending=False)
    
    # Get the place_id with the highest average rating
    top_place_id = int(sorted_ratings.iloc[0]['Place_Id'])
    top_place_data = tourism_with_id[tourism_with_id['Place_Id'] == top_place_id].iloc[0]
    
    # Extract details of the recommended place
    top_place_name = top_place_data['Place_Name']
    top_place_category = top_place_data['Category']
    top_place_city = top_place_data['City']
    
    # Find other places with the same category and city
    similar_places = tourism_with_id[
        (tourism_with_id['Category'] == top_place_category) & 
        (tourism_with_id['City'] == top_place_city) &
        (tourism_with_id['Place_Id'] != top_place_id)
    ]['Place_Name'].tolist()
    
    # Prepare a list of all places visited by the user with their ratings
    visited_places = []
    for _, row in sorted_ratings.iterrows():
        visited_places.append({
            "place_name": tourism_with_id[tourism_with_id['Place_Id'] == int(row['Place_Id'])].iloc[0]['Place_Name'],
            "rating": float(row['Place_Ratings'])  # Convert to Python float
        })
    
    return jsonify({
        "User_Id": int(user_id),  # Convert to Python int
        "Age": user_age,
        "Recommended_Place": {
            "name": top_place_name,
            "category": top_place_category,
            "city": top_place_city
        },
        "Visited_Places": visited_places,
        "Additional_Recommendations": f"Mungkin Anda juga suka... {', '.join(similar_places)}"
    })

@app.route('/category/<category_name>', methods=['GET'])
def kategori(category_name):
    # Convert category_name to lowercase
    category_name = category_name.lower()
    
    # Filter places by category
    category_places = tourism_with_id[tourism_with_id['Category'] == category_name]
    
    if category_places.empty:
        return jsonify({"error": "Category not found"}), 404
    
    # Get the indices of the places in the category
    indices = category_places.index.tolist()
    
    # Calculate similarity scores for places in the category
    sim_scores = []
    for idx in indices:
        score = cosine_sim[idx].mean()
        sim_scores.append((idx, score))
    
    # Sort places by similarity scores in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Take top 20 places
    top_20_sim_scores = sim_scores[:20]
    
    # Prepare recommendations
    recommendations = []
    for idx, score in top_20_sim_scores:
        place = tourism_with_id.iloc[idx]
        recommendations.append({
            "Place_Id": int(place['Place_Id']),
            "Place_Name": place['Place_Name'],
            "Description": place['Description'],
            "Category": place['Category'],
            "City": place['City'],
            "Price": int(place['Price']) if not pd.isna(place['Price']) else None,
            "Rating": float(place['Rating']),
            "Time_Minutes": int(place['Time_Minutes']) if not pd.isna(place['Time_Minutes']) else None,
            "Coordinate": place['Coordinate'],
            "Lat": float(place['Lat']),
            "Long": float(place['Long']),
            "Similarity_Score": round(score * 100, 2)  # Convert to percentage
        })
    
    return jsonify({
        "Category": category_name,
        "Recommendations": recommendations
    })

if __name__ == '__main__':
    app.run(debug=True)
