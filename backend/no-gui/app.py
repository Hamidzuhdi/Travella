from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load datasets
tourism_rating = pd.read_csv('backend/data/tourism_rating.csv')
tourism_with_id = pd.read_csv('backend/data/tourism_with_id.csv')
user_data = pd.read_csv('backend/data/user.csv')  # Load user data

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
def category(category_name):
    # Define the valid categories
    valid_categories = ['budaya', 'tamanhiburan', 'cagar alam', 'bahari', 'pusat pembelanjaan', 'tempat ibadah']
    
    if category_name not in valid_categories:
        return jsonify({"error": "Invalid category"}), 400
    
    # Log the category name for debugging
    print(f"Category requested: {category_name}")
    
    # Filter places by the given category
    category_places = tourism_with_id[tourism_with_id['Category'].str.lower() == category_name.lower()]
    
    if category_places.empty:
        print(f"No places found for category: {category_name}")
        return jsonify({"error": "No places found for this category"}), 404
    
    # Merge with ratings to get the average rating for each place
    category_ratings = pd.merge(category_places, place_avg_rating, on='Place_Id')
    
    # Check if 'Place_Name' column exists after merge
    if 'Place_Name' not in category_ratings.columns:
        print("Column 'Place_Name' not found after merge")
        return jsonify({"error": "Internal server error"}), 500
    
    # Sort by average rating and get the top 20
    top_places = category_ratings.sort_values(by='avg_rating', ascending=False).head(20)
    
    # Prepare the response data
    top_places_list = []
    for _, row in top_places.iterrows():
        top_places_list.append({
            "place_name": row['Place_Name'],
            "rating": float(row['avg_rating']),  # Convert to Python float
            "description": row.get('Description', 'No description available')  # Assuming there's a 'Description' column
        })
    
    return jsonify({
        "Category": category_name,
        "Top_Places": top_places_list
    })


if __name__ == '__main__':
    app.run(debug=True)
