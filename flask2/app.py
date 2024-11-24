from flask import Flask, render_template, request, flash, redirect, url_for, session
import pandas as pd
from dotenv import load_dotenv
import os
from flask2.model import recommend_place

app = Flask(__name__)

load_dotenv()

app.secret_key = os.getenv("SECRET_KEY")

@app.route('/', methods= ["GET", "POST"])
def index():
    if request.method == "POST":
        email = request.form.get('email')
        password = request.form.get('password')
        print(f"{email}, {password}")
        try:
            user = pd.read_csv('../flask_ver1/data/preprocessed_users.csv')
        except FileNotFoundError:
            flash("Data pengguna tidak ditemukan. Harap periksa file user.csv.", "error")
            return render_template("templates/index.html")
        
        user_exist = user[(user['Email'] == email) & (user['Password'] == password)]
        if not user_exist.empty:
            #idUser = user_exist.iloc[0]['User_Id']
            idUser = int(user_exist.iloc[0]['User_Id'])

           
            session['user_id'] = idUser
            
            flash(f"Login berhasil! Selamat datang, User ID: {idUser}", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Email atau password salah. Silakan coba lagi.", "error")
            return render_template("templates/index.html")
            
    return render_template("templates/index.html")



@app.route("/dashboard", methods=["GET"])
def dashboard():
    user_id = session.get('user_id')
    
    if not user_id:
        flash("Sesi telah berakhir. Silakan login kembali.", "error")
        return redirect(url_for('index'))
    
    categories = [
        {"name": "Budaya", "slug": "budaya"},
        {"name": "Taman Hiburan", "slug": "taman-hiburan"},
        {"name": "Cagar Alam", "slug": "cagar-alam"},
        {"name": "Taman Hiburan", "slug": "taman-hiburan"},
        {"name": "Bahari", "slug": "bahari"},
        {"name": "Pusat Perbelanjaan", "slug": "pusat-perbelanjaan"},
        {"name": "Tempat Ibadah", "slug": "tempat-ibadah"},
    ] 
    recommendations = recommend_place(user_id)
    
    print("Tipe data recommendations:", type(recommendations))

    if isinstance(recommendations, pd.DataFrame):
        recommendations = recommendations.to_dict('records')
        print("recommendations sudah diubah menjadi list of dictionaries.")

    print(type(recommendations))  
    
    
    return render_template('index.html', user_id=user_id, recommendations=recommendations, categories=categories)


@app.route('/category/<category_slug>', methods=["GET"])
def category_content(category_slug):
    user_id = session.get('user_id')
    
    if not user_id:
        flash("Sesi telah berakhir. Silakan login kembali.", "error")
        return redirect(url_for('index'))
    
    categories = [
        {"name": "Budaya", "slug": "budaya"},
        {"name": "Taman Hiburan", "slug": "taman-hiburan"},
        {"name": "Cagar Alam", "slug": "cagar-alam"},
        {"name": "Taman Hiburan", "slug": "taman-hiburan"},
        {"name": "Bahari", "slug": "bahari"},
        {"name": "Pusat Perbelanjaan", "slug": "pusat-perbelanjaan"},
        {"name": "Tempat Ibadah", "slug": "tempat-ibadah"},
    ]
    
    recommendations = recommend_place(user_id, category=category_slug)
    
    print("Tipe data recommendations:", type(recommendations))

    if isinstance(recommendations, pd.DataFrame):
        recommendations = recommendations.to_dict('records')
        print("recommendations sudah diubah menjadi list of dictionaries.")

    print(type(recommendations))  
    
    
    return render_template('categories.html', user_id=user_id, recommendations=recommendations, categories=categories) 

@app.route('/logout')
def logout():
    session.clear()  
    flash("Anda telah logout.", "success")
    return redirect(url_for('index'))


if __name__ == "__main__":
    app.run(debug=True)

