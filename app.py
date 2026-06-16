import streamlit as st
from ml_app import run_ml_app

def main():
    st.sidebar.title("📌 Navigation")
    menu = ["Laptop Price Prediction", "About"]
    choice = st.sidebar.radio("Go to", menu)

    if choice == "Laptop Price Prediction":
        run_ml_app()
    elif choice == "About":
        st.title("ℹ️ About This App")
        st.write("""
        Aplikasi ini dibuat menggunakan **Streamlit** untuk memprediksi harga laptop
        berdasarkan atribut input.  
        Model dilatih menggunakan dataset laptop prices.
        """)

if __name__ == '__main__':
    main()
