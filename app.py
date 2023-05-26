import streamlit as st
import joblib


def main():

    st.header("Data Mining")
    
            
    word = st.text_input("Enter the text:  ")

    # Carica il modello dal file .pkl
    pipe = joblib.load('NLPEs1.pkl')
    
    if word == "":
        st.warning("Input text is required.")
    else:
    
        fake = []
        fake.append(word)
        
        predictions = pipe.predict(fake)
        
        
        # Cambia il colore del background in base al sentiment
        if predictions[0] == "real":
            st.write(
                f'<div style="background-color: lightgreen; padding: 5px;">'
                f'Sentiment: {predictions[0].upper()}'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            st.write(
                f'<div style="background-color: rgb(255, 100, 100); padding: 5px;">'
                f'Sentiment: {predictions[0].upper()}'
                '</div>',
                unsafe_allow_html=True
            )

                

if __name__ == "__main__":
    main()
