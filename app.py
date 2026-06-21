
import streamlit as st
import pandas as pd
import tempfile
import matplotlib.pyplot as plt
from matcher import SongMatcher
from database import build_database

st.set_page_config(page_title="Sonic Signature Identifier")
st.title("Q3B Sonic Signature")


import os

if not os.path.exists("database/fingerprints.pkl"):
    st.error(
        "Fingerprint database missing in deployment."
    )
    st.stop()

matcher = SongMatcher()

mode=st.radio("Mode",["Single Clip","Batch"])

if mode=="Single Clip":
    f=st.file_uploader("Upload clip",type=["wav","mp3","flac"])
    if f:
        with tempfile.NamedTemporaryFile(delete=False,suffix=".wav") as tmp:
            tmp.write(f.read())
            p=tmp.name

        result=matcher.match(p)

        st.success("Prediction: "+result["prediction"])
        st.write("Confidence:",result["confidence"])

        fig=plt.figure()
        plt.imshow(result["spectrogram"],aspect="auto",origin="lower")
        plt.title("Spectrogram")
        st.pyplot(fig)

        peaks=result["peaks"]
        if len(peaks)>0:
            x=[p[1] for p in peaks]
            y=[p[0] for p in peaks]
            fig=plt.figure()
            plt.scatter(x,y,s=2)
            plt.title("Constellation Peaks")
            st.pyplot(fig)

        hist=result["offset_histogram"]
        if result["prediction"] in hist:
            fig=plt.figure()
            plt.hist(list(hist[result["prediction"]].elements()),bins=50)
            plt.title("Offset Histogram")
            st.pyplot(fig)

else:
    files=st.file_uploader("Upload query clips",accept_multiple_files=True)

    if st.button("Generate results.csv"):
        rows=[]
        for f in files:
            with tempfile.NamedTemporaryFile(delete=False,suffix=".wav") as tmp:
                tmp.write(f.read())
                p=tmp.name
            r=matcher.match(p)
            rows.append({"filename":f.name,
                         "prediction":r["prediction"]})

        df=pd.DataFrame(rows)
        df.to_csv("results.csv",index=False)

        st.dataframe(df)
        st.download_button("Download results.csv",
                           open("results.csv","rb"),
                           "results.csv")
