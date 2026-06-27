import tempfile
import pandas as pd
import streamlit as st
import plotly.express as px
from matcher import SubstringAudioMatcher

class SignalAnalysisDashboard:

    def __init__(self):
        st.set_page_config(page_title="Sonic Signature Identifier", layout="wide")
        self.matcher = self._initialize_engine()

    @st.cache_resource
    def _initialize_engine(_self):
        return SubstringAudioMatcher()

    def draw_header(self):
        st.title("Q3B Sonic Signature")

    def render_single_audit(self):
        f = st.file_uploader("Upload clip", type=["wav", "mp3", "flac"])
        
        if f:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(f.read())
                p = tmp.name

            with st.spinner("Analyzing audio footprint..."):
                metrics = self.matcher.execute_identification(p)
            
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"**Prediction:** {metrics['prediction']}")
                st.metric("Confidence Score", f"{metrics['confidence']}%")
                st.metric("Peak Alignment Votes", metrics['votes'])

            # --- PLOTLY SPECTROGRAM HEATMAP ---
            st.subheader("Spectrogram Heatmap")
            if "spectrogram" in metrics and metrics["spectrogram"] is not None:
                spec_data = metrics["spectrogram"]
                
                if spec_data.shape[0] > 150:
                    spec_data = spec_data[::4, :]
                if spec_data.shape[1] > 400:
                    spec_data = spec_data[:, ::int(spec_data.shape[1]/400)]
                    
                fig_spec = px.imshow(
                    spec_data, 
                    color_continuous_scale="Viridis", 
                    origin="lower",
                    labels=dict(x="Time Bin", y="Frequency Bin", color="Intensity")
                )
                fig_spec.update_layout(
                    margin=dict(l=40, r=20, t=20, b=40),
                    height=350,
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_spec, use_container_width=True)

            # --- PLOTLY CONSTELLATION PEAKS ---
            st.subheader("Constellation Peaks")
            peaks = metrics.get("peaks", [])
            if len(peaks) > 0:
                peaks_df = pd.DataFrame({
                    "Time Bin": [p["time_frame"] for p in peaks],
                    "Frequency Bin": [p["frequency"] for p in peaks]
                })
                
                fig_peaks = px.scatter(
                    peaks_df, 
                    x="Time Bin", 
                    y="Frequency Bin",
                    labels={"Time Bin": "Time Bin", "Frequency Bin": "Frequency Bin"}
                )
                fig_peaks.update_traces(marker=dict(size=6, color="#EF553B"))
                fig_peaks.update_layout(
                    margin=dict(l=40, r=20, t=20, b=40),
                    height=400
                )
                st.plotly_chart(fig_peaks, use_container_width=True)

            # --- PLOTLY OFFSET FREQUENCY ALIGNMENT ---
            st.subheader("Offset Frequency Alignment")
            hist = metrics.get("histogram_data", {})
            
            if hist:
                hist_df = pd.DataFrame(list(hist.items()), columns=["Time Offset", "Votes"]).sort_values("Time Offset")
                
                fig_hist = px.bar(
                    hist_df, 
                    x="Time Offset", 
                    y="Votes",
                    labels={"Time Offset": "Time Offset", "Votes": "Votes"}
                )
                fig_hist.update_layout(
                    margin=dict(l=40, r=20, t=20, b=40),
                    height=400
                )
                st.plotly_chart(fig_hist, use_container_width=True)

    def render_batch_audit(self):
        files = st.file_uploader("Upload query clips", accept_multiple_files=True, type=["wav", "mp3", "flac"])

        if st.button("Generate results.csv") and files:
            rows = []
            progress_bar = st.progress(0)
            
            for i, f in enumerate(files):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(f.read())
                    p = tmp.name
                
                r = self.matcher.execute_identification(p)
                rows.append({
                    "filename": f.name,
                    "prediction": r["prediction"],
                    "confidence": f"{r['confidence']}%"
                })
                progress_bar.progress((i + 1) / len(files))

            df = pd.DataFrame(rows)
            df.to_csv("results.csv", index=False)
            st.dataframe(df, use_container_width=True)
            
            with open("results.csv", "rb") as file_data:
                st.download_button(label="Download results.csv", data=file_data, file_name="results.csv", mime="text/csv")

    def execute(self):
        self.draw_header()
        mode = st.radio("Mode", ["Single Clip", "Batch"])
        
        if mode == "Single Clip":
            self.render_single_audit()
        else:
            self.render_batch_audit()

if __name__ == "__main__":
    dashboard = SignalAnalysisDashboard()
    dashboard.execute()
    
