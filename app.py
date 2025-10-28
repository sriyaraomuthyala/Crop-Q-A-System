import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import re

st.set_page_config(page_title="Rainfall & Crop QnA — Professional", layout="wide")

# -----------------------
# Paths & dataset links
# -----------------------
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "Data")
INDEX_DIR = os.path.join(BASE_DIR, "Qna_index")
os.makedirs(INDEX_DIR, exist_ok=True)

FILES = {
    "completed": os.path.join(DATA_DIR, "completed_rainfall_by_season.csv"),
    "seasonal_district": os.path.join(DATA_DIR, "seasonal_rainfall_district_level.csv"),
    "rainfall_data": os.path.join(DATA_DIR, "rainfall_data.csv"),
    "subdivision": os.path.join(DATA_DIR, "subdivision_data.csv"),
    "crop": os.path.join(DATA_DIR, "crop_data.csv"),
}

DATASET_LINKS = {
    "Rainfall Data (district normals 1951-2000)": "https://www.data.gov.in/resource/district-rainfall-normal-mm-monthly-seasonal-and-annual-data-period-1951-2000",
    "Crop Data (district-wise seasonwise, 1997)": "https://www.data.gov.in/resource/district-wise-season-wise-crop-production-statistics-1997",
    "Seasonal/Daily District Rainfall": "https://www.data.gov.in/resource/daily-district-wise-rainfall-data",
    "Subdivision Monthly Rainfall 1901-2017": "https://www.data.gov.in/resource/sub-divisional-monthly-rainfall-1901-2017",
}

# -----------------------
# Safe CSV load
# -----------------------
def safe_read_csv(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, encoding="latin1", on_bad_lines="skip")

def norm_cols(df):
    if df is None or df.empty:
        return df
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df

completed = norm_cols(safe_read_csv(FILES["completed"]))
seasonal_district = norm_cols(safe_read_csv(FILES["seasonal_district"]))
rainfall_data = norm_cols(safe_read_csv(FILES["rainfall_data"]))
subdivision = norm_cols(safe_read_csv(FILES["subdivision"]))
crop = norm_cols(safe_read_csv(FILES["crop"]))

# normalize annual
for df in (completed, seasonal_district, subdivision):
    if df is None or df.empty:
        continue
    for c in list(df.columns):
        if c.lower() in ["year_avg_rainfall","avg_annual_rainfall","annual","yearavg","year_avg"]:
            df.rename(columns={c:"annual"}, inplace=True)

# -----------------------
# TF-IDF index
# -----------------------
VEC_PATH = os.path.join(INDEX_DIR, "tfidf_vectorizer.pkl")
MAT_PATH = os.path.join(INDEX_DIR, "tfidf_matrix.npz")
META_PATH = os.path.join(INDEX_DIR, "meta.pkl")

def build_documents():
    docs = []
    sources = [(completed, "completed"), (rainfall_data, "rainfall"), (seasonal_district, "seasonal"), (subdivision, "subdivision"), (crop, "crop")]
    for df, tag in sources:
        if df is None or df.empty:
            continue
        for i, r in df.iterrows():
            parts = []
            for c in df.columns:
                v = r.get(c)
                if pd.notna(v) and str(v).strip() != "":
                    parts.append(f"{c}:{v}")
            text = " ".join(parts)
            if text.strip():
                docs.append({"id": f"{tag}_{i}", "text": text, "meta": {"source": tag}})
    return docs

def build_tfidf_index(force=False):
    if (not force) and os.path.exists(VEC_PATH) and os.path.exists(MAT_PATH) and os.path.exists(META_PATH):
        try:
            vec = pickle.load(open(VEC_PATH,"rb"))
            X = sparse.load_npz(MAT_PATH)
            meta = pickle.load(open(META_PATH,"rb"))
            return vec, X, meta
        except Exception:
            pass
    docs = build_documents()
    if not docs:
        docs = [{"id":"dummy","text":"no_data_available","meta":{"source":"none"}}]
    texts = [d["text"] for d in docs]
    vec = TfidfVectorizer(stop_words=None, token_pattern=r"(?u)\b\w+\b", max_features=50000)
    X = vec.fit_transform(texts)
    os.makedirs(INDEX_DIR, exist_ok=True)
    pickle.dump(vec, open(VEC_PATH,"wb"))
    sparse.save_npz(MAT_PATH, X)
    pickle.dump(docs, open(META_PATH,"wb"))
    return vec, X, docs

tfidf_vec, tfidf_matrix, meta_docs = build_tfidf_index()

def retrieve_tfidf(query, k=5):
    try:
        qv = tfidf_vec.transform([query])
        sims = cosine_similarity(qv, tfidf_matrix).flatten()
        idxs = sims.argsort()[::-1][:k]
        return [{"score": float(sims[i]), "text": meta_docs[i]["text"], "meta": meta_docs[i]["meta"]} for i in idxs]
    except Exception:
        return []

# -----------------------
# Formatting helper (universal)
# -----------------------
def prettify_record(raw_text: str):
    """Turn 'col:value' strings into natural human-like sentences."""
    import re

    if not isinstance(raw_text, str):
        return str(raw_text)

    pairs = []
    for token in raw_text.split():
        if ":" in token:
            k, v = token.split(":", 1)
            k = k.replace("_", " ").strip().lower()
            pairs.append((k, v.strip()))

    def format_number(val):
        try:
            num = float(val)
            if num.is_integer():
                return f"{num:,.0f}"
            else:
                return f"{num:,.2f}"
        except Exception:
            return val

    # Extract fields
    year = None
    location = None
    crop_name = None
    area = None
    production = None
    rainfall = None
    others = []

    for k, v in pairs:
        v_fmt = format_number(v)
        if "year" in k and re.match(r"^\d{4}$", v):
            year = v
        elif any(x in k for x in ["district", "subdivision", "state"]):
            location = v.title()
        elif "crop" in k:
            crop_name = v.title()
        elif "area" in k:
            area = v_fmt
        elif "production" in k:
            production = v_fmt
        elif "rainfall" in k or "annual" in k:
            rainfall = v_fmt
        else:
            others.append(f"{k}: {v_fmt}")

    # Construct smooth readable sentence
    parts = []
    if year:
        parts.append(f"In {year},")
    if location:
        parts.append(location)

    # Production-based phrasing
    if production and crop_name:
        phrase = f"produced {production} tonnes of {crop_name}"
        if area:
            phrase += f" over an area of {area} hectares"
        parts.append(phrase + ".")
    elif crop_name and area:
        parts.append(f"had {crop_name} grown over an area of {area} hectares.")
    elif crop_name:
        parts.append(f"had {crop_name} cultivation data recorded.")
    elif production:
        parts.append(f"recorded a production of {production} tonnes.")

    # Add rainfall info
    if rainfall:
        if any("produced" in p or "had" in p for p in parts):
            parts[-1] = parts[-1].rstrip(".") + f" and rainfall of {rainfall} mm."
        else:
            parts.append(f"recorded rainfall of {rainfall} mm.")

    # Handle any leftover columns
    if others:
        extra = ", ".join(others)
        parts.append(f"({extra})")

    sentence = " ".join(parts)
    sentence = sentence.replace(" ,", ",").strip()
    if not sentence.endswith("."):
        sentence += "."
    return sentence[0].upper() + sentence[1:]


# -----------------------
# Sidebar (only buttons)
# -----------------------
with st.sidebar:
    st.title("Select Operation")

    operations = [
        "Free-text Q&A",
        "Compare average rainfall & top crops",
        "State's District-wise max/min production",
        "Production trend & rainfall correlation",
        "Policy advisor: top 3 arguments"
    ]

    if "selected_op" not in st.session_state:
        st.session_state.selected_op = operations[0]

    for op in operations:
        if st.button(op, use_container_width=True):
            st.session_state.selected_op = op

operation = st.session_state.selected_op

# -----------------------
# UI helpers
# -----------------------
def result_block(title, html_body):
    st.markdown(f"""
    <div style="background:#E8F5E9; border-left:4px solid #2E7D32; padding:20px; border-radius:8px; margin-bottom:10px;">
      <div style="font-size:20px; color:#1B5E20; font-weight:700;">{title}</div>
      <div style="font-size:16px; color:#155724; margin-top:8px;">{html_body}</div>
    </div>
    """, unsafe_allow_html=True)

def sources_box(names):
    """
    Render clickable dataset source links reliably using Streamlit Markdown.
    Converts dataset names into Markdown link syntax if found in DATASET_LINKS.
    """
    if not names:
        return

    lines = []
    for item in names:
        # handle dicts or plain strings
        if isinstance(item, dict):
            for label, url in item.items():
                lines.append(f"- [{label}]({url})")
            continue

        label = str(item).strip()
        url = DATASET_LINKS.get(label)

        # if no exact match, try case-insensitive
        if not url:
            for key, val in DATASET_LINKS.items():
                if key.lower() == label.lower():
                    url = val
                    break

        # markdown link if URL found
        if url:
            lines.append(f"- [{label}]({url})")
        else:
            lines.append(f"- {label}")

    md = "#### 📚 Data Sources\n\n" + "\n".join(lines)
    st.markdown(md, unsafe_allow_html=False)

# -----------------------
# Main header
# -----------------------
st.title("Rainfall & Crop QnA System🌿")


# -----------------------
# Free-text Q&A
# -----------------------
if operation == "Free-text Q&A":
    st.header("Free-text Q&A")
    st.markdown("""
<div style="color:#bbb; font-size:15px;">
<b>Purpose:</b> Understand how a particular crop performs in a specific state or district over time.  
It provides total production, average area cultivated, and productivity details.  
<br><br>
<b>Example:</b> “Rice production in Tamil Nadu has consistently increased from 2000 to 2015 with an average productivity of 3.2 tonnes per hectare.”
</div>
""", unsafe_allow_html=True)

    query = st.text_input("Ask your question")
    topk = st.slider("Top evidence rows", 1, 8, 5)

    if st.button("Search"):
        q = (query or "").strip()
        if not q:
            st.warning("Please enter a query.")
        else:
            results = retrieve_tfidf(q, topk)
            if results:
                top = results[0]
                formatted = prettify_record(top["text"])
                result_block("Top Matching Result", f"<b>Score:</b> {top['score']:.3f}<br>{formatted}")
                df = pd.DataFrame([{"Rainfall": r["score"], "Evidence": prettify_record(r["text"])} for r in results])
                st.subheader("Top Evidence Rows")
                st.dataframe(df, use_container_width=True)
                sources_box(list(DATASET_LINKS.keys()))
            else:
                result_block("No Evidence Found", "No matching rows found.")
                sources_box(list(DATASET_LINKS.keys()))

# -----------------------
# Compare average rainfall & top crops
# -----------------------
elif operation == "Compare average rainfall & top crops":
    st.header("Compare average rainfall & top crops")
    st.markdown("""<div style="color:#bbb; font-size:15px;">
<b>Purpose:</b> Analyze the effect of rainfall on crop yield across regions or time periods.  
Correlates rainfall levels with production to understand climate influence.  
<br><br>
<b>Example:</b> “In Maharashtra, groundnut yield shows a strong positive correlation (r = 0.78) with annual rainfall between 2005–2015.”
</div>
""", unsafe_allow_html=True)


    # get available states dynamically
    def get_states():
        cols = ["state"]
        dfs = [completed, seasonal_district, rainfall_data, crop]
        states = set()
        for d in dfs:
            if d is not None and not d.empty:
                for c in d.columns:
                    if any(x in c.lower() for x in cols):
                        states.update(d[c].dropna().astype(str).str.strip().unique())
        return sorted(list(states))

    states = get_states()
    sx = st.selectbox("Select State X", [""] + states, index=0)
    sy = st.selectbox("Select State Y", [""] + states, index=0)
    n_years = st.number_input("Last N years (0 = all)", min_value=0, value=5)

    if st.button("Run comparison"):
        def get_state_annual(state):
            parts = []
            if completed is not None and not completed.empty and "state" in completed.columns and "annual" in completed.columns:
                parts.append(completed[["state","year","annual"]])
            if seasonal_district is not None and not seasonal_district.empty:
                tmp = seasonal_district.copy()
                if "avg_annual_rainfall" in tmp.columns:
                    tmp = tmp.rename(columns={"avg_annual_rainfall":"annual"})
                if "state" in tmp.columns and "annual" in tmp.columns:
                    parts.append(tmp[["state","year","annual"]])
            if parts:
                df = pd.concat(parts, ignore_index=True)
                df["year"] = pd.to_numeric(df["year"], errors="coerce")
                df = df[df["state"].str.upper() == state.upper()]
                df = df.dropna(subset=["annual","year"])
                return df
            return pd.DataFrame()

        df_x = get_state_annual(sx)
        df_y = get_state_annual(sy)

        def get_avg(df, n):
            if df.empty:
                return None
            yrs = sorted(df["year"].unique())
            if n > 0:
                yrs = yrs[-n:]
            subset = df[df["year"].isin(yrs)]
            return subset["annual"].mean() if not subset.empty else None

        avg_x = get_avg(df_x, n_years)
        avg_y = get_avg(df_y, n_years)

        def top_crops(state):
            if crop is None or crop.empty:
                return pd.DataFrame()
            df = crop.copy()
            if "state" not in df.columns or "production" not in df.columns:
                return pd.DataFrame()
            df = df[df["state"].str.upper() == state.upper()]
            if df.empty:
                return pd.DataFrame()
            df["production"] = pd.to_numeric(df["production"], errors="coerce")
            df["area"] = pd.to_numeric(df.get("area", 0), errors="coerce")
            agg = (
                df.groupby("crop")[["production","area"]]
                .sum()
                .reset_index()
                .sort_values("production", ascending=False)
                .head(5)
            )
            agg["production"] = agg["production"].round(2)
            agg["area"] = agg["area"].round(2)
            return agg

        left_col, right_col = st.columns(2)

        with left_col:
            st.subheader(f"🌿 {sx or 'State X'}")
            if avg_x is not None:
                st.success(f"**Average annual rainfall:** {avg_x:.2f} mm")
            else:
                st.warning("No rainfall data found.")
            tx = top_crops(sx)
            if not tx.empty:
                st.markdown("**Top 5 Crops by Production**")
                st.dataframe(tx, use_container_width=True, hide_index=True)
            else:
                st.info("No crop data available.")

        with right_col:
            st.subheader(f"🌾 {sy or 'State Y'}")
            if avg_y is not None:
                st.success(f"**Average annual rainfall:** {avg_y:.2f} mm")
            else:
                st.warning("No rainfall data found.")
            ty = top_crops(sy)
            if not ty.empty:
                st.markdown("**Top 5 Crops by Production**")
                st.dataframe(ty, use_container_width=True, hide_index=True)
            else:
                st.info("No crop data available.")

        if (avg_x is not None) and (avg_y is not None):
            fig, ax = plt.subplots(figsize=(5,3))
            ax.bar([sx, sy], [avg_x, avg_y], color=["#2E7D32", "#81C784"])
            ax.set_ylabel("Average annual rainfall (mm)")
            ax.set_title("Rainfall Comparison")
            st.pyplot(fig)

        sources_box(["Rainfall Data (district normals 1951-2000)", "Crop Data (district-wise seasonwise, 1997)"])

# -----------------------
# District max/min production
# -----------------------
elif operation == "State's District-wise max/min production":
    st.header("State's District-wise max/min production")
    st.markdown("""
<div style="color:#bbb; font-size:15px;">
<b>Purpose:</b> Compare districts or states with the highest and lowest crop production.  
Helps identify best-performing regions versus those needing intervention.  
<br><br>
<b>Example:</b> “Karnataka’s Mandya district shows the highest sugarcane production (120,000 tonnes), while Himachal Pradesh’s Una records the lowest.”
</div>
""", unsafe_allow_html=True)

    # Dropdowns for state & crop
    states = sorted(set(crop["state"].dropna().astype(str))) if not crop.empty else []
    sx = st.selectbox("State X (highest production)", [""] + states)
    sy = st.selectbox("State Y (lowest production)", [""] + states)
    crops_list = sorted(set(crop["crop"].dropna().astype(str))) if not crop.empty else []
    crop_z = st.selectbox("Crop", [""] + crops_list)

    if st.button("Compare"):
        if not crop_z or not sx or not sy:
            st.warning("Please select both states and a crop.")
        else:
            df_crop = crop.copy()
            df_crop["production"] = pd.to_numeric(df_crop["production"], errors="coerce")
            df_crop["area"] = pd.to_numeric(df_crop["area"], errors="coerce")

            # Add rainfall if possible
            rainfall_map = {}
            if rainfall_data is not None and not rainfall_data.empty:
                rain_tmp = rainfall_data.copy()
    # Find any likely 'annual' column (case-insensitive)
                rain_col = next((c for c in rain_tmp.columns if "annual" in c.lower()), None)
                if rain_col:
                    rain_tmp[rain_col] = pd.to_numeric(rain_tmp[rain_col], errors="coerce")
                    sub_col = next((c for c in rain_tmp.columns if "sub" in c.lower() or "district" in c.lower()), None)
                    if sub_col:
                        for _, row in rain_tmp.iterrows():
                            rain_key = str(row.get(sub_col, "")).strip().upper()
                            rainfall_map[rain_key] = row.get(rain_col)

            df_crop["rainfall"] = df_crop["district"].str.upper().map(rainfall_map)

            # Filter for selected crop
            df_crop = df_crop[df_crop["crop"].str.upper() == crop_z.upper()]
            if df_crop.empty:
                st.warning(f"No data found for crop: {crop_z}")
            else:
                left_col, right_col = st.columns(2)

                # Helper to format table
                def fmt_table(df):
                    df = df.copy()
                    df["production"] = df["production"].round(2)
                    df["area"] = df["area"].round(2)
                    df["rainfall"] = df["rainfall"].round(2)
                    df = df.rename(columns={
                        "district": "District",
                        "production": "Production (tonnes)",
                        "area": "Area (ha)",
                        "rainfall": "Rainfall (mm)"
                    })
                    return df

                # LEFT — Highest production state
                with left_col:
                    df_high = df_crop[df_crop["state"].str.upper() == sx.upper()]
                    if not df_high.empty:
                        top5 = (
                            df_high.groupby("district")[["production", "area", "rainfall"]]
                            .mean(numeric_only=True)
                            .reset_index()
                            .sort_values("production", ascending=False)
                            .head(5)
                        )
                        st.subheader(f"🌾 {sx} — Top 5 Districts for {crop_z}")
                        st.dataframe(fmt_table(top5), use_container_width=True, hide_index=True)
                        total_prod = df_high["production"].sum()
                        total_area = df_high["area"].sum()
                        avg_rain = df_high["rainfall"].mean()
                        st.success(
                            f"**{sx}** produces **{total_prod:,.0f} tonnes** of {crop_z} "
                            f"across **{total_area:,.0f} ha** (avg rainfall: {avg_rain:.2f} mm)."
                        )
                    else:
                        st.warning(f"No data for {crop_z} in {sx}.")

                # RIGHT — Lowest production state
                with right_col:
                    df_low = df_crop[df_crop["state"].str.upper() == sy.upper()]
                    if not df_low.empty:
                        bottom5 = (
                            df_low.groupby("district")[["production", "area", "rainfall"]]
                            .mean(numeric_only=True)
                            .reset_index()
                            .sort_values("production", ascending=True)
                            .head(5)
                        )
                        st.subheader(f"🌱 {sy} — Bottom 5 Districts for {crop_z}")
                        st.dataframe(fmt_table(bottom5), use_container_width=True, hide_index=True)
                        total_prod = df_low["production"].sum()
                        total_area = df_low["area"].sum()
                        avg_rain = df_low["rainfall"].mean()
                        st.info(
                            f"**{sy}** produces only **{total_prod:,.0f} tonnes** of {crop_z} "
                            f"across **{total_area:,.0f} ha** (avg rainfall: {avg_rain:.2f} mm)."
                        )
                    else:
                        st.warning(f"No data for {crop_z} in {sy}.")

                # Combined chart
                if not df_high.empty and not df_low.empty:
                    total_high = df_high["production"].sum()
                    total_low = df_low["production"].sum()

                    fig, ax = plt.subplots(figsize=(5,3))
                    ax.bar([sx, sy], [total_high, total_low], color=["#2E7D32", "#81C784"])
                    ax.set_ylabel("Total production (tonnes)")
                    ax.set_title(f"Total {crop_z} Production Comparison")
                    for i, val in enumerate([total_high, total_low]):
                        ax.text(i, val, f"{val:,.0f}", ha="center", va="bottom", fontsize=10)
                    st.pyplot(fig)

                sources_box([
                    "Crop Production Data (district-wise, seasonwise 1997)",
                    "Rainfall Data (district normals 1951–2000)"
                ])


# -----------------------
# Production trend & rainfall correlation
# -----------------------
elif operation == "Production trend & rainfall correlation":
    st.header("Production trend & rainfall correlation")
    st.markdown("""
<div style="color:#bbb; font-size:15px;">
<b>Purpose:</b> Visualize how production of a selected crop changes over recent years and how it aligns with average annual rainfall in the same period.  
Highlights correlation between rainfall patterns and crop performance.  
<br><br>
<b>Example:</b> “For Arhar/Tur in Andhra Pradesh and Kerala (2006–2015), production peaks correspond closely with years of higher rainfall.”
</div>
""", unsafe_allow_html=True)

    # Dynamically populate dropdowns
    crops = sorted(set(crop["crop"].dropna().astype(str))) if crop is not None and not crop.empty else []
    states = sorted(set(crop["state"].dropna().astype(str))) if crop is not None and not crop.empty else []

    crop_type = st.selectbox("Select Crop", [""] + crops)
    region_states = st.multiselect("Select State(s)/Region(s)", options=states)
    years_window = st.number_input("Years window (last N years)", min_value=1, value=10)

    if st.button("Analyze"):
        if crop is None or crop.empty:
            result_block("No crop data", "Crop dataset missing.")
            sources_box(["Crop Data (district-wise seasonwise, 1997)"])
        elif not crop_type or not region_states:
            st.warning("Please select a crop and at least one state.")
        else:
            df = crop.copy()

            # normalize columns safely
            year_col = next((c for c in df.columns if "year" in c.lower()), None)
            prod_col = next((c for c in df.columns if "prod" in c.lower()), None)

            if year_col:
                df["year"] = pd.to_numeric(df[year_col], errors="coerce")
            else:
                df["year"] = pd.to_numeric(df.get("year", pd.Series()), errors="coerce")

            if prod_col:
                df["production"] = pd.to_numeric(df[prod_col], errors="coerce")
            else:
                df["production"] = pd.to_numeric(df.get("production", pd.Series()), errors="coerce")

            # Determine valid range
            if df["year"].dropna().empty:
                st.warning("No valid year data in dataset.")
                st.stop()

            maxy = int(df["year"].max())
            years_range = list(range(maxy - years_window + 1, maxy + 1))

            # Filter by crop & states
            df_reg = df[
                df["state"].str.upper().isin([s.upper() for s in region_states])
                & df["crop"].str.contains(crop_type, case=False, na=False)
                & df["year"].isin(years_range)
            ].dropna(subset=["year", "production"])

            if df_reg.empty:
                st.warning("No matching production data for the selected filters.")
                st.stop()

            # ✅ Aggregate production correctly across all states and years
            agg_df = (
                df_reg.groupby(["state", "year"], as_index=False)["production"]
                .sum(numeric_only=True)
            )

            prod_ts = (
                agg_df.groupby("year")["production"]
                .sum(numeric_only=True)
                .reindex(years_range, fill_value=0)
            )

            # ✅ Rainfall data handling
            rain_ts = None
            if completed is not None and not completed.empty and "annual" in completed.columns:
                r = completed.copy()
                r["year"] = pd.to_numeric(r["year"], errors="coerce")
                r = r[
                    r["state"].str.upper().isin([s.upper() for s in region_states])
                    & r["year"].isin(years_range)
                ]
                rain_ts = (
                    r.groupby("year")["annual"]
                    .mean(numeric_only=True)
                    .reindex(years_range)
                )
            elif seasonal_district is not None and not seasonal_district.empty:
                tmp = seasonal_district.copy()
                if "avg_annual_rainfall" in tmp.columns:
                    tmp = tmp.rename(columns={"avg_annual_rainfall": "annual"})
                tmp["year"] = pd.to_numeric(tmp["year"], errors="coerce")
                tmp = tmp[
                    tmp["state"].str.upper().isin([s.upper() for s in region_states])
                    & tmp["year"].isin(years_range)
                ]
                rain_ts = (
                    tmp.groupby("year")["annual"]
                    .mean(numeric_only=True)
                    .reindex(years_range)
                )

            # ✅ Summary block
            title = f"Production trend: {crop_type} — {', '.join(region_states)}"
            latest_prod = prod_ts.iloc[-1] if len(prod_ts) > 0 else 0
            avg_prod = prod_ts.mean()
            total_prod = prod_ts.sum()
            body = (
                f"Period: {years_range[0]}–{years_range[-1]}. "
                f"Latest total production: {latest_prod:,.0f} tonnes. "
                f"Average yearly production: {avg_prod:,.0f} tonnes. "
                f"Total cumulative production: {total_prod:,.0f} tonnes."
            )
            result_block(title, body)

            # ✅ Visualization
            st.subheader("📈 Time Series Trends")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Crop Production (yearly + smoothed)**")
                df_plot = pd.DataFrame({"Production": prod_ts})
                df_plot["Smoothed (3-year avg)"] = df_plot["Production"].rolling(window=3, min_periods=1).mean()
                st.line_chart(df_plot, use_container_width=True)

            with col2:
                if rain_ts is not None:
                    st.markdown("**Average Annual Rainfall (mm)**")
                    st.line_chart(rain_ts, use_container_width=True)
                else:
                    st.info("No rainfall data available for selected region.")

            # ✅ Correlation
            if rain_ts is not None:
                joined = pd.DataFrame(
                    {"production": prod_ts.values, "rainfall": rain_ts.values},
                    index=years_range
                ).dropna()
                if len(joined) >= 2:
                    corr = joined["production"].corr(joined["rainfall"])
                    st.markdown(f"**📊 Pearson correlation between production and rainfall:** `{corr:.3f}`")
                else:
                    st.write("Not enough overlapping data for correlation.")

            sources_box([
                "Crop Production Data (district-wise, seasonwise 1997)",
                "Rainfall Data (district normals 1951–2000)",
                "Seasonal/Daily District Rainfall"
            ])


# -----------------------
# Policy advisor
# -----------------------
elif operation == "Policy advisor: top 3 arguments":
    st.header("Policy advisor — top 3 arguments")
    st.markdown("""
<div style="color:#bbb; font-size:15px;">
<b>Purpose:</b> Helps policymakers decide which crop to promote by comparing evidence-based insights across states — including production, area, and climate data.  
<br><br>
<b>Example:</b> “Between Arecanut and Bajra, Arecanut shows higher productivity and better rainfall resilience in southern states like Kerala and Karnataka.”
</div>
""", unsafe_allow_html=True)

    # 🌾 Dropdown inputs
    crops_list = sorted(set(crop["crop"].dropna().astype(str))) if not crop.empty else []
    states = sorted(set(crop["state"].dropna().astype(str))) if not crop.empty else []

    crop_a = st.selectbox("Crop A (promote)", [""] + crops_list)
    crop_b = st.selectbox("Crop B (compare)", [""] + crops_list)
    region_states = st.multiselect("Select Region/States", options=states)

    if st.button("Generate"):
        if not crop_a or not crop_b or not region_states:
            st.warning("Please select both crops and at least one region/state.")
        else:
            retr = retrieve_tfidf(
                f"policy promote {crop_a} over {crop_b} {' '.join(region_states)}", 6
            )

            if not retr:
                result_block("No evidence found", "No data supports this comparison.")
                sources_box(list(DATASET_LINKS.keys()))
                st.stop()

            # ✅ Clear result summary
            result_block(
                f"Recommendation Summary: Promote {crop_a} over {crop_b}",
                f"Regions analyzed: {', '.join(region_states)}.<br>"
                f"The top supporting evidence for promoting <b>{crop_a}</b> is shown below.",
            )

            formatted = []
            for r in retr[:3]:
                text = prettify_record(r["text"])
                formatted.append({
                    "Relevance Score": round(r["score"], 3),
                    "Evidence Summary": text
                })

            df = pd.DataFrame(formatted)

            st.subheader("📊 Top 3 Supporting Evidences")
            st.dataframe(df, use_container_width=True, hide_index=True)

            # 🌿 Numeric summary comparison
            df_crop = crop.copy()
            df_crop["production"] = pd.to_numeric(df_crop["production"], errors="coerce")

            df_a = df_crop[
                (df_crop["crop"].str.upper() == crop_a.upper()) &
                (df_crop["state"].str.upper().isin([s.upper() for s in region_states]))
            ]
            df_b = df_crop[
                (df_crop["crop"].str.upper() == crop_b.upper()) &
                (df_crop["state"].str.upper().isin([s.upper() for s in region_states]))
            ]

            if not df_a.empty and not df_b.empty:
                avg_a = df_a["production"].mean()
                avg_b = df_b["production"].mean()
                diff = avg_a - avg_b

                if diff > 0:
                    st.success(
                        f"✅ On average, **{crop_a}** shows a higher production ({avg_a:,.0f} tonnes) "
                        f"than **{crop_b}** ({avg_b:,.0f} tonnes) across the selected regions."
                    )
                else:
                    st.info(
                        f"ℹ️ On average, **{crop_b}** performs slightly better ({avg_b:,.0f} tonnes) "
                        f"than **{crop_a}** ({avg_a:,.0f} tonnes) in these regions."
                    )
            else:
                st.info("Not enough production data for numeric comparison.")

            sources_box([
                "Crop Production Data (district-wise, seasonwise 1997)",
                "Rainfall Data (district normals 1951–2000)"
            ])



st.markdown("---")
st.caption("Design: left panel controls; right panel shows results first, then tables/charts, then sources.")

