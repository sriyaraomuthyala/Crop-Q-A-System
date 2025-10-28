# app.py
# Professional neutral UI: left-side operations + right-side main content
# Results appear first (large green summary), then table/chart, then a small sources box.
# No animations.

import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

st.set_page_config(page_title="Rainfall & Crop QnA — Professional", layout="wide")

# -----------------------
# Paths & dataset links
# -----------------------
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")   # put your CSVs here
INDEX_DIR = os.path.join(BASE_DIR, "qna_index")
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
# Safe CSV load + normalize columns
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

# normalize 'annual' column name candidates
for df in (completed, seasonal_district, subdivision):
    if df is None or df.empty:
        continue
    for c in list(df.columns):
        if c.lower() in ["year_avg_rainfall","avg_annual_rainfall","annual","yearavg","year_avg"]:
            df.rename(columns={c:"annual"}, inplace=True)

# -----------------------
# heuristics to find key columns
# -----------------------
def find_col(df, candidates):
    if df is None or df.empty:
        return None
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return c
    # case-insensitive fallback
    lower_map = {col.lower(): col for col in cols}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None

state_col_candidates = ["state","state_name"]
district_col_candidates = ["district","subdivision","sub_division","sub-division"]
crop_col_candidates = ["crop","crop_name"]
year_col_candidates = ["year","yr"]
prod_col_candidates = ["production","prod","production_quantity","production_tonnes","yield"]

# try detect columns in crop dataset first (most relevant)
state_col = find_col(crop, state_col_candidates) or find_col(completed, state_col_candidates) or find_col(seasonal_district, state_col_candidates)
district_col = find_col(crop, district_col_candidates) or find_col(rainfall_data, district_col_candidates) or find_col(completed, district_col_candidates)
crop_col = find_col(crop, crop_col_candidates)
year_col = find_col(crop, year_col_candidates) or find_col(completed, year_col_candidates) or find_col(rainfall_data, year_col_candidates)
prod_col = find_col(crop, prod_col_candidates) or find_col(crop, ["production"])

# -----------------------
# Prepare dropdown values (guided)
# -----------------------
def unique_sorted(df, col):
    if df is None or df.empty or col not in df.columns:
        return []
    vals = df[col].dropna().astype(str).str.strip().unique().tolist()
    return sorted([v for v in vals if v])

states = sorted(set(unique_sorted(completed, "state") + unique_sorted(seasonal_district, "state") + unique_sorted(rainfall_data, "state") + unique_sorted(crop, state_col if state_col else "state")))
districts = sorted(set(unique_sorted(completed, "district") + unique_sorted(rainfall_data, "subdivision") + unique_sorted(crop, district_col if district_col else "district")))
crops = sorted(set(unique_sorted(crop, crop_col if crop_col else "crop")))
years = sorted(set([int(y) for y in unique_sorted(completed, "year") if str(y).isdigit()] + [int(y) for y in unique_sorted(crop, year_col if year_col else "year") if str(y).isdigit()] + [int(y) for y in unique_sorted(rainfall_data, "year") if str(y).isdigit()]), reverse=True)

# -----------------------
# TF-IDF fallback (safe)
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
# Sidebar (left panel): operations + filters + dataset links
# -----------------------
with st.sidebar:
    st.title("Operations & Filters")
    operation = st.radio("Choose operation", [
        "Free-text Q&A",
        "Compare average rainfall & top crops",
        "District max/min production",
        "Production trend & rainfall correlation",
        "Policy advisor: top 3 arguments"
    ])
    st.markdown("---")
    st.subheader("Guided filters")
    sel_state = st.selectbox("State (optional)", [""] + states, index=0)
    sel_district = st.selectbox("District (optional)", [""] + districts, index=0)
    sel_crop = st.selectbox("Crop (optional)", [""] + crops, index=0)
    sel_year = st.selectbox("Year (optional)", [""] + [str(y) for y in years], index=0)
    st.markdown("---")
    st.subheader("Dataset sources")
    for name, url in DATASET_LINKS.items():
        st.markdown(f"- [{name}]({url})")
    st.caption("Dropdowns reflect what's present in ./data/*.csv")

# -----------------------
# Helpers for UI: large result block and sources box
# -----------------------
def result_block(title, html_body):
    """Large green-accented result block (title plain text; html_body allowed HTML)"""
    st.markdown(f"""
    <div style="background:#E8F5E9; border-left:4px solid #2E7D32; padding:20px; border-radius:8px; margin-bottom:10px;">
      <div style="font-size:20px; color:#1B5E20; font-weight:700;">{title}</div>
      <div style="font-size:16px; color:#155724; margin-top:8px;">{html_body}</div>
    </div>
    """, unsafe_allow_html=True)

def sources_box(names):
    lines = ""
    for n in names:
        url = DATASET_LINKS.get(n)
        if url:
            lines += f"- <a href='{url}' target='_blank'>{n}</a><br>"
        else:
            lines += f"- {n}<br>"
    st.markdown(f"""
    <div style="background:#F3F4F6; padding:10px; border-radius:6px; font-size:13px; color:#374151; margin-top:12px;">
      <strong>Sources</strong><br>{lines}
    </div>
    """, unsafe_allow_html=True)

# -----------------------
# Main area header
# -----------------------
st.title("Rainfall & Crop QnA — Professional View")
st.write("Results are shown first (large colored block), then supporting tables/charts, then a compact sources box below. No animations.")

# -----------------------
# Operation: Free-text Q&A (right content)
# -----------------------
if operation == "Free-text Q&A":
    st.header("Free-text Q&A (guided)")
    query = st.text_input("Question (or use the left-side filters)")
    topk = st.slider("Top evidence rows", 1, 8, 5)
    if st.button("Search"):
        q = (query or "").strip()
        if sel_state: q += " " + sel_state
        if sel_district: q += " " + sel_district
        if sel_crop: q += " " + sel_crop
        if sel_year: q += " " + sel_year
        results = retrieve_tfidf(q, topk)
        if results:
            top = results[0]
            # RESULTS FIRST
            result_block("Search result (top evidence)", f"<b>Score:</b> {top['score']:.3f} — {top['text'][:600]}...")
            # then evidence table
            df = pd.DataFrame([{"score": r["score"], "text": r["text"]} for r in results])
            st.subheader("Evidence rows (top matches)")
            st.dataframe(df)
            # then sources
            sources_box(list(DATASET_LINKS.keys()))
        else:
            result_block("No evidence found", "No matching rows. Try changing filters or uploading additional datasets.")
            sources_box(list(DATASET_LINKS.keys()))

# -----------------------
# Operation: Compare avg rainfall & top crops
# -----------------------
elif operation == "Compare average rainfall & top crops":
    st.header("Compare average rainfall & top crops")
    sx = st.selectbox("State X", [""] + states, index=0)
    sy = st.selectbox("State Y", [""] + states, index=0)
    n_years = st.number_input("Last N years (0 = all)", min_value=0, value=5)
    m_top = st.number_input("Top M crops", min_value=1, value=5)
    if st.button("Run comparison"):
        # get annual series for each state (prefer completed -> seasonal)
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
        df_x = get_state_annual(sx) if sx else pd.DataFrame()
        df_y = get_state_annual(sy) if sy else pd.DataFrame()
        def last_n(df, n):
            if df.empty: return []
            yrs = sorted(df["year"].unique())
            return yrs[-n:] if n>0 else yrs
        yrs_x = last_n(df_x, n_years)
        yrs_y = last_n(df_y, n_years)
        years_common = sorted(list(set(yrs_x).intersection(set(yrs_y)))) if yrs_x and yrs_y else sorted(list(set(yrs_x + yrs_y)))
        if not years_common:
            years_common = sorted(list(set(yrs_x + yrs_y)))
        avg_x = df_x[df_x["year"].isin(years_common)]["annual"].mean() if not df_x.empty else None
        avg_y = df_y[df_y["year"].isin(years_common)]["annual"].mean() if not df_y.empty else None
        # RESULTS FIRST (large block)
        title = f"Average annual rainfall — {sx} vs {sy}"
        body_parts = []
        if avg_x is not None:
            body_parts.append(f"<b>{sx}</b>: {avg_x:.1f} mm (avg across {len(years_common)} years)")
        else:
            body_parts.append(f"<b>{sx}</b>: no annual rainfall data in CSVs")
        if avg_y is not None:
            body_parts.append(f"<b>{sy}</b>: {avg_y:.1f} mm (avg across {len(years_common)} years)")
        else:
            body_parts.append(f"<b>{sy}</b>: no annual rainfall data in CSVs")
        result_block(title, "<br>".join(body_parts))
        # then visuals / tables
        if (avg_x is not None) and (avg_y is not None):
            fig, ax = plt.subplots(figsize=(6,3))
            ax.bar([sx, sy], [avg_x, avg_y], color=["#2E7D32", "#66BB6A"])
            ax.set_ylabel("Avg annual rainfall (mm)")
            ax.set_title("Average annual rainfall")
            st.pyplot(fig)
        # top crops table
        if crop is not None and not crop.empty and prod_col in crop.columns:
            cdf = crop.copy()
            if year_col in cdf.columns and years_common:
                cdf[year_col] = pd.to_numeric(cdf[year_col], errors="coerce")
                cdf = cdf[cdf[year_col].isin(years_common)]
            def top_m_for(state):
                df_s = cdf[cdf["state"].str.upper() == state.upper()]
                if df_s.empty:
                    return pd.DataFrame()
                df_s[prod_col] = pd.to_numeric(df_s[prod_col], errors="coerce")
                agg = df_s.groupby(crop_col if crop_col in df_s.columns else "crop")[prod_col].sum().reset_index().sort_values(prod_col, ascending=False).head(m_top)
                return agg
            tx = top_m_for(sx)
            ty = top_m_for(sy)
            st.subheader("Top crops (by production)")
            left, right = st.columns(2)
            with left:
                st.markdown(f"**{sx}**")
                st.dataframe(tx if not tx.empty else pd.DataFrame({"note":[f"No crop rows for {sx}"]}))
            with right:
                st.markdown(f"**{sy}**")
                st.dataframe(ty if not ty.empty else pd.DataFrame({"note":[f"No crop rows for {sy}"]}))
        else:
            st.write("Crop dataset not available or missing production column; showing retrieval fallback evidence.")
            retr = retrieve_tfidf(f"top crops {sx} {sy}", 6)
            if retr:
                st.subheader("Fallback evidence")
                st.dataframe(pd.DataFrame(retr))
        # finally sources
        sources_box(list(DATASET_LINKS.keys()))

# -----------------------
# District max/min production
# -----------------------
elif operation == "District max/min production":
    st.header("District max/min production")
    sx = st.selectbox("State X (find highest)", [""] + states)
    sy = st.selectbox("State Y (find lowest)", [""] + states)
    crop_z = st.selectbox("Crop", [""] + crops)
    if st.button("Compare"):
        if crop is None or crop.empty or prod_col is None:
            result_block("No crop production data", "Crop CSV missing or production column not detected. Using retrieval fallback.")
            retr = retrieve_tfidf(f"{crop_z} production {sx} {sy}", 6)
            st.dataframe(pd.DataFrame(retr))
            sources_box(list(DATASET_LINKS.keys()))
        else:
            df = crop.copy()
            df[prod_col] = pd.to_numeric(df[prod_col], errors="coerce")
            df[year_col if year_col in df.columns else "year"] = pd.to_numeric(df[year_col if year_col in df.columns else "year"], errors="coerce")
            sx_df = df[df["state"].str.upper() == sx.upper()] if sx else pd.DataFrame()
            sy_df = df[df["state"].str.upper() == sy.upper()] if sy else pd.DataFrame()
            if sx_df.empty or sy_df.empty:
                result_block("No exact crop rows", "No crop rows for selected states. Showing fallback evidence.")
                retr = retrieve_tfidf(f"{crop_z} production {sx} {sy}", 6)
                st.dataframe(pd.DataFrame(retr))
                sources_box(list(DATASET_LINKS.keys()))
            else:
                rx = int(sx_df["year"].max())
                ry = int(sy_df["year"].max())
                sx_recent = sx_df[(sx_df["year"]==rx) & (sx_df[crop_col].str.contains(crop_z, case=False, na=False))] if crop_col in sx_df.columns else sx_df[sx_df["crop"].str.contains(crop_z, case=False, na=False)]
                sy_recent = sy_df[(sy_df["year"]==ry) & (sy_df[crop_col].str.contains(crop_z, case=False, na=False))] if crop_col in sy_df.columns else sy_df[sy_df["crop"].str.contains(crop_z, case=False, na=False)]
                # build summary first
                if not sx_recent.empty:
                    agg_sx = sx_recent.groupby("district")[prod_col].sum().reset_index().dropna()
                    if not agg_sx.empty:
                        best = agg_sx.loc[agg_sx[prod_col].idxmax()].to_dict()
                        result_block("Highest producing district (State X)", f"{best['district']} produced {best[prod_col]:.0f} units in {rx}")
                        st.dataframe(agg_sx.sort_values(prod_col, ascending=False).head(10))
                else:
                    result_block("No matching rows (State X)", f"No matching rows found for {crop_z} in {sx} for year {rx}")
                if not sy_recent.empty:
                    agg_sy = sy_recent.groupby("district")[prod_col].sum().reset_index().dropna()
                    if not agg_sy.empty:
                        worst = agg_sy.loc[agg_sy[prod_col].idxmin()].to_dict()
                        st.subheader("Comparison (State Y)")
                        result_block("Lowest producing district (State Y)", f"{worst['district']} produced {worst[prod_col]:.0f} units in {ry}")
                        st.dataframe(agg_sy.sort_values(prod_col, ascending=True).head(10))
                else:
                    st.write(f"No matching rows for {crop_z} in {sy} for year {ry}.")
                sources_box(["Crop Data (district-wise seasonwise, 1997)"])

# -----------------------
# Production trend & rainfall correlation
# -----------------------
elif operation == "Production trend & rainfall correlation":
    st.header("Production trend & rainfall correlation")
    crop_type = st.selectbox("Crop", [""] + crops)
    region_states = st.multiselect("Region states", options=states)
    years_window = st.number_input("Years window (last N years)", min_value=1, value=10)
    if st.button("Analyze"):
        if crop is None or crop.empty:
            result_block("No crop data", "Crop dataset missing.")
            sources_box(["Crop Data (district-wise seasonwise, 1997)"])
        else:
            df = crop.copy()
            if year_col in df.columns:
                df["year"] = pd.to_numeric(df[year_col], errors="coerce")
            else:
                df["year"] = pd.to_numeric(df.get("year", pd.Series()), errors="coerce")
            df["production"] = pd.to_numeric(df.get(prod_col if prod_col in df.columns else "production", pd.Series()), errors="coerce")
            maxy = int(df["year"].max())
            years_range = list(range(maxy - years_window + 1, maxy + 1))
            df_reg = df[df["state"].str.upper().isin([s.upper() for s in region_states]) & df["crop"].str.contains(crop_type, case=False, na=False) & df["year"].isin(years_range)]
            prod_ts = df_reg.groupby("year")["production"].sum().reindex(years_range, fill_value=0)
            # rainfall
            rain_ts = None
            if (completed is not None and not completed.empty and "annual" in completed.columns):
                r = completed.copy()
                r["year"] = pd.to_numeric(r["year"], errors="coerce")
                r = r[r["state"].str.upper().isin([s.upper() for s in region_states]) & r["year"].isin(years_range)]
                rain_ts = r.groupby("year")["annual"].mean().reindex(years_range)
            elif (seasonal_district is not None and not seasonal_district.empty):
                tmp = seasonal_district.copy()
                if "avg_annual_rainfall" in tmp.columns:
                    tmp = tmp.rename(columns={"avg_annual_rainfall":"annual"})
                tmp["year"] = pd.to_numeric(tmp["year"], errors="coerce")
                tmp = tmp[tmp["state"].str.upper().isin([s.upper() for s in region_states]) & tmp["year"].isin(years_range)]
                rain_ts = tmp.groupby("year")["annual"].mean().reindex(years_range)
            # result summary
            title = f"Production trend: {crop_type} — {', '.join(region_states)}"
            latest_prod = int(prod_ts.values[-1]) if len(prod_ts)>0 else 0
            body = f"Period: {years_range[0]}–{years_range[-1]}. Latest total production: {latest_prod} units."
            result_block(title, body)
            # charts and correlation
            st.subheader("Time series")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("Production (yearly)")
                st.line_chart(prod_ts)
            with col2:
                if rain_ts is not None:
                    st.markdown("Average annual rainfall (region mean)")
                    st.line_chart(rain_ts)
            if rain_ts is not None:
                joined = pd.DataFrame({"production": prod_ts.values, "rainfall": rain_ts.values}, index=years_range).dropna()
                if len(joined) >= 2:
                    corr = joined["production"].corr(joined["rainfall"])
                    st.markdown(f"**Pearson correlation:** {corr:.3f}")
                else:
                    st.write("Not enough overlapping data.")
            sources_box(["Seasonal/Daily District Rainfall", "Rainfall Data (district normals 1951-2000)"])

# -----------------------
# Policy advisor
# -----------------------
elif operation == "Policy advisor: top 3 arguments":
    st.header("Policy advisor — top 3 data-backed arguments")
    crop_a = st.selectbox("Crop A (promote)", [""] + crops)
    crop_b = st.selectbox("Crop B (compare)", [""] + crops)
    region_states = st.multiselect("Region states", options=states)
    n_years = st.number_input("Last N years (0=all)", min_value=0, value=10)
    if st.button("Generate"):
        if crop is None or crop.empty:
            result_block("No crop data", "Crop dataset missing.")
            sources_box(["Crop Data (district-wise seasonwise, 1997)"])
        else:
            df = crop.copy()
            if year_col in df.columns:
                df["year"] = pd.to_numeric(df[year_col], errors="coerce")
            df["production"] = pd.to_numeric(df.get(prod_col if prod_col in df.columns else "production", pd.Series()), errors="coerce")
            maxy = int(df["year"].max()) if not df["year"].isnull().all() else None
            years = list(range(maxy - n_years + 1, maxy+1)) if (n_years>0 and maxy) else sorted(df["year"].dropna().unique())
            sub = df[df["state"].str.upper().isin([s.upper() for s in region_states]) & df["year"].isin(years)]
            def stats(name):
                s = sub[sub["crop"].str.contains(name, case=False, na=False)]
                ts = s.groupby("year")["production"].sum().reindex(years, fill_value=0)
                slope = 0.0
                if len(ts.dropna()) >= 2:
                    x = np.arange(len(ts)); y = ts.fillna(0).values
                    m, c = np.linalg.lstsq(np.vstack([x, np.ones_like(x)]).T, y, rcond=None)[0]
                    slope = float(m)
                return {"mean": float(ts.mean()) if len(ts)>0 else 0.0, "slope": slope}
            A = stats(crop_a); B = stats(crop_b)
            # rainfall exposure
            avg_rain = None
            if (completed is not None and not completed.empty and "annual" in completed.columns):
                r = completed.copy()
                r["year"] = pd.to_numeric(r["year"], errors="coerce")
                rsel = r[r["state"].str.upper().isin([s.upper() for s in region_states]) & r["year"].isin(years)]
                if not rsel.empty:
                    avg_rain = float(rsel["annual"].mean())
            # result summary first
            title = f"Top 3 arguments: promote {crop_a} over {crop_b}"
            body = f"Period: {years[0] if years else 'N/A'}–{years[-1] if years else 'N/A'}. {crop_a} mean={A['mean']:.1f}, {crop_b} mean={B['mean']:.1f}."
            result_block(title, body)
            # arguments
            args = []
            if avg_rain is not None and avg_rain < 800:
                args.append(("Water resilience", f"Average annual rainfall ≈ {avg_rain:.0f} mm — {crop_a} lowers irrigation needs vs {crop_b}."))
            else:
                args.append(("Comparative production", f"{crop_a} mean {A['mean']:.1f} vs {crop_b} {B['mean']:.1f}."))
            args.append(("Trend evidence", f"{crop_a} slope {A['slope']:.3f} vs {crop_b} slope {B['slope']:.3f}."))
            args.append(("Risk diversification", "Promoting drought-tolerant crops reduces income volatility during dry years."))
            for t, ev in args:
                st.markdown(f"**{t}**")
                st.write(ev)
            st.subheader("Supporting stats")
            st.write(f"{crop_a}: mean={A['mean']:.1f}, slope={A['slope']:.3f}")
            st.write(f"{crop_b}: mean={B['mean']:.1f}, slope={B['slope']:.3f}")
            sources_box(["Crop Data (district-wise seasonwise, 1997)", "Rainfall Data (district normals 1951-2000)"])

# footer
st.markdown("---")
st.caption("Design: left panel controls; right panel shows results first, then tables/charts, then sources. No animations; large green summary blocks for results.")
