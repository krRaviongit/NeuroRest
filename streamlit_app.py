import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import streamlit as st




def build_theme_css(mode: str, accent: str) -> str:
	is_dark = mode == "Dark"
	bg = '#0b1220' if is_dark else '#f6f7fb'
	surface = '#101827' if is_dark else '#ffffff'
	border = '#1f2a44' if is_dark else '#e5e7eb'
	text = '#e5e7eb' if is_dark else '#0f172a'
	muted = '#9aa4b2' if is_dark else '#64748b'
	badge_bg = '#0b1220' if is_dark else '#eef2ff'
	badge_text = '#a5b4fc' if is_dark else '#3730a3'
	badge_border = '#1f2a44' if is_dark else '#c7d2fe'
	grad = 'linear-gradient(180deg, #0e172a 0%, #0b1220 100%)' if is_dark else 'linear-gradient(180deg, #fafbff 0%, #ffffff 100%)'
	css = """
	<style>
	@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
	html, body, .stApp {{ background: {bg}; color: {text}; font-family: 'Poppins', ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }}
	.block-container {{ padding-top: 1.5rem; max-width: 1200px; }}
	.app-title {{ margin-top: .75rem; margin-bottom: .5rem; }}
	.app-title h1 {{ font-size: 2rem; font-weight: 700; margin: 0; letter-spacing: .2px; }}
	.card {{ background: {surface}; border-radius: 14px; padding: 1rem 1.25rem; box-shadow: 0 6px 20px rgba(2,6,23,.12); border: 1px solid {border}; }}
	.card:hover {{ box-shadow: 0 12px 28px rgba(2,6,23,.18); transition: box-shadow .25s ease; }}
	.section-title {{ font-weight: 700; font-size: 1.05rem; margin: .2rem 0 .6rem; display:flex; align-items:center; gap:.5rem; }}
	.section-title:before {{ content:''; display:inline-block; width:6px; height:18px; border-radius:3px; background:{accent}; }}
	.h-sep {{ border-top: 1px solid {border}; margin: .75rem 0; }}
	.small {{ font-size: .875rem; color: {muted}; }}
	.muted {{ color: {muted}; }}
	.hero {{ border: 1px solid {border}; background: linear-gradient(135deg, {accent}26, {accent}1A); border-radius: 16px; padding: 1.1rem 1.3rem; margin-bottom: .75rem; }}
	.hero h1 {{ margin: 0; font-size: 1.32rem; color: {text}; letter-spacing: .2px; }}
	.badges {{ display: flex; gap: .5rem; flex-wrap: wrap; margin-top: .5rem; }}
	.badge {{ background: {badge_bg}; color: {badge_text}; padding: .25rem .5rem; border-radius: 999px; border: 1px solid {badge_border}; font-size: .8rem; }}
	.result {{ display:flex; align-items:center; gap:.6rem; padding:.75rem 1rem; border:1px solid {border}; border-radius:12px; background: {grad}; }}
	.result .label {{ font-weight:600; color:{muted}; }}
	.result .value {{ font-weight:700; color:{text}; }}
	/* Navigation styling */
	.nav-section {{ margin-bottom: 1.5rem; }}
	.nav-section h3 {{ font-size: 1.1rem; font-weight: 700; color: {accent}; margin-bottom: .75rem; text-transform: uppercase; letter-spacing: .5px; }}
	.nav-item {{ display: flex; align-items: center; gap: .75rem; padding: .75rem 1rem; margin-bottom: .5rem; border-radius: 10px; border: 1px solid {border}; background: {surface}; transition: all .2s ease; cursor: pointer; }}
	.nav-item:hover {{ border-color: {accent}; background: {accent}0A; }}
	.nav-item.active {{ border-color: {accent}; background: {accent}1A; }}
	.nav-item .icon {{ font-size: 1.25rem; }}
	.nav-item .label {{ font-weight: 500; color: {text}; }}
	/* Inputs & buttons like React UI */
	.stSelectbox>div>div, .stNumberInput>div>div>input, .stTextInput>div>div>input {{ border-radius:10px; border:1px solid {border}; background:{surface}; color:{text}; }}
	.stSlider>div div[role='slider'] {{ background:{accent}; box-shadow: 0 0 0 4px {accent}33; }}
	.stSlider>div [data-baseweb='slider']>div>div {{ background:{accent}55; }}
	.stButton>button {{ background:{accent}; color:#fff; border-radius:10px; padding:.5rem 1rem; border:1px solid transparent; }}
	.stButton>button:hover {{ filter:brightness(1.05); box-shadow:0 6px 18px {accent}55; }}
	.stButton.secondary>button {{ background:transparent; color:{text}; border-color:{border}; }}
	</style>
	""".format(bg=bg, text=text, surface=surface, border=border, muted=muted, badge_bg=badge_bg, badge_text=badge_text, badge_border=badge_border, grad=grad, accent=accent)
	return css


def inject_css(theme_mode: str, accent: str) -> None:
	st.markdown(build_theme_css(theme_mode, accent), unsafe_allow_html=True)


EXPECTED_FILE_1 = "Sleep_Health_and_Lifestyle_Updated.csv"
EXPECTED_FILE_2 = "Student_Stress_Factors_Updated.csv"
EXPECTED_FILE_3 = "Student Stress Factors (2).csv"
EXPECTED_FILE_4 = "Sleep_health_and_lifestyle_dataset.csv"

FEATURES = ["sleep_hours", "activity_level", "screen_time", "steps", "age", "gender", "occupation"]
SLEEP_TARGET = "sleep_quality"
STRESS_TARGET = "stress_level"


def find_dataset(path: str) -> Optional[str]:
	return path if os.path.exists(path) else None


def load_dataset(obj: Union[str, Any]) -> Optional[pd.DataFrame]:
	try:
		if hasattr(obj, "read"):
			return pd.read_csv(obj)
		return pd.read_csv(str(obj))
	except Exception:
		return None


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
	df = df.copy()
	df.columns = (
		df.columns.str.strip().str.lower()
		.str.replace(" ", "_", regex=False)
		.str.replace("-", "_", regex=False)
		.str.replace("/", "_", regex=False)
		.str.replace(r"[^a-z0-9_]", "", regex=True)  # drop punctuation/emojis
		.str.replace(r"_+", "_", regex=True)         # collapse multiple underscores
		.str.strip("_")
	)
	return df


def alias_columns(df: pd.DataFrame) -> pd.DataFrame:
	aliases: Dict[str, List[str]] = {
		"age": ["age", "person_age", "years", "user_age"],
		"gender": ["gender", "sex"],
		"occupation": ["occupation", "job", "profession", "work"],
		"sleep_hours": ["sleep_hours", "sleep_duration", "sleep_time", "hours_of_sleep"],
		"activity_level": ["activity_level", "physical_activity", "physical_activity_level", "activity", "exercise_level"],
		"screen_time": ["screen_time", "daily_screen_time", "screen_hours", "screentime"],
		"steps": ["steps", "daily_steps", "step_count", "steps_count"],
		"stress_level": ["stress_level", "stress", "perceived_stress", "stressscore", "stress_score", "stress_category"],
		"sleep_quality": ["sleep_quality", "sleep_score", "quality_of_sleep"],
	}

	canonical_to_found: Dict[str, str] = {}
	for canonical, candidates in aliases.items():
		for c in candidates:
			if c in df.columns:
				canonical_to_found[canonical] = c
				break

	renames = {v: k for k, v in canonical_to_found.items() if v != k}
	return df.rename(columns=renames)


def merge_datasets(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
	df1 = alias_columns(standardize_columns(df1))
	df2 = alias_columns(standardize_columns(df2))
	keys = [k for k in ["age", "gender", "occupation"] if k in df1.columns and k in df2.columns]
	if keys:
		try:
			return pd.merge(df1, df2, on=keys, how="inner", suffixes=("_x", "_y"))
		except Exception:
			pass
	cols = sorted(list(set(df1.columns).union(df2.columns)))
	return pd.concat([df1.reindex(cols, axis=1), df2.reindex(cols, axis=1)], ignore_index=True)


def coerce_and_impute(df: pd.DataFrame) -> pd.DataFrame:
	df = df.copy()
	for col in df.columns:
		if df[col].dtype == object:
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				coerced = pd.to_numeric(df[col], errors="ignore")
				if pd.api.types.is_numeric_dtype(coerced):
					df[col] = coerced
	# impute
	num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
	cat_cols = [c for c in df.columns if c not in num_cols]
	for c in num_cols:
		if df[c].isna().any():
			df[c] = df[c].fillna(df[c].median())
	for c in cat_cols:
		if df[c].isna().any():
			mode = df[c].mode(dropna=True)
			df[c] = df[c].fillna(mode.iloc[0] if not mode.empty else "Unknown")
	return df


def _quantize_stress_from_numeric(vals: pd.Series) -> pd.Series:
	vals = pd.to_numeric(vals, errors="coerce")
	if not vals.notna().any():
		return pd.Series(["Medium"] * len(vals), index=vals.index)
	vals = vals.fillna(vals.median())
	try:
		cats = pd.qcut(vals, q=3, labels=["Low", "Medium", "High"], duplicates="drop")
		# If fewer than 3 bins, fall back to 2; if still not possible, use median split
		if hasattr(cats, "cat") and len(cats.cat.categories) == 3:
			return cats.astype(str)
		cats2 = pd.qcut(vals, q=2, labels=["Low", "High"], duplicates="drop")
		return cats2.astype(str)
	except Exception:
		median_val = np.median(vals)
		return pd.Series(np.where(vals >= median_val, "High", "Low"), index=vals.index)


def ensure_targets(df: pd.DataFrame) -> pd.DataFrame:
	df = df.copy()
	# sleep quality
	if SLEEP_TARGET not in df.columns:
		if "sleep_hours" in df.columns:
			df[SLEEP_TARGET] = np.where(pd.to_numeric(df["sleep_hours"], errors="coerce").fillna(0) >= 7, "Good", "Poor")
		else:
			df[SLEEP_TARGET] = "Unknown"
	# stress level
	if STRESS_TARGET not in df.columns:
		if "screen_time" in df.columns:
			df[STRESS_TARGET] = _quantize_stress_from_numeric(df["screen_time"]).values
		else:
			df[STRESS_TARGET] = "Medium"
	# numeric targets -> bins
	if pd.api.types.is_numeric_dtype(df[SLEEP_TARGET]):
		vals = pd.to_numeric(df[SLEEP_TARGET], errors="coerce").fillna(df[SLEEP_TARGET].median() if df[SLEEP_TARGET].notna().any() else 0)
		med = np.median(vals)
		df[SLEEP_TARGET] = np.where(vals >= med, "Good", "Poor")
	if pd.api.types.is_numeric_dtype(df[STRESS_TARGET]):
		df[STRESS_TARGET] = _quantize_stress_from_numeric(df[STRESS_TARGET]).values
	return df


def prepare_model_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str], List[str]]:
	df = df.copy()
	defaults: Dict[str, Tuple[object, str]] = {
		"sleep_hours": (df["sleep_hours"].median() if "sleep_hours" in df.columns and pd.api.types.is_numeric_dtype(df["sleep_hours"]) else 7.0, "numeric"),
		"activity_level": (df["activity_level"].median() if "activity_level" in df.columns and pd.api.types.is_numeric_dtype(df["activity_level"]) else 5.0, "numeric"),
		"screen_time": (df["screen_time"].median() if "screen_time" in df.columns and pd.api.types.is_numeric_dtype(df["screen_time"]) else 3.0, "numeric"),
		"steps": (df["steps"].median() if "steps" in df.columns and pd.api.types.is_numeric_dtype(df["steps"]) else 5000.0, "numeric"),
		"age": (df["age"].median() if "age" in df.columns and pd.api.types.is_numeric_dtype(df["age"]) else 30.0, "numeric"),
		"gender": ("Unknown", "categorical"),
		"occupation": ("Unknown", "categorical"),
	}
	for f, (val, _) in defaults.items():
		if f not in df.columns:
			df[f] = val
	for f, (_, kind) in defaults.items():
		if kind == "numeric":
			df[f] = pd.to_numeric(df[f], errors="coerce").fillna(defaults[f][0])
	
	df = ensure_targets(df)
	X = df[FEATURES].copy()
	y_sleep = df[SLEEP_TARGET].astype(str)
	y_stress = df[STRESS_TARGET].astype(str)
	num_feats = ["sleep_hours", "activity_level", "screen_time", "steps", "age"]
	cat_feats = ["gender", "occupation"]
	return X, y_sleep, X.copy(), y_stress, num_feats, cat_feats


def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
	numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
	categorical_transformer = OneHotEncoder(handle_unknown="ignore")
	return ColumnTransformer([
		("num", numeric_transformer, numeric_features),
		("cat", categorical_transformer, categorical_features),
	])


def train_classifiers(X: pd.DataFrame, y_sleep: pd.Series, y_stress: pd.Series, numeric_features: List[str], categorical_features: List[str]):
	pre = build_preprocessor(numeric_features, categorical_features)
	sleep_clf = Pipeline([("pre", pre), ("model", RandomForestClassifier(n_estimators=200, random_state=42))])
	stress_clf = Pipeline([("pre", pre), ("model", RandomForestClassifier(n_estimators=200, random_state=42))])
	X_train, X_test, ys_train, ys_test = train_test_split(X, y_sleep, test_size=0.2, random_state=42, stratify=y_sleep if y_sleep.nunique() > 1 else None)
	_, Xs_test, yt_train, yt_test = train_test_split(X, y_stress, test_size=0.2, random_state=42, stratify=y_stress if y_stress.nunique() > 1 else None)
	sleep_clf.fit(X_train, ys_train)
	stress_clf.fit(X_train, yt_train)
	ys_pred = sleep_clf.predict(X_test)
	yt_pred = stress_clf.predict(Xs_test)
	metrics = {
		"sleep_accuracy": accuracy_score(ys_test, ys_pred) if len(set(ys_test)) > 0 else np.nan,
		"sleep_report": classification_report(ys_test, ys_pred, zero_division=0),
		"stress_accuracy": accuracy_score(yt_test, yt_pred) if len(set(yt_test)) > 0 else np.nan,
		"stress_report": classification_report(yt_test, yt_pred, zero_division=0),
	}
	return sleep_clf, {"stress": stress_clf, "metrics": metrics}


def safe_plot(fig) -> None:
	st.pyplot(fig, use_container_width=True)
	plt.close(fig)


# Sidebar theme and accent
_default_theme = st.session_state.get("ui_theme", "System")
st.sidebar.markdown("### Appearance")
ui_theme = st.sidebar.radio("Theme", ["Light", "Dark", "System"], index=["Light", "Dark", "System"].index(_default_theme))
accent_choice = st.sidebar.selectbox("Accent color", ["#6366F1", "#2563EB", "#22C55E", "#E11D48", "#F59E0B"], index=0)
st.session_state["ui_theme"] = ui_theme

inject_css("Dark" if ui_theme == "Dark" else ("Light" if ui_theme == "Light" else "Light"), accent_choice)
sns.set_theme(style="whitegrid" if ui_theme != "Dark" else "darkgrid", palette="deep")

# Hero header will render after data loads
st.markdown("<div class='app-title'><h1>💤 NeuroRest: Stress and Sleep predictor 🌙</h1></div>", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("<div class='nav-section'><h3>Navigation</h3></div>", unsafe_allow_html=True)

# Use radio but style it to look like navigation cards
mode = st.sidebar.radio("Choose a mode", ["📊 Data Exploration", "🧠 Model Training Results", "🔮 Predictions", "📘 Learn About Stress & Sleep"], index=0, label_visibility="collapsed")

st.sidebar.markdown("---")

# About section
st.sidebar.markdown("<div class='nav-section'><h3>About</h3></div>", unsafe_allow_html=True)
with st.sidebar.expander("About NeuroRest", expanded=False):
    st.markdown("""
    **NeuroRest** is an intelligent sleep and stress prediction system that uses machine learning to analyze lifestyle factors and predict sleep quality and stress levels.
    
    **Features:**
    • Data exploration and visualization
    • Machine learning model training
    • Real-time predictions
    • Interactive dashboard
    
    **Technology Stack:**
    • Streamlit for web interface
    • Scikit-learn for ML models
    • Pandas for data processing
    • Seaborn for visualizations
    """)

st.sidebar.markdown("---")

# Footer with credits
st.sidebar.markdown("<div class='nav-section'><h3>Team</h3></div>", unsafe_allow_html=True)
st.sidebar.markdown("""
<div style='text-align: center; padding: 1rem; background: rgba(99, 102, 241, 0.1); border-radius: 10px; border: 1px solid rgba(99, 102, 241, 0.2);'>
    <p style='margin: 0; font-size: 0.9rem; color: #6366F1; font-weight: 600;'>Made with ❤️ by</p>
    <p style='margin: 0.5rem 0 0 0; font-size: 0.85rem; color: #64748b;'>Kumar Ravi</p>
    <p style='margin: 0; font-size: 0.85rem; color: #64748b;'>Abinash Giri</p>
    <p style='margin: 0; font-size: 0.85rem; color: #64748b;'>Jay Gupta</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

st.sidebar.subheader("Data Sources")
st.sidebar.write("Upload up to two CSVs, or the app will auto-detect local files.")

upload1 = st.sidebar.file_uploader("Upload dataset A (CSV)", type="csv")
upload2 = st.sidebar.file_uploader("Upload dataset B (CSV)", type="csv")

st.sidebar.markdown("Suggested filenames:")
st.sidebar.code(f"{EXPECTED_FILE_1}\n{EXPECTED_FILE_2}\n{EXPECTED_FILE_3}\n{EXPECTED_FILE_4}")

f1 = find_dataset(EXPECTED_FILE_1)
f2 = find_dataset(EXPECTED_FILE_2)
f3 = find_dataset(EXPECTED_FILE_3)
f4 = find_dataset(EXPECTED_FILE_4)
existing = [p for p in [f1, f2, f3, f4] if p]
if not upload1 and not upload2 and len(existing) < 2:
	try:
		any_csvs = [os.path.join(os.getcwd(), f) for f in os.listdir(os.getcwd()) if f.lower().endswith(".csv")]
		existing = list(dict.fromkeys(existing + any_csvs))
	except Exception:
		existing = existing

df1 = load_dataset(upload1) if upload1 else (load_dataset(existing[0]) if len(existing) >= 1 else None)
df2 = load_dataset(upload2) if upload2 else (load_dataset(existing[1]) if len(existing) >= 2 else None)

if df1 is not None and df2 is not None:
	df_merged: Optional[pd.DataFrame] = merge_datasets(df1, df2)
elif df1 is not None:
	df_merged = alias_columns(standardize_columns(df1))
elif df2 is not None:
	df_merged = alias_columns(standardize_columns(df2))
else:
	df_merged = None

if df_merged is None:
	st.warning("No CSVs loaded. Upload at least one CSV or place files in the working directory.")
	st.stop()

num_rows, num_cols = df_merged.shape
missing_pct = round((df_merged.isna().sum().sum() / (num_rows * max(num_cols,1))) * 100, 2) if num_rows and num_cols else 0.0

dataset_labels: List[str] = []
if upload1: dataset_labels.append(getattr(upload1, 'name', 'dataset_A.csv'))
elif existing: dataset_labels.append(os.path.basename(existing[0]))
if upload2: dataset_labels.append(getattr(upload2, 'name', 'dataset_B.csv'))
elif len(existing) >= 2: dataset_labels.append(os.path.basename(existing[1]))

st.markdown("<div class='hero'>" 
			f"<h1>Smarter EDA and predictions for sleep and stress</h1>" 
			+ "<div class='badges'>" 
			+ "".join([f"<span class='badge'>{lbl}</span>" for lbl in dataset_labels])
			+ f"<span class='badge'>Rows: {num_rows}</span>"
			+ f"<span class='badge'>Columns: {num_cols}</span>"
			+ f"<span class='badge'>Missing: {missing_pct}%</span>"
			+ "</div>"
			+ "</div>", unsafe_allow_html=True)

# Clean and targets

df_clean = coerce_and_impute(df_merged)
df_clean = ensure_targets(df_clean)


if mode == "📊 Data Exploration":
	st.markdown("<div class='section-title'>Dataset Overview</div>", unsafe_allow_html=True)
	with st.container():
		st.markdown("<div class='card'>", unsafe_allow_html=True)
		exp = st.expander("Preview (first 10 rows)", expanded=True)
		with exp:
			st.dataframe(df_clean.head(10), use_container_width=True)
		exp2 = st.expander("Summary statistics")
		with exp2:
			st.dataframe(df_clean.describe(include="all").transpose(), use_container_width=True)
		st.markdown("</div>", unsafe_allow_html=True)

	st.markdown("<div class='section-title'>Correlations</div>", unsafe_allow_html=True)
	with st.container():
		st.markdown("<div class='card'>", unsafe_allow_html=True)
		num_df = df_clean.select_dtypes(include=[np.number])
		if num_df.shape[1] >= 2:
			corr = num_df.corr(numeric_only=True)
			fig, ax = plt.subplots(figsize=(8, 5))
			sns.heatmap(corr, annot=False, cmap=("rocket" if ui_theme=="Dark" else "Blues"), ax=ax)
			ax.set_title("Correlation heatmap")
			safe_plot(fig)
		else:
			st.info("Not enough numeric columns for a correlation heatmap.")
		st.markdown("</div>", unsafe_allow_html=True)

	st.markdown("<div class='section-title'>Feature Visualizations</div>", unsafe_allow_html=True)
	col1, col2 = st.columns(2)
	with col1:
		st.markdown("<div class='card'>", unsafe_allow_html=True)
		st.markdown("**Average sleep hours by age group & gender**")
		if {"sleep_hours", "age", "gender"}.issubset(df_clean.columns):
			df_v = df_clean.copy()
			df_v["age_group"] = pd.cut(pd.to_numeric(df_v["age"], errors="coerce").fillna(df_v["age"].median()), [0, 18, 30, 45, 60, 100], labels=["0-18", "19-30", "31-45", "46-60", "60+"])
			pivot = df_v.pivot_table(index="age_group", columns="gender", values="sleep_hours", aggfunc="mean")
			fig, ax = plt.subplots(figsize=(7, 4))
			pivot.plot(kind="bar", ax=ax, color=["#60a5fa", "#a78bfa", "#f472b6", "#34d399"])
			ax.set_xlabel("Age group")
			ax.set_ylabel("Avg sleep hours")
			ax.set_title("Avg sleep hours by age group & gender")
			plt.xticks(rotation=0)
			safe_plot(fig)
		else:
			st.info("Required columns not found: sleep_hours, age, gender")
		st.markdown("</div>", unsafe_allow_html=True)
	with col2:
		st.markdown("<div class='card'>", unsafe_allow_html=True)
		st.markdown("No stress vs screen time chart shown.")
		st.markdown("</div>", unsafe_allow_html=True)

	col3, col4 = st.columns(2)
	with col3:
		st.markdown("<div class='card'>", unsafe_allow_html=True)
		st.markdown("**Steps vs sleep quality**")
		if {"steps", SLEEP_TARGET}.issubset(df_clean.columns):
			fig, ax = plt.subplots(figsize=(7, 4))
			sns.boxplot(data=df_clean, x=SLEEP_TARGET, y="steps", ax=ax, palette=("crest" if ui_theme!="Dark" else "mako"))
			ax.set_title("Steps by sleep quality")
			safe_plot(fig)
		else:
			st.info("Required columns not found: steps and sleep_quality")
		st.markdown("</div>", unsafe_allow_html=True)
	with col4:
		st.markdown("<div class='card'>", unsafe_allow_html=True)
		st.markdown("**Occupation vs stress level distribution**")
		if {"occupation", STRESS_TARGET}.issubset(df_clean.columns):
			fig, ax = plt.subplots(figsize=(7, 4))
			ct = pd.crosstab(df_clean["occupation"], df_clean[STRESS_TARGET], normalize="index")
			ct.plot(kind="bar", stacked=True, ax=ax, colormap=("Purples" if ui_theme!="Dark" else "magma"))
			ax.set_ylabel("Proportion")
			ax.set_title("Stress distribution by occupation")
			plt.xticks(rotation=45, ha="right")
			safe_plot(fig)
		else:
			st.info("Required columns not found: occupation and stress_level")
		st.markdown("</div>", unsafe_allow_html=True)


elif mode == "🧠 Model Training Results":
	st.markdown("<div class='section-title'>Model Training</div>", unsafe_allow_html=True)
	X_sleep, y_sleep, X_stress, y_stress, num_feats, cat_feats = prepare_model_data(df_clean)
	with st.spinner("Training classifiers..."):
		sleep_model, extra = train_classifiers(X_sleep, y_sleep, y_stress, num_feats, cat_feats)
		stress_model = extra["stress"]
		metrics = extra["metrics"]
	st.markdown("<div class='card'>", unsafe_allow_html=True)
	col_a, col_b = st.columns(2)
	with col_a:
		st.metric("Sleep Quality Accuracy", f"{metrics['sleep_accuracy']:.3f}" if not np.isnan(metrics["sleep_accuracy"]) else "N/A")
		st.text("Classification report (Sleep Quality)")
		st.code(metrics["sleep_report"])
	with col_b:
		st.metric("Stress Level Accuracy", f"{metrics['stress_accuracy']:.3f}" if not np.isnan(metrics["stress_accuracy"]) else "N/A")
		st.text("Classification report (Stress Level)")
		st.code(metrics["stress_report"])
	st.markdown("</div>", unsafe_allow_html=True)
	st.info("Models are trained with preprocessing pipelines.")


elif mode == "🔮 Predictions":
	st.markdown("<div class='section-title'>Make Predictions</div>", unsafe_allow_html=True)
	X_sleep, y_sleep, X_stress, y_stress, num_feats, cat_feats = prepare_model_data(df_clean)
	sleep_model, extra = train_classifiers(X_sleep, y_sleep, y_stress, num_feats, cat_feats)
	stress_model = extra["stress"]

	gender_options = sorted(list(pd.Series(df_clean.get("gender", pd.Series(["Unknown"])) ).astype(str).unique()))
	occupation_options = sorted(list(pd.Series(df_clean.get("occupation", pd.Series(["Unknown"])) ).astype(str).unique()))

	with st.form("prediction_form"):
		col1, col2 = st.columns(2)
		with col1:
			sleep_hours = st.slider("Sleep hours", 0.0, 14.0, float(np.clip(np.nanmedian(pd.to_numeric(df_clean.get("sleep_hours", pd.Series([7])), errors="coerce")), 0, 14)), 0.5)
			activity_level = st.slider("Activity level (0-10)", 0.0, 10.0, float(np.clip(np.nanmedian(pd.to_numeric(df_clean.get("activity_level", pd.Series([5])), errors="coerce")), 0, 10)), 0.5)
			screen_time = st.slider("Screen time (hours/day)", 0.0, 16.0, float(np.clip(np.nanmedian(pd.to_numeric(df_clean.get("screen_time", pd.Series([3])), errors="coerce")), 0, 16)), 0.5)
			steps = st.slider("Steps (per day)", 0, 30000, int(np.clip(np.nanmedian(pd.to_numeric(df_clean.get("steps", pd.Series([5000])), errors="coerce")), 0, 30000)), 500)
		with col2:
			age = st.number_input("Age", 0, 120, int(np.clip(np.nanmedian(pd.to_numeric(df_clean.get("age", pd.Series([30])), errors="coerce")), 0, 120)))
			gender = st.selectbox("Gender", gender_options, index=gender_options.index("Unknown") if "Unknown" in gender_options else 0)
			occupation = st.selectbox("Occupation", occupation_options, index=occupation_options.index("Unknown") if "Unknown" in occupation_options else 0)
		go = st.form_submit_button("Predict", type="primary")

	if go:
		row = pd.DataFrame([{ "sleep_hours": sleep_hours, "activity_level": activity_level, "screen_time": screen_time, "steps": steps, "age": age, "gender": str(gender), "occupation": str(occupation) }])
		pred_sleep = sleep_model.predict(row)[0]
		pred_stress = stress_model.predict(row)[0]
		st.markdown("<div class='card'>", unsafe_allow_html=True)
		colx, coly = st.columns(2)
		with colx:
			st.markdown(f"<div class='result'><span class='label'>Sleep quality</span><span class='value'>{pred_sleep}</span></div>", unsafe_allow_html=True)
		with coly:
			st.markdown(f"<div class='result'><span class='label'>Stress level</span><span class='value'>{pred_stress}</span></div>", unsafe_allow_html=True)
		st.markdown("</div>", unsafe_allow_html=True)

		# Tailored tips
		st.markdown("<div class='card'>", unsafe_allow_html=True)
		st.markdown("**💡 Tips & Recommendations**")
		if str(pred_stress).strip().lower().startswith("high"):
			st.write("For High Stress: Try a 10-minute breathing exercise (box breathing 4-4-4-4).")
		elif str(pred_stress).strip().lower().startswith("medium"):
			st.write("For Medium Stress: Take a 5-minute walk or stretch break every hour.")
		else:
			st.write("For Low Stress: Keep a simple gratitude journal to maintain balance.")
		if str(pred_sleep).strip().lower().startswith("poor"):
			st.write("For Poor Sleep: Avoid screens 1 hour before bed and keep the room dark.")
		else:
			st.write("For Good Sleep: Maintain a consistent sleep schedule and morning light exposure.")
		st.markdown("</div>", unsafe_allow_html=True)


elif mode == "📘 Learn About Stress & Sleep":
	st.markdown("<div class='section-title'>📘 Learn About Stress & Sleep</div>", unsafe_allow_html=True)
	st.markdown("<div class='card'>", unsafe_allow_html=True)
	st.markdown("**What increases stress?**")
	st.write("• Excessive screen time • Heavy workload • Irregular sleep • Lack of exercise")
	st.markdown("**What harms sleep?**")
	st.write("• Caffeine late in the day • Screens before bed • Irregular schedule • Stress")
	st.markdown("**Do's**")
	st.write("• 7–9 hours sleep • 20–30 min daytime activity • Morning sunlight • Wind-down routine")
	st.markdown("**Don'ts**")
	st.write("• Big meals late • Blue light in last hour • Late caffeine • Work in bed")
	st.markdown("</div>", unsafe_allow_html=True)


st.markdown("<div class='small'>Built with Streamlit, pandas, scikit-learn, seaborn.</div>", unsafe_allow_html=True)
