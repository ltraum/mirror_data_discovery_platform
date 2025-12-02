# AVP Data Discovery MVP (Secure Enclave)
# Run with: streamlit run new_app.py

import os
import yaml
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import chi2_contingency
from datetime import datetime
import altair as alt


# ----------------------- CONFIG -----------------------

st.set_page_config(
    page_title="American Voices Project Data Discovery Dashboard",
    layout="wide"
)

PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": [
        "zoom", "pan", "lasso2d", "select2d", "resetScale2d"
    ]
}

if "user_role" not in st.session_state:
    st.session_state["user_role"] = "admin"  # or "researcher"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "example.csv")
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

RESPONDENT_ID = "respondent_id"
MIN_CELL = 11
SUPPRESS_TOKEN = "<=10"

# ----------------------- HELPERS -----------------------

def load_csv(path):
    return pd.read_csv(path)

def load_config(path):
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception:
        return {}

def infer_types(df, id_col=None):
    inferred = {}
    for col in df.columns:
        if id_col and col == id_col:
            inferred[col] = {"type": "id"}
        elif pd.api.types.is_numeric_dtype(df[col]):
            inferred[col] = {"type": "numeric"}
        else:
            inferred[col] = {"type": "categorical"}
    return inferred

def get_var_meta(var, config):
    meta = config.get("variables", {}).get(var, {}) if config else {}
    return {
        "label": meta.get("label", var),
        "question": meta.get("question", ""),
        "type": meta.get("type", ""),
        "topic": meta.get("topic", ""),
        "notes": meta.get("notes", ""),
    }

def suppress_small_cells(ct, min_cell=MIN_CELL, token=SUPPRESS_TOKEN):
    ct = ct.copy().astype("object")
    mask = ct < min_cell
    ct[mask] = token
    return ct.astype(str)

def safe_pct_table(tbl_counts):
    with np.errstate(divide="ignore", invalid="ignore"):
        col_pct = tbl_counts.div(tbl_counts.sum(axis=0), axis=1) * 100
        row_pct = tbl_counts.div(tbl_counts.sum(axis=1), axis=0) * 100
        total_pct = tbl_counts / tbl_counts.values.sum() * 100
    return col_pct, row_pct, total_pct

def get_continuous_numeric_vars(df, var_types, min_unique=6):
    numeric_vars = [c for c, t in var_types.items() if t.get("type") == "numeric"]
    return [c for c in numeric_vars if df[c].nunique(dropna=True) >= min_unique]

def corr_matrix(df, numeric_cols):
    return df[numeric_cols].corr(method="pearson")

def download_csv_button(df, filename, label="Download CSV"):
    csv_bytes = df.to_csv(index=True).encode("utf-8")
    st.download_button(label, data=csv_bytes, file_name=filename, mime="text/csv")

# ----------------------- LOAD DATA -----------------------

if not os.path.exists(CSV_PATH):
    st.error(f"CSV not found at `{CSV_PATH}`.")
    st.stop()

df = load_csv(CSV_PATH)
config = load_config(CONFIG_PATH) if os.path.exists(CONFIG_PATH) else {}
var_types = infer_types(
    df,
    id_col=RESPONDENT_ID if RESPONDENT_ID in df.columns else None
)
df_typed = df.copy()

# -------------------------------------------------------------------
# ----------------------- PAGE DEFINITIONS ---------------------------
# -------------------------------------------------------------------

# 📚 VARIABLES --------------------------------------------------------


def page_variables():
    st.title("📚 Variable Catalog")

    st.markdown("""
    Browse all variables from the AVP codebook.  
    Use the filters below to browse the dataset dictionary or highlight the variables included in the simplified dataset (n ≈ 76).
    """)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CODEBOOK_PATH = os.path.join(BASE_DIR, "data", "codebook_12-16-23.csv")

    @st.cache_data
    def load_codebook():
        df = pd.read_csv(CODEBOOK_PATH, dtype=str)
        df = df.fillna("")

        # Normalize column names
        df.columns = (
            df.columns.str.strip()
                      .str.lower()
                      .str.replace(" ", "_")
                      .str.replace("/", "_")
        )

        # Standard column renames
        df = df.rename(columns={
            "description___universe": "description"
        })

        # -----------------------------
        # Detect umbrella headers correctly
        # -----------------------------
        umbrella = None
        umbrella_list = []
        rows_to_drop = []

        for idx, row in df.iterrows():

            var = row["variable"].strip()
            cat = row["category"].strip()
            desc = row["description"].strip()

            # Detect umbrella: variable is TITLE CASE & category empty
            if var != "" and var.istitle() and cat == "":
                umbrella = var   # store umbrella name
                rows_to_drop.append(idx)
                continue

            # Normal variable row → store current umbrella
            umbrella_list.append(umbrella)

        # Add umbrella column
        df = df.drop(rows_to_drop)
        df["umbrella_category"] = umbrella_list

        # Clean simplified flag
        df["simplified_flag"] = (
            df["simplified"].fillna("").astype(str).str.lower().eq("yes")
        )

        return df

    # Load dataset
    df = load_codebook()

    # ---------------------------
    # FILTERS
    # ---------------------------
    st.subheader("Filters")

    show_mode = st.radio(
        "Show:",
        ["Simplified variables only", "All variables"],
        index=0
    )

    umbrella_filter = st.selectbox(
        "Filter by umbrella section:",
        ["All"] + sorted(df["umbrella_category"].dropna().unique())
    )

    category_filter = st.multiselect(
        "Filter by category:",
        sorted(df["category"].dropna().unique())
    )

    search = st.text_input(
        "Search variable name or description:",
        placeholder="e.g., earnings, education, remote"
    )

    # ---------------------------
    # Apply filtering
    # ---------------------------
    filtered = df.copy()

    if show_mode == "Simplified variables only":
        filtered = filtered[filtered["simplified_flag"]]

    if umbrella_filter != "All":
        filtered = filtered[filtered["umbrella_category"] == umbrella_filter]

    if category_filter:
        filtered = filtered[filtered["category"].isin(category_filter)]

    if search:
        s = search.lower()
        filtered = filtered[
            filtered["variable"].str.lower().str.contains(s)
            | filtered["description"].str.lower().str.contains(s)
        ]

    # ---------------------------
    # Display
    # ---------------------------
    st.subheader("Variable List")

    st.dataframe(
        filtered[[
            "variable",
            "category",
            "umbrella_category",
            "description",
            "values",
            "simplified_flag"
        ]],
        use_container_width=True,
        hide_index=True
    )

    st.caption("Variables marked as **simplified** are included in the 76-variable public-facing dataset.")


# 📊 DISTRIBUTIONS ----------------------------------------------------

def page_distributions():
    st.title("📊 Distribution Explorer")

    col1, col2 = st.columns([2, 1])
    with col2:
        var = st.selectbox("Variable", options=list(df_typed.columns))
        vtype = var_types.get(var, {}).get("type", "")
        filter_var = st.selectbox(
            "Optional filter (categorical)",
            options=["<none>"] + [
                c for c, t in var_types.items()
                if t.get("type") == "categorical" and c != var
            ]
        )
        filter_val = None
        if filter_var != "<none>":
            vals = ["<all>"] + list(df_typed[filter_var].dropna().unique())
            filter_val = st.selectbox("Filter value", options=vals)

    with col1:
        dff = df_typed.copy()
        if filter_var != "<none>" and filter_val and filter_val != "<all>":
            dff = dff[dff[filter_var] == filter_val]

        st.markdown(f"**Variable:** `{var}` — **Type:** `{vtype}`")
        meta = get_var_meta(var, config)
        if meta["question"]:
            st.caption(meta["question"])

        if vtype == "numeric":
            fig = px.histogram(dff, x=var, nbins=30, marginal="box")
            st.plotly_chart(fig, config=PLOTLY_CONFIG, use_container_width=True)
            st.write(dff[var].describe())
        else:
            counts = dff[var].value_counts(dropna=False).rename_axis(var).reset_index(name="count")
            if (counts["count"] < MIN_CELL).sum() > 0:
                st.warning("Some categories have < 11 respondents.")
            fig = px.bar(counts, x=var, y="count")
            st.plotly_chart(fig, config=PLOTLY_CONFIG, use_container_width=True)
            st.dataframe(counts, use_container_width=True)


# 🔢 CROSSTABS --------------------------------------------------------

def page_crosstabs():
    st.title("🔢 Categorical Correlations")
    st.caption("Crosstabs, Chi-Square, and Cramér’s V.")

    cat_vars = [c for c, t in var_types.items() if t.get("type") == "categorical"]
    if len(cat_vars) < 2:
        st.info("Not enough categorical variables.")
        return

    col1, col2 = st.columns(2)
    with col1:
        row_var = st.selectbox("Row", options=cat_vars)
    with col2:
        col_var = st.selectbox("Column", options=[c for c in cat_vars if c != row_var])

    strat = st.selectbox(
        "Stratify by (optional)",
        options=["<none>"] + [c for c in cat_vars if c not in (row_var, col_var)]
    )

    dff = df_typed[[row_var, col_var] + ([strat] if strat != "<none>" else [])]

    def render_slice(df_slice, title_suffix=""):
        ct = pd.crosstab(df_slice[row_var], df_slice[col_var], dropna=False)
        chi2, p, dof, _ = chi2_contingency(ct.fillna(0))

        st.markdown(f"### {row_var} × {col_var} {title_suffix}")
        st.markdown(f"**χ² = {chi2:.2f}, dof = {dof}, p = {p:.4g}**")

        ct_s = suppress_small_cells(ct)
        col_pct, row_pct, total_pct = safe_pct_table(ct)

        left, right = st.columns([1.3, 1])

        with left:
            st.markdown("#### Raw Counts (suppressed)")
            download_csv_button(ct_s, f"crosstab_{row_var}_{col_var}.csv")
            st.dataframe(ct_s)

            st.markdown("#### Column %")
            st.dataframe(col_pct.round(1))

            st.markdown("#### Row %")
            st.dataframe(row_pct.round(1))

            st.markdown("#### Total %")
            st.dataframe(total_pct.round(1))

        with right:
            try:
                fig = px.imshow(ct.fillna(0), color_continuous_scale="Blues")
                st.plotly_chart(fig, config=PLOTLY_CONFIG)
            except Exception:
                pass

            try:
                row_pct_plot = row_pct.round(1).reset_index().melt(
                    id_vars=row_var, var_name=col_var, value_name="Row %"
                )
                fig2 = px.bar(
                    row_pct_plot,
                    x=row_var,
                    y="Row %",
                    color=col_var,
                    barmode="stack",
                )
                st.plotly_chart(fig2, config=PLOTLY_CONFIG)
            except Exception:
                pass

    if strat == "<none>":
        render_slice(dff)
    else:
        for level in dff[strat].dropna().unique():
            st.markdown(f"## {strat} = {level}")
            render_slice(dff[dff[strat] == level], f"({level})")


# 📈 NUMERIC CORRELATIONS ----------------------------------------------

def page_numeric_corr():
    st.title("📈 Numeric Correlations")

    numeric_vars = get_continuous_numeric_vars(df_typed, var_types)

    pick = st.multiselect(
        "Select numeric variables:",
        options=numeric_vars,
        default=numeric_vars[: min(10, len(numeric_vars))]
    )

    if len(pick) >= 2:
        cm = corr_matrix(df_typed, pick)
        fig = px.imshow(cm)
        fig.update_traces(texttemplate="%{z:.2f}", textfont_size=10)
        st.plotly_chart(fig, config=PLOTLY_CONFIG)
        st.dataframe(cm.round(3))
        download_csv_button(cm.round(6), "correlations.csv", "Download correlation matrix")
    else:
        st.info("Pick ≥ 2 numeric variables.")

    st.subheader("Visual Correlation Explorer")

    if len(numeric_vars) < 2:
        st.info("Not enough numeric variables.")
        return

    col1, col2 = st.columns(2)
    with col1:
        x_var = st.selectbox("X", options=numeric_vars)
    with col2:
        y_var = st.selectbox("Y", options=[c for c in numeric_vars if c != x_var])

    dff = df_typed[[x_var, y_var]].dropna()

    if len(dff) == 0:
        st.warning("No overlapping data.")
        return

    r = dff[x_var].corr(dff[y_var])
    st.markdown(f"**Pearson r = {r:.3f}**")

    fig = px.scatter(dff, x=x_var, y=y_var, trendline="ols")
    st.plotly_chart(fig, config=PLOTLY_CONFIG)


# 🗂 FILTER & DOWNLOAD ------------------------------------------------

def page_filter_download():
    st.title("🗂 Filter and Download Data")

    st.caption("""
        Interactively filter respondent data and preview results.
        Only subsets with ≥ 11 respondents may be exported.
    """)

    cat_vars = [
        c for c, t in var_types.items()
        if t.get("type") == "categorical" and c != RESPONDENT_ID
    ]
    num_vars = [c for c, t in var_types.items() if t.get("type") == "numeric"]

    st.markdown("### Select Variables to Filter")
    selected = st.multiselect("Variables", options=cat_vars + num_vars)
    st.divider()

    filters = {}
    if selected:
        cols = st.columns(4)
        for i, var in enumerate(selected):
            with cols[i % 4]:
                vtype = var_types[var]["type"]
                st.markdown(f"**{var}**")
                if vtype == "categorical":
                    opts = sorted(df_typed[var].dropna().unique())
                    chosen = st.multiselect(" ", opts, key=f"cat_{var}")
                    if chosen:
                        filters[var] = chosen
                else:
                    min_v = float(df_typed[var].min())
                    max_v = float(df_typed[var].max())
                    rng = st.slider(" ", min_value=min_v, max_value=max_v, value=(min_v, max_v), key=f"num_{var}")
                    filters[var] = rng
    else:
        st.info("Select variables above.")

    st.divider()

    subset = df_typed.copy()
    for var, filt in filters.items():
        if var_types[var]["type"] == "categorical":
            subset = subset[subset[var].isin(filt)]
        else:
            subset = subset[(subset[var] >= filt[0]) & (subset[var] <= filt[1])]

    n_subset = subset.shape[0]
    st.markdown(f"### Matching Respondents: **{n_subset}**")

    if n_subset == 0:
        st.warning("No matches.")
    else:
        ids = subset[RESPONDENT_ID].astype(str).tolist()
        st.markdown("#### Respondent IDs")
        st.text_area("Copy/paste IDs:", "\n".join(ids), height=120)

        if n_subset >= MIN_CELL:
            download_csv_button(subset, "filtered_subset.csv", "Download filtered subset")
        else:
            st.warning("Subset < 11 respondents — export disabled.")

        st.data_editor(subset.head(100), use_container_width=True, height=400)

    # Audit Log
    LOG_PATH = os.path.join(BASE_DIR, "data", "audit_log.csv")
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    log_entry = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "user": st.session_state.get("user_role", "unknown"),
        "n_matched": n_subset,
        "exported": n_subset >= MIN_CELL,
        "filters": str(filters)
    }])

    if os.path.exists(LOG_PATH):
        log_entry.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        log_entry.to_csv(LOG_PATH, index=False)

    st.caption("*Note:* Subsets < 11 are preview-only per privacy rules.")


# ℹ️ ABOUT ------------------------------------------------------------


def page_about():
    st.title("AVP Data Discovery Dashboard")

    # ===========================
    #  Layout: Text (left) | Charts (right)
    #  More even split + better spacing
    # ===========================
    left, right = st.columns([1.8, 1.2])

    # ----------------------------------
    # LEFT COLUMN (TEXT)
    # ----------------------------------
    with left:
        st.markdown("""
        The **AVP Data Discovery Dashboard** is an orientation tool for becoming familiar with the survey and demographic elements of the AVP dataset, spot variables of interest, and shape research ideas **before applying for secure-server access**.
        """)

        st.markdown("---")

        st.markdown("### What You Can Do")
        st.markdown("""
        - Search the **codebook** and variable definitions  
        - Explore associations between variables 
        - Identify variables relevant to potential future research
        """)

        st.markdown("---")

        st.markdown("### Privacy & Disclosure")
        st.markdown("""
        This dashboard displays **only aggregated information** to protect respondent confidentiality. No individual-level data are available. Researchers needing deeper access may apply for **secure-server use** under established safeguards.
        """)

    # ----------------------------------
    # RIGHT COLUMN (CHARTS)
    # ----------------------------------
    with right:
        st.markdown(
            "<div style='font-size:20px; font-weight:600; margin-bottom:10px;'>AVP Sample at a Glance</div>",
            unsafe_allow_html=True
        )

        # Palette
        palette = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#64B5CD", "#CCB974"]

        # Data
        region_data = pd.DataFrame({
            "Region": ["Northeast", "South", "Midwest", "West"],
            "Percent": [15.6, 42.0, 18.2, 24.2]
        })

        income_data = pd.DataFrame({
            "Income Bracket": ["≤ $24,000", "$24,001–$48,000", "$48,001–$72,000",
                               "$72,001–$120,000", "> $120,000", "Missing"],
            "Percent": [35.0, 21.5, 12.8, 11.6, 8.1, 11.0]
        })

        # ----------------------------------
        # Region Donut Chart (smaller title + tighter size)
        # ----------------------------------
        donut = (
            alt.Chart(region_data)
            .mark_arc(innerRadius=70)
            .encode(
                theta=alt.Theta("Percent:Q"),
                color=alt.Color("Region:N", scale=alt.Scale(range=palette[:4])),
                tooltip=["Region", "Percent"]
            )
            .properties(
                title="Region",
                width=250,
                height=250
            )
        )
        st.altair_chart(donut, use_container_width=True)

        # ----------------------------------
        # Income Bar Chart (s






# 🧾 AUDIT LOG (ADMIN) ------------------------------------------------

def page_audit_log():
    st.title("🧾 Audit Log (to be made private for admin team use)")

    LOG_PATH = os.path.join(BASE_DIR, "data", "audit_log.csv")

    if os.path.exists(LOG_PATH):
        log_df = pd.read_csv(LOG_PATH)
        st.dataframe(log_df, use_container_width=True, height=400)
        st.download_button(
            "Download audit log CSV",
            log_df.to_csv(index=False).encode("utf-8"),
            "audit_log.csv"
        )
    else:
        st.info("No audit entries yet.")


# -------------------------------------------------------------------
# ----------------------- NAVIGATION MENU ---------------------------
# -------------------------------------------------------------------

pages = {
    "System": [
        st.Page(page_about, title="About", icon="ℹ️")
    ],
    "Data Exploration": [
        st.Page(page_variables, title="Variables", icon="📚"),
        st.Page(page_distributions, title="Distributions", icon="📊"),
        st.Page(page_crosstabs, title="Categorical Correlations", icon="🔢"),
        st.Page(page_numeric_corr, title="Numeric Correlations", icon="📈"),
        st.Page(page_filter_download, title="Filter & Download", icon="🗂"),
    ]
}

if st.session_state["user_role"] == "admin":
    pages["System"].append(
        st.Page(page_audit_log, title="Audit Log", icon="🧾")
    )

pg = st.navigation(pages, position="sidebar", expanded=True)
pg.run()


pg = st.navigation(pages, position="sidebar", expanded=True)
pg.run()
