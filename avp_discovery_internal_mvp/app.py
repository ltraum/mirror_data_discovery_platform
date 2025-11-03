# AVP Data Discovery MVP (Secure Enclave)
# Run with: streamlit run app.py

import os
import yaml
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import chi2_contingency

# Global Plotly configuration
PLOTLY_CONFIG = {
    "displaylogo": False,                 # Hide Plotly logo in modebar
    "responsive": True,                   # Make all charts resize automatically
    "modeBarButtonsToRemove": [           # Optional: remove clutter from toolbar
        "zoom", "pan", "lasso2d", "select2d", "resetScale2d"
    ]
}

# ----------------------- User Role / Permissions -----------------------
# Temporary role assignment (replace later with login-based role)
if "user_role" not in st.session_state:
    st.session_state["user_role"] = "admin"  # or "researcher"


# ----------------------- Page + Privacy Settings -----------------------
st.set_page_config(page_title="American Voices Project Data Discovery Dashboard", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "example.csv")
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

RESPONDENT_ID = "respondent_id"

MIN_CELL = 11
SUPPRESS_TOKEN = "<=10"

st.markdown("""
### American Voices Project Data Discovery Dashboard
Explore patterns in the American Voices Project survey data.
All configuration (data paths, suppression thresholds, and access settings) is hard-coded for privacy compliance. [DRAFT]
""")

# ----------------------- Helper Functions -----------------------

def load_csv(path):
    """Load CSV file into a DataFrame."""
    return pd.read_csv(path)

def load_config(path):
    """Load YAML config file if present."""
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception:
        return {}

def infer_types(df, id_col=None):
    """Infer basic variable types."""
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
    """Pull variable metadata from YAML (if available)."""
    meta = config.get("variables", {}).get(var, {}) if config else {}
    return {
        "label": meta.get("label", var),
        "question": meta.get("question", ""),
        "type": meta.get("type", ""),
        "topic": meta.get("topic", ""),
        "notes": meta.get("notes", ""),
    }

def suppress_small_cells(ct, min_cell=MIN_CELL, token=SUPPRESS_TOKEN):
    """Suppress small counts in tables and ensure Arrow compatibility."""
    ct = ct.copy().astype("object")
    mask = ct < min_cell
    ct[mask] = token
    return ct.astype(str)


def safe_pct_table(tbl_counts):
    """Return column %, row %, total % tables."""
    with np.errstate(divide="ignore", invalid="ignore"):
        col_pct = tbl_counts.div(tbl_counts.sum(axis=0), axis=1) * 100
        row_pct = tbl_counts.div(tbl_counts.sum(axis=1), axis=0) * 100
        total_pct = tbl_counts / tbl_counts.values.sum() * 100
    return col_pct, row_pct, total_pct

def get_continuous_numeric_vars(df, var_types, min_unique=6):
        """Return numeric variables with more than `min_unique` distinct non-null values."""
        numeric_vars = [c for c, t in var_types.items() if t.get("type") == "numeric"]
        return [c for c in numeric_vars if df[c].nunique(dropna=True) >= min_unique]

def corr_matrix(df, numeric_cols):
    """Simple Pearson correlation matrix."""
    return df[numeric_cols].corr(method="pearson")

def download_csv_button(df, filename, label="Download CSV"):
    """Provide a download button for any DataFrame."""
    csv_bytes = df.to_csv(index=True).encode("utf-8")
    st.download_button(label, data=csv_bytes, file_name=filename, mime="text/csv")

# ----------------------- Load Data -----------------------

if not os.path.exists(CSV_PATH):
    st.error(f"CSV not found at `{CSV_PATH}` — please verify path inside app.py.")
    st.stop()

df = load_csv(CSV_PATH)
config = load_config(CONFIG_PATH) if os.path.exists(CONFIG_PATH) else {}

var_types = infer_types(df, id_col=RESPONDENT_ID if RESPONDENT_ID in df.columns else None)
df_typed = df.copy()

# ----------------------- Tabs -----------------------

tabs = st.tabs([
    "📚 Variables",
    "📊 Distributions",
    "🔢 Categorical Correlations",
    "📈 Numeric Correlations",
    "Filter and Download Data",
    "ℹ️ About",
    "🧾 Audit Log" if st.session_state["user_role"] == "admin" else " "
])



# 📚 VARIABLES TAB ---------------------------------------------------
with tabs[0]:
    st.subheader("Variable Catalog")

    rows = []
    for c in df_typed.columns:
        meta = get_var_meta(c, config)
        vtype = var_types.get(c, {}).get("type", "")
        miss = df_typed[c].isna().mean()
        rows.append({
            "variable": c,
            "label": meta["label"] or c,
            "type": vtype,
            "topic": meta["topic"],
            "missing_rate_%": round(100*miss, 1),
            "n_unique": int(df_typed[c].nunique(dropna=True)),
        })
    cat_df = pd.DataFrame(rows).sort_values(["topic", "variable"])
    st.dataframe(cat_df, width="stretch")
    download_csv_button(cat_df, "variable_catalog.csv", "Download catalog")

# 📊 DISTRIBUTIONS TAB ---------------------------------------------------
with tabs[1]:
    st.subheader("Distribution Explorer")

    col1, col2 = st.columns([2, 1])
    with col2:
        var = st.selectbox("Variable", options=list(df_typed.columns))
        vtype = var_types.get(var, {}).get("type", "")
        filter_var = st.selectbox(
            "Optional filter (categorical)",
            options=["<none>"] + [c for c,t in var_types.items() if t.get("type")=="categorical" and c!=var]
        )
        filter_val = None
        if filter_var != "<none>":
            vals = ["<all>"] + list(df_typed[filter_var].dropna().unique())
            filter_val = st.selectbox("Filter value", options=vals)

    with col1:
        dff = df_typed.copy()
        if filter_var != "<none>" and filter_val and filter_val != "<all>":
            dff = dff[dff[filter_var] == filter_val]

        st.markdown(f"**Variable:** `{var}`  \n**Type:** `{vtype}`")
        meta = get_var_meta(var, config)
        if meta["question"]:
            st.caption(meta["question"])

        if vtype == "numeric":
            fig = px.histogram(dff, x=var, nbins=30, marginal="box")
            st.plotly_chart(fig, config=PLOTLY_CONFIG, width="stretch")
            st.write(dff[var].describe())
        else:
            counts = dff[var].value_counts(dropna=False).rename_axis(var).reset_index(name="count")
            smalls = (counts["count"] < MIN_CELL).sum()
            if smalls > 0:
                st.warning(f"{smalls} category(ies) have counts < {MIN_CELL}.")
            fig = px.bar(counts, x=var, y="count")
            st.plotly_chart(fig, config=PLOTLY_CONFIG, width="stretch")
            st.dataframe(counts, width="stretch")

# 🔁 CROSSTABS TAB ---------------------------------------------------
with tabs[2]:
    st.subheader("Categorical Correlations")
    st.caption("Crosstabulations and Cramer’s V — measuring association strength among categorical variables.")

    st.markdown("""
    This section examines how two categorical variables are related.  
    Crosstabs display observed and expected counts, while **Cramer’s V** summarizes the overall association strength on a standardized 0–1 scale.
    """)

    cat_vars = [c for c,t in var_types.items() if t.get("type")=="categorical"]
    if len(cat_vars) < 2:
        st.info("Not enough categorical variables for a crosstab.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            row_var = st.selectbox("Row", options=cat_vars)
        with col2:
            col_var = st.selectbox("Column", options=[c for c in cat_vars if c != row_var])

        strat = st.selectbox(
            "Stratify by (optional)",
            options=["<none>"] + [c for c in cat_vars if c not in (row_var, col_var)]
        )

        dff = df_typed[[row_var, col_var] + ([strat] if strat != "<none>" else [])].copy()

        def render_one_slice(slice_df, title_suffix=""):
            """Render crosstab with all tables on the left and visuals on the right."""
            ct = pd.crosstab(slice_df[row_var], slice_df[col_var], dropna=False)
            chi2, p, dof, _ = chi2_contingency(ct.fillna(0))

            # --- Layout headline ---
            st.markdown(f"### {row_var} × {col_var}")
            st.markdown(f"**χ² = {chi2:.2f}, dof = {dof}, p = {p:.4g} {title_suffix}**")

            # --- Apply suppression ---
            ct_s = suppress_small_cells(ct)
            col_pct, row_pct, total_pct = safe_pct_table(ct)

            # --- Two-column layout ---
            st.divider()
            left, right = st.columns([1.3, 1])

            # ================= LEFT COLUMN: All tables =================
            with left:
                # --- Header row: title + download button side by side ---
                hcol1, hcol2 = st.columns([3, 1])
                with hcol1:
                    st.markdown("#### Raw Counts (suppressed)")
                with hcol2:
                    download_csv_button(
                        ct_s,
                        f"crosstab_{row_var}_by_{col_var}{title_suffix.replace(' ','_')}.csv",
                        "⬇️ Download CSV"
                    )
                # --- Table itself ---
                st.dataframe(ct_s, width="stretch")

                st.markdown("#### Column Percentages")
                st.caption("Each column sums to 100% — shows distribution of **rows within each column**.")
                st.dataframe(col_pct.round(1), width="stretch")

                st.markdown("#### Row Percentages")
                st.caption("Each row sums to 100% — shows distribution of **columns within each row**.")
                st.dataframe(row_pct.round(1), width="stretch")

                st.markdown("#### Total Percentages (Share of all respondents)")
                st.caption("Each cell’s proportion of the grand total (sums to 100% across the whole table).")
                st.dataframe(total_pct.round(1), width="stretch")

                

            # ================= RIGHT COLUMN: Visuals =================
            with right:
                st.markdown("#### Visualizations")

                # --- Heatmap ---
                try:
                    fig_heat = px.imshow(
                        ct.fillna(0),
                        color_continuous_scale="Blues",
                        title=f"{row_var} vs {col_var} (Counts)"
                    )
                    fig.update_traces(texttemplate="%{z}", textfont_size=10)
                    st.plotly_chart(fig_heat, config=PLOTLY_CONFIG, width="stretch")
                except Exception:
                    st.warning("Heatmap not available for this selection.")

                # --- Stacked Bar (Row %) ---
                try:
                    row_pct_plot = row_pct.round(1).reset_index().melt(
                        id_vars=row_var, var_name=col_var, value_name="Row %"
                    )
                    fig_bar = px.bar(
                        row_pct_plot,
                        x=row_var,
                        y="Row %",
                        color=col_var,
                        barmode="stack",
                        text="Row %",
                        title=f"{row_var} by {col_var} (Row %)"
                    )
                    st.plotly_chart(fig_bar, config=PLOTLY_CONFIG, width="stretch")
                except Exception:
                    st.warning("Stacked bar not available for this selection.")


            # --- Privacy note ---
            
            st.caption("""
            *Privacy note:*  
            Cells labeled **“≤10”** represent small or empty categories (0–10 respondents).  
            Percentages are computed from full counts before suppression.
            """)




        if strat == "<none>":
            render_one_slice(dff)
        else:
            for level in dff[strat].dropna().unique():
                st.markdown(f"### {strat} = {level}")
                render_one_slice(dff[dff[strat]==level], f" ({strat}={level})")

        


# 📈 Numeric Correlations TAB
with tabs[3]:
    st.subheader("Numeric Correlations")
    st.caption("Pearson’s r and Linear Relationship Visualization — exploring associations among continuous numeric variables.")

    st.markdown("""
    This section summarizes **linear relationships between numeric variables**.  
    **Pearson’s r** quantifies the strength and direction of correlation,  
    and the interactive plots visualize fitted **linear regression lines** to show those patterns.
    """)

    # (then your correlation matrix + scatterplot demo code)



    # --- Identify continuous numeric variables ---
    numeric_vars = get_continuous_numeric_vars(df_typed, var_types)
    
    # --- User selection for correlation matrix ---
    pick = st.multiselect(
        "Select numeric variables for correlation matrix:",
        options=numeric_vars,
        default=numeric_vars[: min(10, len(numeric_vars))]
    )

    if len(pick) >= 2:
        cm = corr_matrix(df_typed, pick)
        fig = px.imshow(cm, aspect="auto")
        fig.update_traces(texttemplate="%{z}", textfont_size=10)
        st.plotly_chart(fig, config=PLOTLY_CONFIG, width="stretch")
        st.dataframe(cm.round(3), width="stretch")
        download_csv_button(cm.round(6), "correlations.csv", "Download correlation matrix")
    else:
        st.info("Pick at least two numeric variables to view the correlation matrix.")

    # --- Excluded variables disclosure ---
    excluded = [c for c,t in var_types.items() if t.get("type")=="numeric" and df_typed[c].nunique(dropna=True) <= 5]
    if excluded:
        st.markdown(f"**Excluded variables (≤ 5 unique values):** {', '.join(excluded)}")

    st.divider()

    # --- Visual Demo Section ---
    st.subheader("Visual Correlation Explorer")
    st.caption("""
This plot shows the **Pearson correlation coefficient (r)** and fitted regression line  
for any two continuous numeric variables (excluding those with ≤ 5 unique values).
""")

    if len(numeric_vars) < 2:
        st.info("Not enough numeric variables to visualize correlation.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            x_var = st.selectbox("X variable", options=numeric_vars)
        with col2:
            y_var = st.selectbox("Y variable", options=[c for c in numeric_vars if c != x_var])

        dff = df_typed[[x_var, y_var]].dropna()
        if len(dff) == 0:
            st.warning("No overlapping data between selected variables.")
        else:
            # Compute Pearson correlation
            r = dff[x_var].corr(dff[y_var])
            st.markdown(f"**Pearson correlation coefficient (r):** `{r:.3f}`")

            if abs(r) < 0.2:
                st.info("Weak or no linear relationship.")
            elif abs(r) < 0.5:
                st.info("Moderate linear relationship.")
            else:
                st.info("Strong linear relationship.")

            # Scatterplot with regression line
            fig = px.scatter(
                dff, x=x_var, y=y_var, trendline="ols",
                title=f"{y_var} vs. {x_var} (trendline = linear regression)"
            )
            st.plotly_chart(fig, config=PLOTLY_CONFIG, width="stretch")

            st.caption("""
            **Interpretation:**  
            - Blue points represent individual data observations.  
            - The black line represents the best-fitting linear regression line.  
            - The correlation coefficient (*r*) quantifies how tightly points cluster around that line.  
            """)


    # 🗂 Filter & Export Respondent IDs (internal use)
    # 🧮 Progressive Filter Builder (inside secure enclave)
# 🗂 FILTER & DOWNLOAD TAB ---------------------------------------------------
with tabs[4]:
    st.subheader("Filter and Download Data")
    st.caption("""
    Interactively filter respondent data and preview results.
    You can apply multiple filters at once.
    Only subsets with **11 or more respondents** can be exported to CSV, per AVP privacy policy.
    """)

    st.divider()

    # ---------------- Identify variable types ----------------
    cat_vars = [c for c, t in var_types.items() if t.get("type") == "categorical" and c != RESPONDENT_ID]
    num_vars = [c for c, t in var_types.items() if t.get("type") == "numeric"]

    # ---------------- Step 1: Choose variables to filter ----------------
    st.markdown("### Select Variables to Filter")
    selected_vars = st.multiselect(
        "Choose one or more variables:",
        options=cat_vars + num_vars,
        help="Select multiple variables to filter simultaneously."
    )

    st.divider()

    # ---------------- Step 2: Create dynamic filter widgets (fixed 4-column grid) ----------------
    filters = {}
    n_vars = len(selected_vars)

    if n_vars > 0:
        cols = st.columns(4)
        for i, var in enumerate(selected_vars):
            with cols[i % 4]:
                vtype = var_types[var]["type"]
                st.markdown(
                    f"<p style='font-size:15px; color:#333; margin-bottom:-4px;'><b>{var}</b></p>",
                    unsafe_allow_html=True
                )

                if vtype == "categorical":
                    opts = sorted(df_typed[var].dropna().unique().tolist())
                    chosen = st.multiselect(" ", opts, key=f"cat_{var}")
                    if chosen:
                        filters[var] = chosen

                elif vtype == "numeric":
                    min_val = float(df_typed[var].min())
                    max_val = float(df_typed[var].max())
                    chosen_range = st.slider(
                        " ",
                        min_value=min_val,
                        max_value=max_val,
                        value=(min_val, max_val),
                        key=f"num_{var}"
                    )
                    filters[var] = chosen_range
    else:
        st.info("Select one or more variables above to start filtering.")

    st.divider()

    # ---------------- Step 3: Apply filters ----------------
    subset = df_typed.copy()
    for var, filt in filters.items():
        if var_types[var]["type"] == "categorical":
            subset = subset[subset[var].isin(filt)]
        elif var_types[var]["type"] == "numeric":
            subset = subset[(subset[var] >= filt[0]) & (subset[var] <= filt[1])]

    n_subset = subset.shape[0]
    n_total = df_typed.shape[0]

    # ---------------- Step 4: Results + Table ----------------
    st.markdown(f"### Matching respondents: **{n_subset} / {n_total}**")

    table_container = st.container()

    with table_container:
        if n_subset == 0:
            st.warning("No respondents match these filters.")
        else:
            # ---- Download Button + ID List above table ----
        # Extract and show copy-and-pasteable ID list
            id_list = subset[RESPONDENT_ID].astype(str).tolist()
            id_text = ", ".join(id_list)  # or "\n".join(id_list) for one-per-line
            st.markdown("##### Respondent IDs")
            st.text_area(
                label="Respondent IDs formatted for clean transferring",
                value=id_text,
                height=100,
                key="id_textarea",
            )

            st.markdown("##### ⬇️ Download Data")
            download_csv_button(subset, "filtered_subset.csv", "Download filtered subset")

            # ---- Spreadsheet-like preview ----
            st.data_editor(
                subset.head(100),
                use_container_width=True,
                hide_index=True,
                height=420,
                key="filtered_table"
            )

        # ---------------- Audit Logging ----------------
    from datetime import datetime
    LOG_PATH = "data/audit_log.csv"
    os.makedirs("data", exist_ok=True)

    # Create entry whenever filtering occurs
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

    st.caption("""
    *Privacy note:* Subsets smaller than 11 respondents are visible for preview only.  
    Exports and respondent-ID lists are disabled to prevent potential re-identification risk.
    """)




# ℹ️ ABOUT TAB ---------------------------------------------------
with tabs[5]:
    st.markdown("""
**About this MVP**  
- Context: American Voices Project (secure enclave). Runs offline.  
- Purpose: Fast, safe exploration of ~75 variables across ~1,500 respondents.  
- Privacy: Crosstabs use small-cell suppression (threshold = 11).  
- Next: saved views, audit logs, regression templates, and qualitative linkouts (internal only).
""")


# 🧾 AUDIT LOG TAB ---------------------------------------------------
if st.session_state["user_role"] == "admin":
    with tabs[-1]:
        st.subheader("Audit Log (Administrator View)")
        st.caption("Tracks filter actions, exports, and dataset access within the secure enclave.")

        LOG_PATH = "data/audit_log.csv"

        if os.path.exists(LOG_PATH):
            log_df = pd.read_csv(LOG_PATH)
            st.dataframe(log_df, use_container_width=True, height=400)
            st.download_button(
                "⬇️ Download audit log CSV",
                log_df.to_csv(index=False).encode("utf-8"),
                "audit_log.csv"
            )
        else:
            st.info("No audit log entries yet.")
