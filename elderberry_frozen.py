import glob
import joblib
import requests

from datetime import datetime
from datetime import timedelta

import colorcet as cc
import pandas as pd
import plotly.express as px
import streamlit as st

from sentence_transformers import SentenceTransformer



SUBFIELD_IDS = {
    "Aging": "2717",
    "Biochemistry": "1303",
    "Biophysics": "1304",
    "Biotechnology": "1305",
    "Cancer Research": "1306",
    "Cell Biology": "1307",
    "Clinical Biochemistry": "1308",
    "Developmental Biology": "1309",
    "Endocrinology": "1310",
    "Genetics": "1311",
    "Molecular Biology": "1312",
    "Molecular Medicine": "1313",
    "Physiology": "1314",
    "Structural Biology": "2401",
}

#def get_recent_openalex_papers(
#    subfield_ids,
#    days_back=7,
#    max_results=500
#):

def get_recent_openalex_papers(
    subfield_ids,
    days_back=7,
    max_results=500,
    api_key="",
    mailto=""
):

    from_date = (
        datetime.today()
        - timedelta(days=days_back)
    ).strftime("%Y-%m-%d")

    filters = [
        f"from_publication_date:{from_date}",
        "type:article|review|preprint"
    ]

    if subfield_ids:

        filters.append(
            "primary_topic.subfield.id:"
            + "|".join(subfield_ids)
        )

    #params = {
    #    "filter": ",".join(filters),
    #    "per-page": min(max_results, 200),
    #    "select": ",".join([
    #        "id",
    #        "doi",
    #        "display_name",
    #        "abstract_inverted_index",
    #        "publication_date",
    #        "cited_by_count",
    #        "primary_topic"
    #    ])
    #}

    params = {
        "filter": ",".join(filters),
        "per-page": min(max_results, 200),
        "select": ",".join([
            "id",
            "doi",
            "display_name",
            "abstract_inverted_index",
            "publication_date",
            "cited_by_count",
            "primary_topic"
        ])
    }

    if api_key:
        params["api_key"] = api_key

    if mailto:
        params["mailto"] = mailto


    #r = requests.get(
    #    "https://api.openalex.org/works",
    #    params=params,
    #    timeout=60
    #)

    headers = {
        "User-Agent":
            "OpenAlexPaperRadar/0.1"
    }

    r = requests.get(
        "https://api.openalex.org/works",
        params=params,
        headers=headers,
        timeout=60
    )

    if not r.ok:
        st.write("Status:", r.status_code)
        st.write(r.text)

    r.raise_for_status()

    results = r.json().get(
        "results",
        []
    )

    rows = []

    for work in results:

        abstract = ""

        inv = work.get(
            "abstract_inverted_index"
        )

        if inv:

            positions = {}

            for word, idxs in inv.items():
                for i in idxs:
                    positions[i] = word

            abstract = " ".join(
                positions[i]
                for i in sorted(
                    positions.keys()
                )
            )

        pt = (
            work.get("primary_topic")
            or {}
        )

        rows.append({
            "Title":
                work.get(
                    "display_name"
                ),

            "Abstract":
                abstract,

            "OpenAlexID":
                work.get("id"),

            "PublicationDate":
                work.get(
                    "publication_date"
                ),

            "CitationCount":
                work.get(
                    "cited_by_count",
                    0
                ),

            "Topic":
                pt.get(
                    "display_name",
                    ""
                )
        })

    return pd.DataFrame(rows)

######################


@st.cache_data(ttl=3600)
def build_radar_dataset(
    subfield_ids,
    days_back,
    api_key,
    mailto,
    model_name
):

    new_papers_df = get_recent_openalex_papers(
        subfield_ids=subfield_ids,
        days_back=days_back,
        max_results=750,
        api_key=api_key,
        mailto=mailto
    )

    if len(new_papers_df) == 0:
        return new_papers_df

    texts = (
        new_papers_df["Title"].fillna("")
        + ". "
        + new_papers_df["Abstract"].fillna("")
    ).tolist()

    embeddings = specter_model.encode(
        texts,
        batch_size=16,
        show_progress_bar=False,
        convert_to_numpy=True
    )

    xy = umap_model.transform(
        embeddings
    )

    new_papers_df["x"] = xy[:, 0]
    new_papers_df["y"] = xy[:, 1]

    new_papers_df["embedding"] = (
        embeddings.tolist()
    )

    return new_papers_df

# =====================================================
# PAGE
# =====================================================

st.set_page_config(
    page_title="OpenAlex Paper Radar",
    layout="wide"
)

st.title("📡OpenAlex Paper Radar")


@st.cache_resource
def load_specter():

    return SentenceTransformer(
        "allenai-specter"
    )

specter_model = load_specter()

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_parquet(source):
    return pd.read_parquet(source)


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("Radar Settings")

# -----------------------------------------------------
# Anchor selection
# -----------------------------------------------------

parquet_files = sorted(
    glob.glob("*.parquet")
)

selected_anchor = None

if parquet_files:
    selected_anchor = st.sidebar.selectbox(
        "Anchor Dataset",
        parquet_files
    )

uploaded_file = st.sidebar.file_uploader(
    "Upload Anchor Dataset",
    type=["parquet"]
)

# -----------------------------------------------------
# UMAP model selection
# -----------------------------------------------------

pkl_files = sorted(
    glob.glob("*.pkl")
)

selected_umap_model = None

if pkl_files:
    selected_umap_model = st.sidebar.selectbox(
        "UMAP Model",
        pkl_files
    )

uploaded_umap = st.sidebar.file_uploader(
    "Upload UMAP Model",
    type=["pkl"]
)

# -----------------------------------------------------
# Load dataset
# -----------------------------------------------------

if uploaded_file is not None:
    df = pd.read_parquet(uploaded_file)

elif selected_anchor is not None:
    df = load_parquet(selected_anchor)

else:
    st.warning("No parquet file found.")
    st.stop()


# -----------------------------------------------------
# Load UMAP model
# -----------------------------------------------------

if uploaded_umap is not None:

    umap_model = joblib.load(
        uploaded_umap
    )

elif selected_umap_model is not None:

    umap_model = joblib.load(
        selected_umap_model
    )

else:

    st.warning(
        "No UMAP model selected."
    )

    st.stop()


# -----------------------------------------------------
# Field filter
# -----------------------------------------------------

field_options = sorted(
    df["Field"].dropna().unique()
)

default_field = 0

if "Biochemistry, Genetics and Molecular Biology" in field_options:
    default_field = field_options.index(
        "Biochemistry, Genetics and Molecular Biology"
    )

selected_field = st.sidebar.selectbox(
    "OpenAlex Field",
    field_options,
    index=default_field
)

df_filtered = df[
    df["Field"] == selected_field
]

# -----------------------------------------------------
# Subfield filter
# -----------------------------------------------------

subfield_options = sorted(
    df_filtered["Subfield"].dropna().unique()
)

selected_subfields = st.sidebar.multiselect(
    "Subfield",
    options=subfield_options,
    default=subfield_options
)

if selected_subfields:
    df_filtered = df_filtered[
        df_filtered["Subfield"].isin(
            selected_subfields
        )
    ]

    # -----------------------------------------------------
    # Lookup OpenAlex subfield IDs
    # -----------------------------------------------------
    selected_subfield_ids = [
        SUBFIELD_IDS[s]
        for s in selected_subfields
        if s in SUBFIELD_IDS
    ]

    ### temporarily changed to this for debugging:

    #selected_subfield_ids = ['1312']

st.sidebar.caption(
    f"OpenAlex IDs: {selected_subfield_ids}"
)

# -----------------------------------------------------
# Colouring
# -----------------------------------------------------

color_options = [
    "Topic",
    "Subfield",
    "Open access"
]

selected_colour = st.sidebar.selectbox(
    "Colour By",
    color_options,
    index=0
)

# -----------------------------------------------------
# Radar controls
# -----------------------------------------------------
st.sidebar.subheader("OpenAlex API")

openalex_api_key = st.sidebar.text_input(
    "API key (optional)",
    value="",
    type="password"
)

mailto = st.sidebar.text_input(
    "Contact email (optional)",
    value=""
)

days_back = st.sidebar.slider(
    "Days Back",
    min_value=1,
    max_value=30,
    value=1
)

show_radar_overlay = st.sidebar.checkbox(
    "Show Radar Overlay",
    value=True
)

refresh_button = st.sidebar.button(
    "🔄 Refresh Radar"
)




### STATUS BOX

st.info(
    f"Monitoring: {selected_field} | "
    f"{len(selected_subfields)} subfields selected | "
    f"Last {days_back} days"
)

# -----------------------------------------------------
# Radar feed
# -----------------------------------------------------



if show_radar_overlay:

    with st.spinner(
        "Querying OpenAlex..."
    ):

        #new_papers_df = get_recent_openalex_papers(
        #    subfield_ids=selected_subfield_ids,
        #    days_back=days_back,
        #    max_results=200
        #)

       


        new_papers_df = build_radar_dataset(
            subfield_ids=selected_subfield_ids,
            days_back=days_back,
            api_key=openalex_api_key,
            mailto=mailto,
            model_name=selected_umap_model
        )

       

else:

    new_papers_df = pd.DataFrame()


st.info(
    f"Monitoring: {selected_field} | "
    f"{len(selected_subfields)} subfields | "
    f"Last {days_back} days | "
    f"{len(new_papers_df)} new papers"
)

# =====================================================
# COLOUR MAP
# =====================================================

categories = sorted(
    df_filtered[selected_colour]
    .dropna()
    .astype(str)
    .unique()
)

color_map = {
    category: cc.glasbey[i % len(cc.glasbey)]
    for i, category in enumerate(categories)
}

# =====================================================
# INFO
# =====================================================

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Anchor Papers",
        f"{len(df_filtered):,}"
    )

with col2:
    st.metric(
        "Radar Contacts",
        f"{len(new_papers_df):,}"
    )

# =====================================================
# MAIN PLOT
# =====================================================

fig = px.scatter(
    df_filtered,
    x="x",
    y="y",
    color=selected_colour,
    color_discrete_map=color_map,
    hover_name="Title",
    hover_data={
        "Topic": True,
        "Subfield": True,
        "Citation count": True,
        "Field": False,
        "x": False,
        "y": False
    },
    render_mode="webgl"
)

# -----------------------------------------------------
# Anchor styling
# -----------------------------------------------------

fig.update_traces(
    marker=dict(
        size=4,
        opacity=0.20
    )
)

# -----------------------------------------------------
# Radar layer
# -----------------------------------------------------

if len(new_papers_df):

    fig.add_scatter(
        x=new_papers_df["x"],
        y=new_papers_df["y"],
        mode="markers",
        name="New Papers",
        text=new_papers_df["Title"],
        hovertemplate="<b>%{text}</b><extra></extra>",
        marker=dict(
            size=12,
            color="yellow",
            symbol="star",
            line=dict(
                color="black",
                width=1
            )
        )
    )

# -----------------------------------------------------
# Layout
# -----------------------------------------------------

fig.update_layout(
    height=950,
    showlegend=True,
    legend_title=selected_colour
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# NEW PAPERS TABLE
# =====================================================

if len(new_papers_df):

    st.subheader(
        f"🆕 New papers ({len(new_papers_df):,})"
    )

    display_cols = [
        c for c in [
            "PublicationDate",
            "Title",
            "Topic",
            "CitationCount",
            "OpenAlexID"
        ]
        if c in new_papers_df.columns
    ]

    st.dataframe(
        new_papers_df[
            display_cols
        ].sort_values(
            "PublicationDate",
            ascending=False
        ),
        use_container_width=True,
        height=600
    )

    st.download_button(
        label="⬇ Download radar papers (CSV)",
        data=new_papers_df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name=(
            f"radar_"
            f"{days_back}d_"
            f"{len(new_papers_df)}papers.csv"
        ),
        mime="text/csv"
    )