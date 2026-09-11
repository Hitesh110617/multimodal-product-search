"""
streamlit_app.py
=================
Interactive Streamlit web application for the Multimodal E-Commerce Product Search Engine.
Supports:
  - Text-to-Image Search (e.g., 'blue denim jacket', 'running shoes black men')
  - Image-to-Image Search (upload a photo or choose an existing product)
  - Index Type Toggle (IndexFlatIP exact vs IndexHNSWFlat fast ANN)
  - Real e-commerce fashion catalog (10,000 products from Kaggle/Myntra)
  - Real-time latency tracking (encoding time, FAISS search time)

Usage:
    streamlit run app/streamlit_app.py
"""

import sys
import time
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np

from src.encoder import CLIPEncoder
from src.indexer import FAISSIndexer
from src.search import ProductSearchEngine

# Page config
st.set_page_config(
    page_title="Multimodal Fashion Search Engine",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_encoder():
    return CLIPEncoder()


@st.cache_resource
def load_catalog_data():
    data_dir = PROJECT_ROOT / "data"
    catalog_path = data_dir / "real_products.csv"
    if not catalog_path.exists():
        catalog_path = data_dir / "sample_products.csv"
    df = pd.read_csv(catalog_path)
    return df


@st.cache_resource
def load_faiss_index(index_type: str):
    models_dir = PROJECT_ROOT / "models"
    ids_path = models_dir / "real_product_ids.npy"
    if not ids_path.exists():
        ids_path = models_dir / "product_ids.npy"

    if index_type == "hnsw":
        idx_path = models_dir / "product_index_hnsw.index"
    elif index_type == "ivf":
        idx_path = models_dir / "product_index_ivf.index"
    else:
        idx_path = models_dir / "product_index_flat.index"
        if not idx_path.exists():
            idx_path = models_dir / "product_index.index"

    return FAISSIndexer.load(str(idx_path), str(ids_path), index_type=index_type)


def render_card(item: dict, col):
    with col:
        img_path = item.get("image_path", "")
        if img_path and Path(img_path).exists():
            try:
                img = Image.open(img_path)
                st.image(img, use_container_width=True)
            except Exception:
                st.warning("Image error")
        else:
            st.info("No image")

        score = item["similarity_score"]
        if score >= 0.28:
            badge = "🟢 High"
        elif score >= 0.22:
            badge = "🟡 Moderate"
        else:
            badge = "⚪ Low"

        st.markdown(f"**{item['title']}**")
        st.caption(f"🏷️ {item.get('articleType', 'N/A')} | 👥 {item.get('gender', 'N/A')}")
        st.caption(f"🎨 {item.get('baseColour', 'N/A')} | 🎯 {item.get('usage', 'N/A')}")
        st.caption(f"Score: **{score:.4f}** ({badge})")
        st.divider()


def main():
    st.title("🛍️ Multimodal E-Commerce Product Search Engine")
    st.markdown(
        "Semantic cross-modal search over **10,000 real fashion products** using "
        "**CLIP (ViT-B/32)** embeddings and **FAISS** vector indexing."
    )
    st.divider()

    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Search Configuration")
        index_choice = st.selectbox(
            "FAISS Index Architecture",
            ["IndexFlatIP (Exact Baseline)", "IndexHNSWFlat (Fast Graph ANN)"],
            index=1,
        )
        index_type = "hnsw" if "HNSW" in index_choice else "flat"

        top_k = st.slider("Results to retrieve (Top-K)", min_value=4, max_value=24, value=8, step=4)
        search_mode = st.radio("Search Mode", ["🔤 Text → Image Search", "📸 Image → Image Search"])

        st.divider()
        st.header("📊 System Stats")
        catalog_df = load_catalog_data()
        st.metric("Catalog Size", f"{len(catalog_df):,} products")
        st.metric("Embedding Model", "CLIP ViT-B/32 (512-D)")
        st.metric("Active Index", index_type.upper())

        st.divider()
        st.markdown("**Sample Text Queries:**")
        sample_queries = [
            "Men Navy Blue Striped Polo T-shirt",
            "Women Black High Heel Sandal",
            "Red Casual Canvas Sneakers",
            "Stainless Steel Analog Chronograph Watch",
            "Floral Print Summer Dress",
            "Slim Fit Blue Denim Jeans",
            "Aviator Sunglasses with Gold Frame",
            "Leather Laptop Backpack Black",
        ]
        for q in sample_queries:
            st.code(q, language=None)

    # Initialize components
    encoder = get_encoder()
    indexer = load_faiss_index(index_type)
    images_dir = PROJECT_ROOT / "data" / "real_product_images"
    if not images_dir.exists():
        images_dir = PROJECT_ROOT / "data" / "product_images"

    engine = ProductSearchEngine(encoder, indexer, catalog_df, images_dir)

    # Main search area
    if search_mode == "🔤 Text → Image Search":
        st.subheader("🔤 Natural Language Product Search")
        query = st.text_input(
            "Enter product description:",
            value="Men Navy Blue Striped Polo T-shirt",
            placeholder="e.g. White running shoes for men, black leather jacket, floral maxi dress...",
        )

        if st.button("🔍 Search Catalog", type="primary") or query:
            with st.spinner("Embedding query & searching FAISS index..."):
                res = engine.search_by_text(query, top_k=top_k)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Latency", f"{res['total_latency_ms']:.2f} ms")
            with col2:
                st.metric("CLIP Encoding", f"{res['encoding_latency_ms']:.2f} ms")
            with col3:
                st.metric("FAISS Search", f"{res['search_latency_ms']:.2f} ms")

            st.markdown(f"**Retrieved {len(res['results'])} matching products:**")
            cols_per_row = 4
            for i in range(0, len(res["results"]), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(res["results"]):
                        render_card(res["results"][i + j], col)

    else:
        st.subheader("📸 Visual Similarity Search")
        uploaded_file = st.file_uploader(
            "Upload an image of a fashion item (or screenshot):",
            type=["jpg", "jpeg", "png", "webp"],
        )

        if uploaded_file is not None:
            query_img = Image.open(uploaded_file).convert("RGB")
            c1, c2 = st.columns([1, 3])
            with c1:
                st.image(query_img, caption="Query Image", width=180)
            with c2:
                st.write("Click below to search the catalog for visually similar items.")
                search_clicked = st.button("🔍 Find Visually Similar Products", type="primary")

            if search_clicked or uploaded_file:
                with st.spinner("Extracting visual features & querying index..."):
                    res = engine.search_by_image(query_img, top_k=top_k)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Latency", f"{res['total_latency_ms']:.2f} ms")
                with col2:
                    st.metric("CLIP Encoding", f"{res['encoding_latency_ms']:.2f} ms")
                with col3:
                    st.metric("FAISS Search", f"{res['search_latency_ms']:.2f} ms")

                st.markdown(f"**Top {len(res['results'])} visually similar items:**")
                cols_per_row = 4
                for i in range(0, len(res["results"]), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, col in enumerate(cols):
                        if i + j < len(res["results"]):
                            render_card(res["results"][i + j], col)

    # Footer
    st.divider()
    st.caption(
        "End-to-End Multimodal Retrieval System • Dataset: Kaggle/Myntra (10K products) • "
        "Model: OpenAI CLIP ViT-B/32 • Vector Index: FAISS IndexFlatIP & IndexHNSWFlat"
    )


if __name__ == "__main__":
    main()
