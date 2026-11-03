import hashlib
import json
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

# ============================================================
# 1. PATHS AND APPLICATION SETTINGS
# ============================================================

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "indian_food_classifier_final.keras"
CLASS_NAMES_PATH = APP_DIR / "config" / "class_names.json"
CONFIG_PATH = APP_DIR / "config" / "final_model_config.json"
NUTRITION_PATH = APP_DIR / "data" / "nutrition" / "Indian_Food_Nutrition_EDA_Final.xlsx"

IMAGE_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.60
SUPPORTED_FORMATS = ["jpg", "jpeg", "png"]
NUTRITION_SHEET = "App Nutrition Lookup"

st.set_page_config(
    page_title="MealMetrics | Food & Nutrition",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# 2. PROFESSIONAL UI STYLING
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --ink: #172B4D;
        --muted: #667085;
        --teal: #087E78;
        --teal-dark: #06645F;
        --teal-soft: #E8F5F3;
        --blue: #1877F2;
        --blue-dark: #155CC0;
        --blue-soft: #EAF2FF;
        --canvas: #F4F7FB;
        --line: #E6EAF0;
        --white: #FFFFFF;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: var(--canvas);
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background: rgba(246, 248, 250, 0.92);
    }

    .block-container {
        max-width: 1320px;
        padding-left: 2.2rem;
        padding-right: 2.2rem;
        padding-top: 1.7rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4 {
        font-family: 'Manrope', sans-serif !important;
        color: var(--ink) !important;
        letter-spacing: -0.035em;
    }

    h1 { font-size: 2.15rem !important; font-weight: 800 !important; }
    h2 { font-size: 1.55rem !important; font-weight: 750 !important; }
    h3 { font-size: 1.12rem !important; font-weight: 700 !important; }

    p, label, li {
        color: #344054;
    }

    .brand-row {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 1.1rem;
    }

    .brand-mark {
        width: 46px;
        height: 46px;
        border-radius: 14px;
        background: #E3F2F0;
        color: var(--teal-dark);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
    }

    .brand-name {
        font-family: 'Manrope', sans-serif;
        font-size: 1.12rem;
        font-weight: 800;
        color: var(--ink);
        line-height: 1.2;
    }

    .brand-caption {
        color: var(--muted);
        font-size: 0.78rem;
        margin-top: 3px;
    }

    .hero {
        background: linear-gradient(115deg, #102B46 0%, #164E59 100%);
        border-radius: 22px;
        padding: 2rem 2.1rem;
        color: white;
        margin: 0.5rem 0 1.5rem 0;
        box-shadow: 0 12px 30px rgba(16, 43, 70, 0.10);
    }

    .hero-eyebrow {
        color: #A8DCD5;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.70rem;
        font-weight: 700;
        margin-bottom: 0.65rem;
    }

    .hero-title {
        color: white;
        font-family: 'Manrope', sans-serif;
        font-size: 1.85rem;
        font-weight: 800;
        line-height: 1.2;
        margin-bottom: 0.55rem;
    }

    .hero-copy {
        color: #D6E5EA;
        font-size: 0.96rem;
        max-width: 690px;
        line-height: 1.6;
    }

    .section-kicker {
        color: var(--teal);
        text-transform: uppercase;
        letter-spacing: 0.10em;
        font-size: 0.70rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }

    .panel {
        background: white;
        border: 1px solid var(--line);
        border-radius: 17px;
        padding: 1.15rem 1.25rem;
        box-shadow: 0 3px 12px rgba(16, 24, 40, 0.035);
        margin-bottom: 1rem;
    }

    .panel-title {
        font-family: 'Manrope', sans-serif;
        color: var(--ink);
        font-size: 1.03rem;
        font-weight: 750;
        margin-bottom: 0.2rem;
    }

    .panel-subtitle {
        color: var(--muted);
        font-size: 0.83rem;
        line-height: 1.5;
    }

    .food-title {
        color: var(--ink);
        font-family: 'Manrope', sans-serif;
        font-size: 1.55rem;
        font-weight: 800;
        margin-bottom: 0.15rem;
    }

    .muted {
        color: var(--muted);
        font-size: 0.84rem;
    }

    .calorie-card {
        background: linear-gradient(135deg, #E8F5F3 0%, #F2FAF8 100%);
        border: 1px solid #CDE8E3;
        border-radius: 17px;
        padding: 1.15rem 1.25rem;
        min-height: 150px;
    }

    .calorie-label {
        color: #356A68;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .calorie-value {
        color: #075E59;
        font-family: 'Manrope', sans-serif;
        font-size: 2.45rem;
        font-weight: 800;
        line-height: 1.2;
        margin: 0.35rem 0;
    }

    .calorie-foot {
        color: #527876;
        font-size: 0.78rem;
    }

    .nutrient-card {
        background: white;
        border: 1px solid var(--line);
        border-radius: 15px;
        padding: 1rem;
        min-height: 105px;
    }

    .nutrient-value {
        color: var(--ink);
        font-family: 'Manrope', sans-serif;
        font-size: 1.45rem;
        font-weight: 800;
    }

    .nutrient-label {
        color: var(--muted);
        font-size: 0.78rem;
        margin-top: 0.1rem;
    }

    /* Consistent, high-contrast primary actions */
    .stButton > button[kind="primary"] {
        background: #087E78 !important;
        border: 1px solid #087E78 !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
    }
    .stButton > button[kind="primary"] p,
    .stButton > button[kind="primary"] span {
        color: #FFFFFF !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #06645F !important;
        border-color: #06645F !important;
        color: #FFFFFF !important;
    }
    .stButton > button:not([kind="primary"]) {
        border-radius: 10px !important;
        border-color: #D5DFEA !important;
    }
    [data-testid="stTabs"] [role="tab"] {
        border-radius: 10px 10px 0 0;
        font-weight: 650;
    }
    [data-testid="stTabs"] [role="tab'][aria-selected="true"] {
        color: #1877F2 !important;
        border-bottom-color: #1877F2 !important;
    }
    .dish-chip {
        display: inline-block;
        background: #EAF2FF;
        color: #174EA6;
        border: 1px solid #D5E4FF;
        border-radius: 999px;
        padding: 6px 11px;
        margin: 4px 5px 4px 0;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .scope-card {
        background: linear-gradient(110deg, #EDF4FF 0%, #F0FAF8 100%);
        border: 1px solid #DCE8F5;
        border-radius: 16px;
        padding: 1rem 1.2rem;
        margin: 1rem 0 1.3rem 0;
    }
    .scope-title { color: #172B4D; font-weight: 750; font-size: 0.96rem; }
    .scope-copy { color: #52647C; font-size: 0.82rem; margin-top: 3px; }
    .status-pill {
        display: inline-block;
        background: #E8F5F3;
        color: #08736D;
        border-radius: 999px;
        padding: 5px 10px;
        font-size: 0.72rem;
        font-weight: 700;
    }

    .disclaimer {
        color: #667085;
        background: #F1F4F7;
        border: 1px solid #E5EAF0;
        border-radius: 11px;
        padding: 0.75rem 0.9rem;
        font-size: 0.76rem;
        line-height: 1.5;
    }

    div[data-testid="stFileUploader"] {
        background: #FBFCFD;
        border: 1px dashed #B9C8D4;
        border-radius: 14px;
        padding: 0.8rem;
    }

    .stButton > button, .stDownloadButton > button {
        border-radius: 10px;
        font-weight: 700;
        min-height: 2.65rem;
        border: 1px solid #D0D5DD;
    }

    .stButton > button[kind="primary"] {
        background: var(--teal);
        border-color: var(--teal);
        color: white;
    }

    .stButton > button[kind="primary"]:hover {
        background: var(--teal-dark);
        border-color: var(--teal-dark);
        color: white;
    }

    div[data-testid="stTabs"] button {
        font-weight: 700;
        color: #667085;
    }

    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--teal-dark);
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 0.85rem 1rem;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 12px;
        overflow: hidden;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    @media (max-width: 700px) {
        .block-container { padding-left: 1rem; padding-right: 1rem; }
        .hero { padding: 1.4rem; }
        .hero-title { font-size: 1.45rem; }
        h1 { font-size: 1.7rem !important; }
    }
    
    .dish-chip {
    display: block;
    background: #EAF3FF;
    color: #2457A6;
    border: 1px solid #D5E4FA;
    border-radius: 9px;
    padding: 10px 8px;
    text-align: center;
    font-size: 13px;
    font-weight: 600;
    transition: all 0.2s ease;}

    .dish-chip:hover {
    background: #DCEBFF;
    border-color: #9DBDEB;}
    
    
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. REQUIRED FILE CHECKS AND DATA LOADING
# ============================================================

required_files = {
    "Trained model": MODEL_PATH,
    "Class labels": CLASS_NAMES_PATH,
    "Nutrition workbook": NUTRITION_PATH,
}

missing_files = [
    f"{label}: {path}" for label, path in required_files.items() if not path.is_file()
]

if missing_files:
    st.error("Some required application files are missing:")
    for item in missing_files:
        st.write(f"- {item}")
    st.stop()


@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path, compile=False)


@st.cache_data
def load_class_names(class_names_path):
    with open(class_names_path, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_model_config(config_path):
    path = Path(config_path)
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_nutrition_data(workbook_path):
    data = pd.read_excel(
        workbook_path,
        sheet_name=NUTRITION_SHEET,
    )

    required_columns = [
        "food_name",
        "servings_unit",
        "standard_serving_weight_g",
        "energy_kcal",
        "protein_g",
        "carb_g",
        "fat_g",
        "fibre_g",
        "standard_serving_energy_kcal",
        "standard_serving_protein_g",
        "standard_serving_carb_g",
        "standard_serving_fat_g",
        "standard_serving_fibre_g",
    ]

    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise ValueError(
            "Nutrition workbook is missing required columns: " + ", ".join(missing)
        )

    data["food_name"] = data["food_name"].astype(str).str.strip().str.lower()
    return data.set_index("food_name", drop=False)


try:
    model = load_model(str(MODEL_PATH))
    class_names = load_class_names(str(CLASS_NAMES_PATH))
    model_config = load_model_config(str(CONFIG_PATH))
    nutrition_df = load_nutrition_data(str(NUTRITION_PATH))
except Exception as error:
    st.error("The application could not load one or more required resources.")
    st.exception(error)
    st.stop()


if model.output_shape[-1] != len(class_names):
    st.error("The number of model outputs does not match the saved class labels.")
    st.stop()


# ============================================================
# 4. SESSION STATE
# ============================================================

if "prediction_history" not in st.session_state:
    st.session_state["prediction_history"] = []


# ============================================================
# 5. IMAGE AND PREDICTION HELPERS
# ============================================================


def format_food_name(food_name):
    return str(food_name).replace("_", " ").title()


def get_image_id(image_bytes):
    return hashlib.sha256(image_bytes).hexdigest()


def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    image_array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(image_array, axis=0)


def predict_food(image):
    predictions = model.predict(preprocess_image(image), verbose=0)[0]
    top_indices = np.argsort(predictions)[-3:][::-1]

    return [
        {
            "food": class_names[int(index)],
            "confidence": float(predictions[index]),
        }
        for index in top_indices
    ]


def get_nutrition_record(food_name):
    key = str(food_name).strip().lower()
    if key not in nutrition_df.index:
        return None
    return nutrition_df.loc[key]


def calculate_nutrition(record, mode, quantity):
    if mode == "Standard serving":
        columns = {
            "Calories": "standard_serving_energy_kcal",
            "Protein": "standard_serving_protein_g",
            "Carbohydrates": "standard_serving_carb_g",
            "Fat": "standard_serving_fat_g",
            "Fibre": "standard_serving_fibre_g",
        }
        multiplier = quantity
        return {
            label: float(record[column]) * multiplier
            for label, column in columns.items()
        }

    columns = {
        "Calories": "energy_kcal",
        "Protein": "protein_g",
        "Carbohydrates": "carb_g",
        "Fat": "fat_g",
        "Fibre": "fibre_g",
    }
    factor = quantity / 100.0
    return {label: float(record[column]) * factor for label, column in columns.items()}


def add_history_record(
    source,
    image_id,
    filename,
    results,
    selected_dish,
    status,
    nutrition_summary=None,
):
    history = st.session_state["prediction_history"]

    record = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Source": source,
        "Filename": filename,
        "Top Prediction": results[0]["food"],
        "Top Confidence (%)": round(results[0]["confidence"] * 100, 2),
        "Second Prediction": results[1]["food"],
        "Second Confidence (%)": round(results[1]["confidence"] * 100, 2),
        "Third Prediction": results[2]["food"],
        "Third Confidence (%)": round(results[2]["confidence"] * 100, 2),
        "User Selection": selected_dish,
        "Confirmation Status": status,
    }

    if nutrition_summary:
        record.update(nutrition_summary)

    record["_image_id"] = image_id

    existing_index = next(
        (
            index
            for index, item in enumerate(history)
            if item["_image_id"] == image_id and item["Source"] == source
        ),
        None,
    )

    if existing_index is None:
        history.append(record)
    else:
        history[existing_index] = record

    st.session_state["prediction_history"] = history


# ============================================================
# 6. DISPLAY COMPONENTS
# ============================================================


def render_nutrition_dashboard(food_name, source, image_id):
    record = get_nutrition_record(food_name)

    st.markdown(
        '<div class="section-kicker">Nutrition analysis</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="panel-title">Your food, quantified.</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="panel-subtitle">Choose a serving size or enter a custom weight '
        "to calculate estimated nutritional values.</div>",
        unsafe_allow_html=True,
    )

    if record is None:
        st.warning(
            "Nutrition information is not available for this selected food category."
        )
        return None

    unit = str(record["servings_unit"])
    default_weight = float(record["standard_serving_weight_g"])

    mode = st.radio(
        "Quantity method",
        ["Standard serving", "Custom weight"],
        horizontal=True,
        key=f"{source}_{image_id}_quantity_mode",
    )

    if mode == "Standard serving":
        col_a, col_b = st.columns([1, 2])
        with col_a:
            servings = st.number_input(
                f"Number of servings",
                min_value=0.5,
                max_value=20.0,
                value=1.0,
                step=0.5,
                key=f"{source}_{image_id}_servings",
            )
        total_weight = default_weight * servings
        with col_b:
            st.markdown(
                f"""
                <div class="panel" style="margin-top: 1.85rem; padding: 0.75rem 1rem;">
                    <div class="muted">Estimated quantity</div>
                    <div style="font-family:Manrope;font-size:1.15rem;font-weight:800;color:#172B4D;">
                        {total_weight:,.0f} g
                    </div>
                    <div class="muted">{servings:g} × {unit}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        quantity = servings
    else:
        grams = st.number_input(
            "Food weight (grams)",
            min_value=1.0,
            max_value=5000.0,
            value=float(default_weight),
            step=10.0,
            key=f"{source}_{image_id}_grams",
        )
        st.caption(
            f"Reference portion: {default_weight:,.0f} g per {unit}. "
            "Enter the actual weight if known."
        )
        quantity = grams

    values = calculate_nutrition(record, mode, quantity)

    if mode == "Standard serving":
        quantity_caption = (
            f"{servings:g} serving(s) · approximately {total_weight:,.0f} g"
        )
    else:
        quantity_caption = f"{grams:,.0f} g"

    st.markdown(
        f"""
        <div class="calorie-card">
            <div class="calorie-label">Estimated energy</div>
            <div class="calorie-value">{values['Calories']:,.0f} <span style="font-size:1rem;font-weight:700;">kcal</span></div>
            <div class="calorie-foot">{quantity_caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    nutrient_columns = st.columns(4)
    nutrient_items = [
        ("Protein", values["Protein"], "g"),
        ("Carbohydrates", values["Carbohydrates"], "g"),
        ("Fat", values["Fat"], "g"),
        ("Fibre", values["Fibre"], "g"),
    ]

    for column, (label, value, unit_label) in zip(nutrient_columns, nutrient_items):
        with column:
            st.markdown(
                f"""
                <div class="nutrient-card">
                    <div class="nutrient-value">{value:,.1f}<span style="font-size:0.8rem;"> {unit_label}</span></div>
                    <div class="nutrient-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="disclaimer">Nutrition values are estimates based on the selected food category '
        "and quantity. Actual values may vary with ingredients, recipe and preparation method.</div>",
        unsafe_allow_html=True,
    )

    return {
        "Estimated Calories (kcal)": round(values["Calories"], 1),
        "Protein (g)": round(values["Protein"], 1),
        "Carbohydrates (g)": round(values["Carbohydrates"], 1),
        "Fat (g)": round(values["Fat"], 1),
        "Fibre (g)": round(values["Fibre"], 1),
        "Quantity Method": mode,
        "Quantity": quantity_caption,
    }


def render_prediction_results(source, image, image_bytes, filename):
    image_id = get_image_id(image_bytes)
    state_key = f"{source.lower()}_prediction"

    previous_state = st.session_state.get(state_key)
    if previous_state is not None and previous_state.get("image_id") != image_id:
        st.session_state.pop(state_key, None)

    current_state = st.session_state.get(state_key)
    if current_state is None:
        current_state = {
            "image_id": image_id,
            "results": None,
            "decision": None,
            "selected_dish": None,
            "history_recorded": False,
            "nutrition_summary": None,
        }
        st.session_state[state_key] = current_state

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown(
            '<div class="section-kicker">Image preview</div>', unsafe_allow_html=True
        )
        st.image(image, use_container_width=True)
        st.caption(filename)

        if st.button(
            "Identify food",
            type="primary",
            use_container_width=True,
            key=f"{source}_identify_button",
        ):
            with st.spinner("Analyzing your food image..."):
                results = predict_food(image)

            current_state = {
                "image_id": image_id,
                "results": results,
                "decision": None,
                "selected_dish": None,
                "history_recorded": False,
                "nutrition_summary": None,
            }
            st.session_state[state_key] = current_state
            st.rerun()

    current_state = st.session_state[state_key]
    results = current_state.get("results")

    with right:
        st.markdown(
            '<div class="section-kicker">Food identification</div>',
            unsafe_allow_html=True,
        )

        if not results:
            st.markdown(
                """
                <div class="panel" style="min-height:245px;display:flex;flex-direction:column;justify-content:center;">
                    <div style="font-size:2rem;margin-bottom:0.7rem;">✦</div>
                    <div class="panel-title">Ready to identify your dish</div>
                    <div class="panel-subtitle">Select <b>Identify food</b> to analyze the image and view the model's top suggestions.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

        top_prediction = results[0]
        st.markdown(
            f'<div class="food-title">{format_food_name(top_prediction["food"])}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<span class="status-pill">Top match · {top_prediction["confidence"]:.1%} confidence</span>',
            unsafe_allow_html=True,
        )
        st.write("")

        with st.expander("View all top 3 predictions", expanded=True):
            for rank, result in enumerate(results, start=1):
                st.markdown(
                    f"**{rank}. {format_food_name(result['food'])}**"
                    f"  ·  {result['confidence']:.1%}"
                )
                st.progress(max(0.0, min(1.0, result["confidence"])))

        st.caption(
            "Confidence is the model's score for each class, not a guarantee of correctness."
        )

        decision = current_state["decision"]

        if decision is None:
            if top_prediction["confidence"] >= CONFIDENCE_THRESHOLD:
                st.markdown("**Does this look correct?**")
                yes_col, no_col = st.columns(2)
                with yes_col:
                    if st.button(
                        "Confirm dish",
                        type="primary",
                        use_container_width=True,
                        key=f"{source}_confirm_top",
                    ):
                        current_state["decision"] = "confirmed_top"
                        current_state["selected_dish"] = top_prediction["food"]
                        st.rerun()
                with no_col:
                    if st.button(
                        "Choose another",
                        use_container_width=True,
                        key=f"{source}_choose_another",
                    ):
                        current_state["decision"] = "choose_another"
                        st.rerun()
            else:
                st.warning(
                    f"The leading score is below the {CONFIDENCE_THRESHOLD:.0%} review threshold. "
                    "Please confirm the correct dish from the suggestions."
                )
                current_state["decision"] = "choose_another"
                st.rerun()

        if current_state["decision"] == "choose_another":
            options = [item["food"] for item in results]
            options.append("None of these is correct")
            selected = st.selectbox(
                "Choose the correct dish",
                options,
                format_func=lambda value: (
                    format_food_name(value)
                    if value != "None of these is correct"
                    else value
                ),
                key=f"{source}_dish_selection",
            )
            if st.button(
                "Confirm selection",
                type="primary",
                key=f"{source}_confirm_selection",
            ):
                if selected == "None of these is correct":
                    current_state["decision"] = "none_correct"
                    current_state["selected_dish"] = None
                else:
                    current_state["decision"] = "confirmed_alternative"
                    current_state["selected_dish"] = selected
                st.rerun()

        decision = current_state["decision"]
        selected_dish = current_state["selected_dish"]

        if decision in ("confirmed_top", "confirmed_alternative"):
            st.success(f"Food confirmed: {format_food_name(selected_dish)}")

            # Streamlit reruns the script after every widget interaction.
            # Re-render the dashboard on every rerun so its controls and
            # calculated values remain visible when quantity changes.
            st.markdown("---")
            summary = render_nutrition_dashboard(
                selected_dish,
                source,
                image_id,
            )
            current_state["nutrition_summary"] = summary

            # Upsert the history row so it reflects the latest quantity choice.
            status = (
                "User confirmed leading prediction"
                if decision == "confirmed_top"
                else "User selected an alternative"
            )
            add_history_record(
                source,
                image_id,
                filename,
                results,
                selected_dish,
                status,
                summary,
            )
            current_state["history_recorded"] = True

        elif decision == "none_correct":
            st.warning(
                "No suggested category was confirmed. Try another image with the food clearly visible."
            )
            if not current_state.get("history_recorded", False):
                add_history_record(
                    source,
                    image_id,
                    filename,
                    results,
                    "None of these is correct",
                    "No suggested class matched",
                )
                current_state["history_recorded"] = True


# ============================================================
# 7. APPLICATION HEADER
# ============================================================

st.markdown(
    """
    <div class="brand-row">
        <div class="brand-mark">🍽️</div>
        <div>
            <div class="brand-name">MealMetrics</div>
            <div class="brand-caption">Explore your food. Understand its nutrition.</div>
        </div>
    </div>
    <div class="hero">
        <div class="hero-eyebrow">Food & nutrition, made understandable</div>
        <div class="hero-title">Know what’s on your plate.</div>
        <div class="hero-copy">
            Upload or capture a meal photo, confirm the dish, and explore estimated
            calories, protein, carbohydrates, fat and fibre for your chosen portion.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 8. NAVIGATION
# ============================================================

identify_tab, camera_tab, history_tab, about_tab = st.tabs(
    ["🍽️  Food analyzer", "📷  Camera", "📊  History", "ℹ️  About"]
)


# ============================================================
# 9. FOOD ANALYZER TAB
# ============================================================

with identify_tab:
    st.markdown(
        '<div class="section-kicker">Start with a photo</div>', unsafe_allow_html=True
    )
    st.header("Analyze a food image")
    st.markdown(
        '<div class="panel-subtitle">Upload a clear image of one dish. For best results, keep the food well-lit and visible.</div>',
        unsafe_allow_html=True,
    )

    supported_dishes = [format_food_name(name) for name in class_names]
    # dish_chips = "".join(
    #     f'<span class="dish-chip">{name}</span>' for name in supported_dishes
    # )

    dish_chips = "".join(
        f'<span class="dish-chip">{dish.replace("_", " ").title()}</span>'
        for dish in supported_dishes
    )

    # st.markdown(
    #     f"""
    #     <div class="scope-card">
    #         <div class="scope-title">Currently supports {len(supported_dishes)} Indian dishes</div>
    #         <div class="scope-copy">MealMetrics identifies dishes from this supported list. Choose a clear photo of one dish for the best result.</div>
    #         <div style="margin-top:8px;">{dish_chips}</div>
    #     </div>
    #     """,
    #     unsafe_allow_html=True,
    # )

    # st.markdown(
    #     f"""
    #     <div class="scope-card">
    #         <div class="scope-title">
    #             Currently supports {len(supported_dishes)} Indian dishes
    #         </div>

    #         <div class="scope-copy">
    #             MealMetrics identifies dishes from these supported categories.
    #             Choose a clear photo of one dish for the best result.
    #         </div>

    #         <div style="
    #             display: grid;
    #             grid-template-columns: repeat(4, minmax(0, 1fr));
    #             gap: 10px;
    #             margin-top: 16px;
    #         ">
    #             {dish_chips}
    #         </div>
    #     </div>
    #     """,
    #     unsafe_allow_html=True,
    # )

    st.markdown(
        f"""<div class="scope-card">
    <div class="scope-title">Currently supports {len(supported_dishes)} Indian dishes</div>
    <div class="scope-copy">
    MealMetrics identifies dishes from these supported categories.
    Choose a clear photo of one dish for the best result.
    </div>
    <div style="display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin-top: 16px;">
    {dish_chips}
    </div>
    </div>""",
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose a food image",
        type=SUPPORTED_FORMATS,
        key="food_image_upload",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        try:
            image = Image.open(BytesIO(image_bytes))
            image.load()
            render_prediction_results(
                source="Upload",
                image=image,
                image_bytes=image_bytes,
                filename=uploaded_file.name,
            )
        except (UnidentifiedImageError, OSError, ValueError):
            st.error(
                "This image could not be opened. Please upload a valid JPG, JPEG or PNG file."
            )
    else:
        st.markdown(
            """
            <div class="panel" style="text-align:center;padding:2.2rem 1rem;">
                <div style="font-size:2rem;">↥</div>
                <div class="panel-title">Your food analysis starts here</div>
                <div class="panel-subtitle">Upload an image above to identify the dish and estimate its nutrition.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# 10. CAMERA TAB
# ============================================================

with camera_tab:
    st.markdown(
        '<div class="section-kicker">Use your camera</div>', unsafe_allow_html=True
    )
    st.header("Capture your meal")
    st.markdown(
        '<div class="panel-subtitle">Take a photo of your dish directly from your device.</div>',
        unsafe_allow_html=True,
    )

    camera_image = st.camera_input(
        "Take a food photo",
        key="food_camera",
    )

    if camera_image is not None:
        camera_bytes = camera_image.getvalue()
        try:
            image = Image.open(BytesIO(camera_bytes))
            image.load()
            render_prediction_results(
                source="Camera",
                image=image,
                image_bytes=camera_bytes,
                filename="Camera capture",
            )
        except (UnidentifiedImageError, OSError, ValueError):
            st.error("The captured image could not be read. Please try again.")
    else:
        st.info(
            "Your camera preview will appear here after you grant camera permission."
        )


# ============================================================
# 11. HISTORY TAB
# ============================================================

with history_tab:
    st.markdown(
        '<div class="section-kicker">Your recent activity</div>', unsafe_allow_html=True
    )
    st.header("Analysis history")
    st.markdown(
        '<div class="panel-subtitle">Confirmed food selections from the current app session.</div>',
        unsafe_allow_html=True,
    )

    history = st.session_state["prediction_history"]

    if history:
        visible_history = [
            {key: value for key, value in record.items() if key != "_image_id"}
            for record in history
        ]
        history_df = pd.DataFrame(visible_history)

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
        )

        csv_data = history_df.to_csv(index=False).encode("utf-8-sig")
        col_download, col_clear = st.columns([1, 1])

        with col_download:
            st.download_button(
                "Download history (CSV)",
                data=csv_data,
                file_name="mealmetrics_analysis_history.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col_clear:
            if st.button(
                "Clear history",
                use_container_width=True,
                key="clear_history_button",
            ):
                st.session_state["prediction_history"] = []
                for state_key in ["upload_prediction", "camera_prediction"]:
                    if state_key in st.session_state:
                        st.session_state[state_key]["history_recorded"] = True
                st.rerun()
    else:
        st.markdown(
            """
            <div class="panel" style="text-align:center;padding:2rem;">
                <div style="font-size:1.8rem;">▤</div>
                <div class="panel-title">No analyses yet</div>
                <div class="panel-subtitle">Your confirmed food analyses will appear here.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# 12. ABOUT TAB
# ============================================================

with about_tab:
    st.markdown(
        '<div class="section-kicker">About the project</div>', unsafe_allow_html=True
    )
    st.header("About MealMetrics")
    st.markdown("""
        **MealMetrics** is an Indian food image analysis and nutrition estimation
        application. It brings together an image classification model and a curated
        nutrition lookup dataset to help users understand the food and portion they
        are viewing.

        The application is designed as a practical two-part workflow: first identify
        and confirm a dish, then estimate its nutritional values using a standard
        serving or a custom weight.
        """)

    st.subheader("What you can do")
    feature_cols = st.columns(2, gap="large")
    features = [
        (
            "📷",
            "Upload or capture",
            "Choose an existing JPG, JPEG or PNG photo, or take a picture directly with your device camera.",
        ),
        (
            "🍽️",
            "Identify a dish",
            "The image classifier evaluates the photo and displays its three highest-scoring dish suggestions.",
        ),
        (
            "🥗",
            "Review nutrition",
            "After you confirm the dish, view estimated energy, protein, carbohydrates, fat and fibre.",
        ),
        (
            "⚖️",
            "Adjust the portion",
            "Use a standard serving quantity or enter a custom food weight in grams to recalculate the estimates.",
        ),
        (
            "📊",
            "Keep a history",
            "Review confirmed analyses during the current session and export the history as a CSV file.",
        ),
    ]
    for i, (icon, title, description) in enumerate(features):
        with feature_cols[i % 2]:
            st.markdown(
                f"""
                <div class="panel" style="min-height:130px;">
                    <div style="font-size:1.35rem;margin-bottom:6px;">{icon}</div>
                    <div class="panel-title">{title}</div>
                    <div class="panel-subtitle">{description}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.subheader("Supported food categories")
    st.markdown(
        f"This version recognises **{len(class_names)} food categories**. It is currently limited to the dishes listed below; dishes outside this list may be assigned to the closest available category."
    )
    st.markdown(
        "".join(
            f'<span class="dish-chip">{format_food_name(name)}</span>'
            for name in class_names
        ),
        unsafe_allow_html=True,
    )

    st.subheader("Model performance")
    metric_cols = st.columns(4)
    metrics = [
        (str(len(class_names)), "Supported food categories"),
        ("82.57%", "Test Top-1 accuracy"),
        ("93.94%", "Test Top-3 accuracy"),
        ("224 × 224", "Image input size"),
    ]
    for column, (value, label) in zip(metric_cols, metrics):
        with column:
            st.metric(label, value)

    st.subheader("How the nutrition estimate is calculated")
    st.markdown("""
        - **Standard serving:** the app uses the nutrition values associated with one
          reference serving and multiplies them by the number of servings selected.
        - **Custom weight:** the app scales the per-100-gram nutrition values to the
          weight entered by the user.
        - **Important:** nutrition values are estimates. Actual values can differ
          depending on recipe, ingredients, oil, preparation method and portion size.
        """)

    st.subheader("Technology and project details")
    st.markdown("""
        - **Image classification:** fine-tuned MobileNetV2 using transfer learning.
        - **Image preparation:** photos are converted to RGB and resized to 224 × 224 pixels.
        - **Application interface:** Streamlit.
        - **Nutrition lookup:** a prepared Indian food nutrition dataset containing serving-level and per-100-gram values.
        - **Prediction review:** the app shows the top three model suggestions and allows the user to confirm the dish or select another suggestion.
        """)
    st.markdown(
        '<div class="disclaimer">MealMetrics is intended for general information and learning. It does not measure the exact contents of a meal or replace professional dietary advice. The classifier only covers the supported food categories shown above.</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# 13. FOOTER
# ============================================================

st.markdown(
    """
    <div style="border-top:1px solid #E6EAF0;margin-top:2.5rem;padding-top:1rem;text-align:center;">
        <div style="color:#667085;font-size:0.75rem;">
            MealMetrics · Explore your food. Understand its nutrition.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
