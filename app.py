import streamlit as st
import sqlite3
import pandas as pd
import json
import re
import matplotlib.pyplot as plt
from datetime import datetime
import requests
import os

# ============================================================
# CONFIG
# ============================================================

DB_FILE_NAME = "project_dessert_breaks.db"

SPOON_TABLE = "spoonacular_desserts_master"
MEALDB_TABLE = "themealdb_readable_recipes"
ALLERGY_TABLE = "ingredient_allergy_reference_final"
ML_TABLE = "ml_recipe_clusters"
USER_SUB_TABLE = "user_ingredient_substitutions"

# ============================================================
# BASIC HELPERS
# ============================================================

def safe_json_loads(x, default=None):
    if default is None:
        default = []
    if x is None:
        return default
    try:
        if pd.isna(x):
            return default
    except:
        pass
    if isinstance(x, (list, dict)):
        return x
    if isinstance(x, str):
        x = x.strip()
        if not x:
            return default
        try:
            return json.loads(x)
        except:
            return default
    return default


def normalize_text(text):
    if text is None:
        return None
    try:
        if pd.isna(text):
            return None
    except:
        pass
    text = str(text).lower().strip()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else None


def parse_json_list(x):
    val = safe_json_loads(x, [])
    return val if isinstance(val, list) else []


def parse_json_dict(x):
    val = safe_json_loads(x, {})
    return val if isinstance(val, dict) else {}


def normalize_ingredient_list(lst):
    if not isinstance(lst, list):
        return []
    cleaned = []
    for x in lst:
        x = normalize_text(x)
        if x:
            cleaned.append(x)
    return list(dict.fromkeys(cleaned))


def format_nutrients(n):
    if not n:
        return "N/A"
    if isinstance(n, dict):
        return ", ".join([f"{k}: {v}" for k, v in n.items()])
    return str(n)


def load_table_if_exists(conn, table_name):
    try:
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        df["source_table"] = table_name
        return df
    except:
        return pd.DataFrame()


# ============================================================
# STANDARDIZE RECIPE TABLES
# ============================================================

def standardize_spoon(df):
    if df.empty:
        return pd.DataFrame(columns=[
            "recipe_id", "recipe_name", "ingredients_clean", "ingredients_combined",
            "image_url", "nutrients", "instructions", "source_url", "source_table"
        ])

    out = df.copy()

    out["recipe_id"] = out["id"] if "id" in out.columns else None
    out["recipe_name"] = out["recipe_name"] if "recipe_name" in out.columns else None
    out["ingredients_combined"] = out["ingredients_combined"] if "ingredients_combined" in out.columns else None
    out["image_url"] = out["image_url"] if "image_url" in out.columns else None
    out["instructions"] = out["instructions"] if "instructions" in out.columns else None
    out["source_url"] = out["source_url"] if "source_url" in out.columns else None

    if "ingredients_clean" in out.columns:
        out["ingredients_clean"] = out["ingredients_clean"].apply(parse_json_list)
    else:
        out["ingredients_clean"] = [[] for _ in range(len(out))]

    out["ingredients_clean"] = out["ingredients_clean"].apply(normalize_ingredient_list)

    if "nutrients" in out.columns:
        out["nutrients"] = out["nutrients"].apply(parse_json_dict)
    else:
        out["nutrients"] = [{} for _ in range(len(out))]

    return out[[
        "recipe_id", "recipe_name", "ingredients_clean", "ingredients_combined",
        "image_url", "nutrients", "instructions", "source_url", "source_table"
    ]]


def standardize_themealdb(df):
    if df.empty:
        return pd.DataFrame(columns=[
            "recipe_id", "recipe_name", "ingredients_clean", "ingredients_combined",
            "image_url", "nutrients", "instructions", "source_url", "source_table"
        ])

    out = df.copy()

    def first_existing(cols):
        for c in cols:
            if c in out.columns:
                return c
        return None

    id_col = first_existing(["recipe_id", "idMeal", "id"])
    name_col = first_existing(["recipe_name", "Meal Name", "meal_name", "strMeal"])
    ingredients_clean_col = first_existing(["ingredients_clean", "Ingredients Clean"])
    ingredients_combined_col = first_existing(["ingredients_combined", "Ingredients Combined"])
    image_col = first_existing(["image_url", "Image", "strMealThumb"])
    nutrients_col = first_existing(["nutrients", "Nutrients"])
    instructions_col = first_existing(["instructions", "Instructions", "strInstructions"])
    source_url_col = first_existing(["source_url", "Source", "strSource"])

    out["recipe_id"] = out[id_col] if id_col else None
    out["recipe_name"] = out[name_col] if name_col else None
    out["ingredients_combined"] = out[ingredients_combined_col] if ingredients_combined_col else None
    out["image_url"] = out[image_col] if image_col else None
    out["instructions"] = out[instructions_col] if instructions_col else None
    out["source_url"] = out[source_url_col] if source_url_col else None

    if ingredients_clean_col:
        out["ingredients_clean"] = out[ingredients_clean_col].apply(parse_json_list)
    elif ingredients_combined_col:
        out["ingredients_clean"] = out[ingredients_combined_col].apply(
            lambda x: normalize_ingredient_list(str(x).split("|")) if pd.notna(x) else []
        )
    else:
        out["ingredients_clean"] = [[] for _ in range(len(out))]

    out["ingredients_clean"] = out["ingredients_clean"].apply(normalize_ingredient_list)

    if nutrients_col:
        out["nutrients"] = out[nutrients_col].apply(parse_json_dict)
    else:
        out["nutrients"] = [{} for _ in range(len(out))]

    return out[[
        "recipe_id", "recipe_name", "ingredients_clean", "ingredients_combined",
        "image_url", "nutrients", "instructions", "source_url", "source_table"
    ]]


# ============================================================
# RECIPE SEARCH
# ============================================================

def find_recipes_by_ingredients(recipes_master, user_ingredients, top_n=5):
    user_ingredients_norm = normalize_ingredient_list(user_ingredients)

    if not user_ingredients_norm:
        return pd.DataFrame()

    user_set = set(user_ingredients_norm)
    rows = []

    for _, row in recipes_master.iterrows():
        recipe_ings = row.get("ingredients_clean", [])
        if not isinstance(recipe_ings, list):
            recipe_ings = []

        recipe_set = set(recipe_ings)
        matched = sorted(list(user_set.intersection(recipe_set)))

        if len(matched) == 0:
            continue

        match_ratio = len(matched) / max(len(user_set), 1)

        rows.append({
            "recipe_id": row.get("recipe_id"),
            "recipe_name": row.get("recipe_name"),
            "ingredients_clean": recipe_ings,
            "ingredients_combined": row.get("ingredients_combined"),
            "image_url": row.get("image_url"),
            "nutrients": row.get("nutrients"),
            "instructions": row.get("instructions"),
            "source_url": row.get("source_url"),
            "source_table": row.get("source_table"),
            "matched_ingredients": matched,
            "match_percentage": round(match_ratio * 100, 1)
        })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    result = result.sort_values(
        by=["match_percentage", "recipe_name"],
        ascending=[False, True]
    ).head(top_n).reset_index(drop=True)

    result.index = range(1, len(result) + 1)
    return result


def load_recipe_by_source_and_id(conn, recipe_id, source_table):
    try:
        if source_table == SPOON_TABLE:
            df = pd.read_sql(
                f"SELECT * FROM {SPOON_TABLE} WHERE id = ?",
                conn,
                params=[recipe_id]
            )
            if df.empty:
                return None
            row = df.iloc[0].to_dict()
            return {
                "recipe_id": row.get("id"),
                "recipe_name": row.get("recipe_name"),
                "ingredients_clean": normalize_ingredient_list(parse_json_list(row.get("ingredients_clean"))),
                "ingredients_combined": row.get("ingredients_combined"),
                "image_url": row.get("image_url"),
                "instructions": row.get("instructions"),
                "source_url": row.get("source_url"),
                "nutrients": parse_json_dict(row.get("nutrients")),
                "source_table": SPOON_TABLE
            }
        else:
            return None
    except:
        return None


# ============================================================
# ALLERGY DETECTION
# ============================================================

def detect_possible_allergies(recipe_ingredients, allergy_df):
    if allergy_df.empty:
        return pd.DataFrame()

    work = allergy_df.copy()

    possible_ing_col = None
    for c in ["ingredient_name", "ingredient", "trigger_ingredient", "ingredient_normalized"]:
        if c in work.columns:
            possible_ing_col = c
            break

    possible_type_col = None
    for c in ["allergy_type", "allergen_type", "allergy"]:
        if c in work.columns:
            possible_type_col = c
            break

    possible_sub_col = None
    for c in ["possible_substitutes", "substitute", "substitutes"]:
        if c in work.columns:
            possible_sub_col = c
            break

    if possible_ing_col is None:
        return pd.DataFrame()

    work["ingredient_ref_norm"] = work[possible_ing_col].apply(normalize_text)
    work["allergy_type_norm"] = work[possible_type_col].apply(normalize_text) if possible_type_col else None

    matches = []

    for ing in recipe_ingredients:
        ing_norm = normalize_text(ing)
        if not ing_norm:
            continue

        exact_hits = work[work["ingredient_ref_norm"] == ing_norm]
        contains_hits = work[
            work["ingredient_ref_norm"].apply(
                lambda x: (x in ing_norm or ing_norm in x) if isinstance(x, str) else False
            )
        ]

        hit_df = pd.concat([exact_hits, contains_hits], ignore_index=True).drop_duplicates()

        if not hit_df.empty:
            hit_df = hit_df.copy()
            hit_df["recipe_ingredient"] = ing
            matches.append(hit_df)

    if not matches:
        return pd.DataFrame()

    result = pd.concat(matches, ignore_index=True).drop_duplicates()

    keep_cols = ["recipe_ingredient"]
    if possible_ing_col in result.columns:
        keep_cols.append(possible_ing_col)
    if possible_type_col and possible_type_col in result.columns:
        keep_cols.append(possible_type_col)
    if possible_sub_col and possible_sub_col in result.columns:
        keep_cols.append(possible_sub_col)

    return result[keep_cols].drop_duplicates().reset_index(drop=True)


def filter_allergy_hits_for_user(allergy_hits, selected_allergies):
    if allergy_hits is None or allergy_hits.empty:
        return pd.DataFrame()

    if not selected_allergies:
        return pd.DataFrame()

    selected_allergies = [normalize_text(x) for x in selected_allergies if normalize_text(x)]

    filtered_rows = []

    for _, row in allergy_hits.iterrows():
        row_text = " ".join([str(x).lower() for x in row.values])
        keep = any(a in row_text for a in selected_allergies)
        if keep:
            filtered_rows.append(row.to_dict())

    if not filtered_rows:
        return pd.DataFrame()

    return pd.DataFrame(filtered_rows).drop_duplicates().reset_index(drop=True)


# ============================================================
# MACHINE LEARNING RECOMMENDATION
# ============================================================

def get_ml_recipe_recommendation(conn, recipe_id, top_n=3):
    try:
        df_ml_all = pd.read_sql(f"""
            SELECT id, recipe_name, cluster, cluster_name, pca_1, pca_2
            FROM {ML_TABLE}
        """, conn)
    except:
        return {
            "cluster_name": None,
            "similar_recipes": pd.DataFrame()
        }

    if df_ml_all.empty:
        return {
            "cluster_name": None,
            "similar_recipes": pd.DataFrame()
        }

    df_ml_all["id_str"] = df_ml_all["id"].astype(str)
    recipe_id_str = str(recipe_id)

    selected_ml = df_ml_all[df_ml_all["id_str"] == recipe_id_str]

    if selected_ml.empty:
        return {
            "cluster_name": None,
            "similar_recipes": pd.DataFrame()
        }

    selected_row = selected_ml.iloc[0]
    cluster_id = selected_row["cluster"]
    cluster_name = selected_row["cluster_name"]

    similar_recipes = df_ml_all[
        (df_ml_all["cluster"] == cluster_id) &
        (df_ml_all["id_str"] != recipe_id_str)
    ].copy()

    if similar_recipes.empty:
        df_candidates = df_ml_all[df_ml_all["id_str"] != recipe_id_str].copy()
        df_candidates["pca_distance"] = (
            (df_candidates["pca_1"] - selected_row["pca_1"]) ** 2 +
            (df_candidates["pca_2"] - selected_row["pca_2"]) ** 2
        ) ** 0.5

        similar_recipes = (
            df_candidates
            .sort_values("pca_distance")
            .head(top_n)
        )
    else:
        similar_recipes = similar_recipes.head(top_n)

    return {
        "cluster_name": cluster_name,
        "similar_recipes": similar_recipes[["recipe_name", "cluster_name"]]
    }


# ============================================================
# USER SUBSTITUTION BACKEND
# ============================================================

def create_user_substitutions_table(conn):
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {USER_SUB_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_ingredient TEXT,
            substitute_ingredient TEXT,
            amount_grams REAL,
            created_at TEXT
        )
    """)
    conn.commit()


def store_user_substitution(conn, original_ingredient, substitute_ingredient, amount_grams):
    create_user_substitutions_table(conn)

    conn.execute(f"""
        INSERT INTO {USER_SUB_TABLE}
        (original_ingredient, substitute_ingredient, amount_grams, created_at)
        VALUES (?, ?, ?, ?)
    """, (
        normalize_text(original_ingredient),
        normalize_text(substitute_ingredient),
        amount_grams,
        datetime.now().isoformat()
    ))

    conn.commit()

# ============================================================
# LLM SUBSTITUTION RECOMMENDATION BACKEND
# ============================================================

LLM_PROVIDER = "openai"

MODEL_NAMES = {
    "openai": "gpt-4o-mini"
}


def get_api_key(provider=LLM_PROVIDER):
    """
    Gets the API key for the selected LLM provider.

    In VS Code / Streamlit, we can use either:
    1. Streamlit secrets: .streamlit/secrets.toml
    2. Environment variables

    This keeps the key hidden and avoids hardcoding it in the project.
    """

    if provider == "openai":
        # First try Streamlit secrets
        try:
            if "OPENAI_API_KEY" in st.secrets:
                return st.secrets["OPENAI_API_KEY"]
        except Exception:
            pass

        # Then try environment variable
        return os.getenv("OPENAI_API_KEY")

    return None


def build_llm_substitution_prompt(recipe_name, ingredients, allergies=None, nutrition_goal=None):
    """
    Builds the prompt we send to the LLM.

    The LLM receives:
    - recipe name
    - original ingredients
    - user allergies
    - nutrition goal

    It returns substitute ingredients and recommended amounts.
    """

    return f"""
You are helping with a baking recipe substitution app.

Recipe name:
{recipe_name}

Original ingredients:
{ingredients}

User allergies:
{allergies or []}

User nutrition goal:
{nutrition_goal or "None"}

Please recommend ingredient substitutions and amounts.

Only recommend substitutions when they are useful for the user's allergy or nutrition goal.
For baking recipes, try to preserve texture, sweetness, moisture, and structure.

Return ONLY valid JSON in this exact format:

[
  {{
    "original_ingredient": "milk",
    "substitute_ingredient": "almond milk",
    "recommended_amount": "same amount as original milk",
    "reason": "This is dairy-free and keeps a similar liquid texture."
  }}
]
"""


def call_llm_for_substitutions(prompt, provider=LLM_PROVIDER):
    """
    Calls the selected LLM API.

    If the API key is missing, the API fails, or the model does not return valid JSON,
    the function returns an empty list instead of crashing the app.
    """

    api_key = get_api_key(provider)
    model = MODEL_NAMES.get(provider)

    if not api_key:
        st.warning("LLM API key is missing. Add OPENAI_API_KEY to Streamlit secrets or environment variables.")
        return []

    try:
        if provider == "openai":
            url = "https://api.openai.com/v1/chat/completions"

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            data = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful baking substitution assistant. Return valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.2
            }

            response = requests.post(
                url,
                headers=headers,
                json=data,
                timeout=30
            )

            if response.status_code != 200:
                st.error("LLM API call failed.")
                st.write("Status code:", response.status_code)
                st.write("Response:", response.text)
                return []

            response_json = response.json()
            llm_text = response_json["choices"][0]["message"]["content"]

        else:
            st.warning("Only OpenAI is connected in this VS Code version.")
            return []

        # Clean response in case the LLM wraps JSON in markdown
        llm_text = llm_text.replace("```json", "").replace("```", "").strip()

        return json.loads(llm_text)

    except json.JSONDecodeError:
        st.error("The LLM response was not valid JSON.")
        st.write("Raw LLM response:")
        st.write(llm_text)
        return []

    except Exception as e:
        st.error("LLM substitution failed.")
        st.write(e)
        return []


def get_llm_substitution_recommendations(recipe, allergies=None, nutrition_goal=None):
    """
    Connects the selected recipe to the LLM.

    This function is the main function we call inside the Streamlit app.
    """

    if recipe is None:
        return []

    prompt = build_llm_substitution_prompt(
        recipe_name=recipe.get("recipe_name"),
        ingredients=recipe.get("ingredients_combined"),
        allergies=allergies,
        nutrition_goal=nutrition_goal
    )

    return call_llm_for_substitutions(prompt)

# ============================================================
# STREAMLIT APP
# ============================================================

st.set_page_config(page_title="BakeSmart", layout="wide")

# ============================================================
# FIGMA-INSPIRED DESIGN STYLE
# ============================================================

st.markdown("""
<style>
/* Whole app background */
.stApp {
    background: linear-gradient(135deg, #FFF7ED 0%, #FFE4E6 100%);
    color: #17213C;
}

/* Hide Streamlit default top spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 4rem;
    max-width: 1200px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #FFF7ED;
    border-right: 1px solid #F2D4C4;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p {
    color: #17213C;
}

/* Main text */
h1 {
    color: #17213C;
    font-size: 52px !important;
    font-weight: 850 !important;
    letter-spacing: -1px;
}

h2 {
    color: #17213C;
    font-size: 34px !important;
    font-weight: 800 !important;
    letter-spacing: -0.5px;
}

h3 {
    color: #17213C;
    font-size: 24px !important;
    font-weight: 750 !important;
}

p, label, span, div {
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* Streamlit buttons */
.stButton > button {
    background-color: #17213C;
    color: white;
    border-radius: 999px;
    border: none;
    padding: 0.75rem 1.5rem;
    font-weight: 700;
    font-size: 16px;
    box-shadow: 0 8px 20px rgba(23, 33, 60, 0.18);
    transition: 0.2s ease;
}

.stButton > button:hover {
    background-color: #D28A5C;
    color: white;
    transform: translateY(-1px);
    box-shadow: 0 10px 24px rgba(210, 138, 92, 0.28);
}

/* Inputs */
textarea, input {
    border-radius: 18px !important;
    border: 1px solid #E8CBB8 !important;
    background-color: #FFFFFF !important;
}

div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea,
div[data-testid="stNumberInput"] input {
    border-radius: 18px !important;
    padding: 12px 16px !important;
}

/* Select boxes and multiselect */
div[data-testid="stMultiSelect"],
div[data-testid="stSelectbox"] {
    background-color: white;
    border-radius: 18px;
}

/* Dataframes */
div[data-testid="stDataFrame"] {
    background-color: white;
    border-radius: 18px;
    box-shadow: 0 8px 24px rgba(23, 33, 60, 0.08);
    padding: 8px;
}

/* Expander */
details {
    background-color: rgba(255, 255, 255, 0.75) !important;
    border-radius: 20px !important;
    border: 1px solid #F2D4C4 !important;
    box-shadow: 0 8px 24px rgba(23, 33, 60, 0.06);
}

/* Custom reusable classes */
.bakesmart-nav {
    background: rgba(255, 255, 255, 0.78);
    border: 1px solid #F2D4C4;
    border-radius: 24px;
    padding: 18px 26px;
    margin-bottom: 34px;
    box-shadow: 0 10px 30px rgba(23, 33, 60, 0.08);
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.bakesmart-logo {
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 28px;
    font-weight: 850;
    color: #17213C;
}

.logo-icon {
    width: 46px;
    height: 46px;
    background: #D28A5C;
    color: white;
    border-radius: 16px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
}

.nav-links {
    display: flex;
    gap: 28px;
    font-weight: 700;
    color: #17213C;
}

.hero-section {
    text-align: center;
    padding: 42px 20px 28px 20px;
}

.hero-title {
    font-size: 58px;
    line-height: 1.05;
    font-weight: 900;
    color: #17213C;
    margin-bottom: 16px;
    letter-spacing: -1.5px;
}

.hero-subtitle {
    font-size: 20px;
    color: #44516A;
    max-width: 760px;
    margin: 0 auto 24px auto;
}

.white-card {
    background: rgba(255, 255, 255, 0.86);
    border: 1px solid #F2D4C4;
    border-radius: 30px;
    padding: 34px;
    margin: 28px 0;
    box-shadow: 0 18px 45px rgba(23, 33, 60, 0.10);
}

.section-card {
    background: rgba(255, 255, 255, 0.86);
    border: 1px solid #F2D4C4;
    border-radius: 28px;
    padding: 30px;
    margin: 26px 0;
    box-shadow: 0 14px 34px rgba(23, 33, 60, 0.08);
}

.recipe-card {
    background: #FFFFFF;
    border: 1px solid #F2D4C4;
    border-radius: 26px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 12px 34px rgba(23, 33, 60, 0.10);
}

.recipe-title {
    font-size: 25px;
    font-weight: 800;
    color: #17213C;
    margin-bottom: 10px;
}

.badge {
    display: inline-block;
    background: #FFF1E8;
    color: #B76D3F;
    padding: 6px 12px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 800;
    margin-right: 8px;
    margin-bottom: 8px;
}

.match-badge {
    display: inline-block;
    background: #17213C;
    color: white;
    padding: 6px 13px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 800;
    margin-bottom: 8px;
}

.pill-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(120px, 1fr));
    gap: 14px;
    margin-top: 16px;
    margin-bottom: 20px;
}

.pill-item {
    background: #F7F2EF;
    border-radius: 18px;
    padding: 14px 18px;
    text-align: center;
    font-weight: 700;
    color: #17213C;
    border: 1px solid transparent;
}

.info-note {
    color: #44516A;
    font-size: 16px;
    line-height: 1.6;
}

.small-muted {
    color: #64748B;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# FIGMA-STYLE HEADER
# ============================================================

st.markdown("""
<div class="bakesmart-nav">
    <div class="bakesmart-logo">
        <span class="logo-icon">🍰</span>
        <span>BakeSmart</span>
    </div>
    <div class="nav-links">
        <span>Home</span>
        <span>Build Recipe</span>
        <span>Analytics</span>
        <span>Community</span>
        <span>About</span>
    </div>
</div>

<div class="hero-section">
    <div class="hero-title">Bake smarter with what you already have.</div>
    <div class="hero-subtitle">
        Find baking recipes based on your ingredients, allergies, nutrition goals,
        and machine-learning-based recipe similarity.
    </div>
</div>
""", unsafe_allow_html=True)

conn = sqlite3.connect(DB_FILE_NAME, check_same_thread=False)

df_spoon = load_table_if_exists(conn, SPOON_TABLE)
df_themealdb = load_table_if_exists(conn, MEALDB_TABLE)
df_allergy = load_table_if_exists(conn, ALLERGY_TABLE)

recipes_spoon = standardize_spoon(df_spoon)
recipes_themealdb = standardize_themealdb(df_themealdb)
recipes_master = pd.concat([recipes_spoon, recipes_themealdb], ignore_index=True)

st.sidebar.header("Project Data")
st.sidebar.write(f"Total recipes loaded: {len(recipes_master)}")

# ============================================================
# INPUT SECTION
# ============================================================

st.markdown("""
<div class="white-card">
    <h2>Build Your Recipe</h2>
    <p class="info-note">Select your available ingredients, allergies, and nutrition goals.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="section-card">
    <h3>Step 1: Select Common Ingredients</h3>
    <p class="info-note">Choose ingredients you already have at home.</p>
</div>
""", unsafe_allow_html=True)

common_ingredients = [
    "flour", "all purpose flour", "sugar", "brown sugar",
    "powdered sugar", "butter", "milk", "whole milk",
    "egg", "heavy cream", "cream cheese", "chocolate",
    "dark chocolate", "cocoa powder", "vanilla extract",
    "baking powder", "baking soda", "salt", "water",
    "oil", "honey", "oats", "banana", "apple",
    "strawberry", "lemon"
]

selected_common_ingredients = st.multiselect(
    "Choose ingredients you have:",
    common_ingredients
)

extra_ingredients_raw = st.text_area(
    "Add other ingredients separated by commas:",
    placeholder="matcha powder, coconut milk"
)

extra_ingredients = [
    x.strip() for x in extra_ingredients_raw.split(",") if x.strip()
]

user_ingredients = selected_common_ingredients + extra_ingredients

st.markdown("""
<div class="section-card">
    <h3>Step 2: Select Allergies or Preferences</h3>
    <p class="info-note">Choose any dietary restrictions or nutrition goals.</p>
</div>
""", unsafe_allow_html=True)

selected_allergies = st.multiselect(
    "Choose allergies:",
    ["dairy", "gluten", "egg", "peanut", "tree nuts", "soy"]
)

nutrition_goal = st.selectbox(
    "Choose one nutrition goal:",
    ["None", "low sugar", "low fat", "high protein", "low calorie"]
)

nutrition_goal = None if nutrition_goal == "None" else nutrition_goal

# ============================================================
# RECIPE RECOMMENDATION
# ============================================================

st.markdown("""
<div class="section-card">
    <h3>Step 3: Find Recipes</h3>
    <p class="info-note">Recipes will be ranked by ingredient match percentage and enhanced with allergy and ML-based recommendations.</p>
</div>
""", unsafe_allow_html=True)

if st.button("Find Recipes"):
    top_matches = find_recipes_by_ingredients(
        recipes_master,
        user_ingredients,
        top_n=5
    )

    if top_matches.empty:
        st.warning("No matching recipes were found.")
    else:
        st.session_state["top_matches"] = top_matches
        st.success("Top matching recipes found!")

if "top_matches" in st.session_state:
    top_matches = st.session_state["top_matches"]

    st.subheader("Top Matching Recipes")
    st.dataframe(
        top_matches[["recipe_name", "match_percentage", "matched_ingredients", "source_table"]],
        use_container_width=True
    )

    recipe_options = {
        f"{idx}. {row['recipe_name']}": idx
        for idx, row in top_matches.iterrows()
    }

    selected_label = st.selectbox(
        "Select a recipe to inspect:",
        list(recipe_options.keys())
    )

    if st.button("View Recipe Details"):
        selected_idx = recipe_options[selected_label]
        selected_row = top_matches.loc[selected_idx]

        selected_id = selected_row["recipe_id"]
        selected_source = selected_row["source_table"]

        selected_recipe = load_recipe_by_source_and_id(
            conn,
            selected_id,
            selected_source
        )

        if selected_recipe is None:
            selected_recipe = {
                "recipe_id": selected_row.get("recipe_id"),
                "recipe_name": selected_row.get("recipe_name"),
                "ingredients_clean": selected_row.get("ingredients_clean", []),
                "ingredients_combined": selected_row.get("ingredients_combined"),
                "image_url": selected_row.get("image_url"),
                "instructions": selected_row.get("instructions"),
                "source_url": selected_row.get("source_url"),
                "nutrients": selected_row.get("nutrients"),
                "source_table": selected_row.get("source_table")
            }

        st.subheader(selected_recipe.get("recipe_name", "N/A"))

        if selected_recipe.get("image_url"):
            st.image(selected_recipe.get("image_url"), width=300)

        st.write("Source:", selected_recipe.get("source_table", "N/A"))
        st.write("Ingredients:", selected_recipe.get("ingredients_combined", "N/A"))
        st.write("Nutrients:", selected_recipe.get("nutrients", "N/A"))
        st.write("Instructions:", selected_recipe.get("instructions", "N/A"))

        if selected_recipe.get("source_url"):
            st.markdown(f"[Recipe Source Link]({selected_recipe.get('source_url')})")

        # Allergy section
        all_allergy_hits = detect_possible_allergies(
            selected_recipe.get("ingredients_clean", []),
            df_allergy
        )

        filtered_allergy_hits = filter_allergy_hits_for_user(
            all_allergy_hits,
            selected_allergies
        )

        st.subheader("Relevant Allergy Sources Based on Your Input")

        if not filtered_allergy_hits.empty:
            st.dataframe(filtered_allergy_hits, use_container_width=True)
        else:
            st.write("No allergy-source rows matched your specific allergy input.")
                    # LLM substitution section
        st.subheader("LLM Recommended Ingredient Substitutions")

        llm_substitutions = get_llm_substitution_recommendations(
            recipe=selected_recipe,
            allergies=selected_allergies,
            nutrition_goal=nutrition_goal
        )

        if llm_substitutions:
            for item in llm_substitutions:
                with st.container(border=True):
                    st.write("Original ingredient:", item.get("original_ingredient", "N/A"))
                    st.write("LLM recommended substitute:", item.get("substitute_ingredient", "N/A"))
                    st.write("Recommended amount:", item.get("recommended_amount", "N/A"))
                    st.write("Reason:", item.get("reason", "N/A"))
        else:
            st.write("No LLM substitution recommendations were generated.")

        # ML section
        ml_result = get_ml_recipe_recommendation(
            conn,
            selected_id,
            top_n=3
        )

        st.subheader("Machine Learning Recipe Recommendation")

        if ml_result and ml_result.get("cluster_name"):
            st.write("Recipe Type:", ml_result.get("cluster_name"))
            st.write(
                "This recipe type is identified by machine learning based on ingredient similarity."
            )

            similar_recipes = ml_result.get("similar_recipes")
            if similar_recipes is not None and not similar_recipes.empty:
                st.write("Similar recipes from the same or closest recipe group:")
                st.dataframe(similar_recipes, use_container_width=True)
            else:
                st.write("No similar recipes were found.")
        else:
            st.write("Machine learning cluster information is not available for this recipe.")

# ============================================================
# USER SUBSTITUTION INPUT
# ============================================================

st.markdown("""
<div class="section-card">
    <h3>Step 4: Share Your Substitution Experience</h3>
    <p class="info-note">
        Help future users by sharing what substitute ingredient you used and how much you used.
    </p>
</div>
""", unsafe_allow_html=True)

st.write(
    "If you changed an ingredient, you can submit what substitute you used. "
    "This will be stored in the backend for future community-based substitution insights."
)

original_ingredient = st.text_input(
    "Original ingredient replaced:",
    placeholder="butter"
)

substitute_ingredient = st.text_input(
    "Substitute ingredient used:",
    placeholder="vegan butter"
)

amount_grams = st.number_input(
    "Amount used in grams:",
    min_value=0.0,
    step=1.0
)

if st.button("Submit Substitution"):
    if original_ingredient and substitute_ingredient and amount_grams > 0:
        store_user_substitution(
            conn,
            original_ingredient,
            substitute_ingredient,
            amount_grams
        )
        st.success("Your substitution has been saved to the backend database.")
    else:
        st.warning("Please fill in all substitution fields.")

# ============================================================
# COMMUNITY SUBSTITUTION DATA
# ============================================================

st.header("Community Substitution Data")

try:
    df_substitutions = pd.read_sql(
        f"SELECT * FROM {USER_SUB_TABLE}",
        conn
    )

    if not df_substitutions.empty:
        st.dataframe(df_substitutions, use_container_width=True)

        chart_data = (
            df_substitutions
            .groupby("substitute_ingredient")["amount_grams"]
            .mean()
            .sort_values(ascending=False)
        )

        st.bar_chart(chart_data)
    else:
        st.write("No user substitution data has been submitted yet.")
except:
    st.write("No user substitution data has been submitted yet.")
