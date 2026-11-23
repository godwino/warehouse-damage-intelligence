from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path("data/warehouse_damage_data.csv")

CATEGORIES = [
    "Grocery",
    "Chemical",
    "Tissue",
    "Pet Food",
    "Beverage",
    "Frozen",
    "Produce",
    "Dairy",
    "Meat",
    "Snacks",
]
AISLES = [f"A{i:02d}" for i in range(1, 13)]
SLOT_LEVELS = ["Ground", "Mid", "Top"]
SHIFTS = ["Day", "Night", "Weekend"]
PROCESS_AREAS = ["Receiving", "Putaway", "Replenishment", "Picking", "Shipping", "Yard / Dock"]
DAMAGE_SOURCES = [
    "Driver backing up",
    "Freight fell over",
    "Receiver handling error",
    "Putaway impact",
    "Replenishment impact",
    "Picker handling",
    "Shipper loading issue",
    "Pallet collapse",
    "Product stacked incorrectly",
    "Already damaged on arrival",
    "Unknown",
]
EQUIPMENT_TYPES = ["Clamp", "Power jack", "Reach truck", "Forklift", "Walkie", "Manual handling", "Truck / trailer"]
HANDLING_STAGES = ["Unloading", "Receiving inspection", "Putaway", "Replenishment", "Picking", "Staging", "Loading", "Dock movement"]
CONGESTION_LEVELS = ["Low", "Medium", "High"]
PALLET_STABILITY = ["Stable", "Slightly unstable", "Unstable"]
EQUIPMENT_CONDITIONS = ["Good", "Needs inspection", "Maintenance concern"]
SLOTTING_RISK = ["Low", "Medium", "High"]

FEATURE_COLUMNS = [
    "category",
    "aisle",
    "slot_level",
    "shift",
    "process_area",
    "equipment_type",
    "handling_stage",
    "congestion_level",
    "pallet_stability",
    "equipment_condition",
    "slotting_risk",
    "cases_handled",
    "weight_lbs",
    "cube",
    "cost_per_case",
]


st.set_page_config(page_title="Warehouse Damage Intelligence Dashboard", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH, parse_dates=["date"])
    data["damage_rate_pct"] = data["damage_rate"] * 100
    return data


@st.cache_resource
def train_model(data: pd.DataFrame) -> dict:
    X = data[FEATURE_COLUMNS]
    y = data["high_damage"]

    categorical_features = [
        "category",
        "aisle",
        "slot_level",
        "shift",
        "process_area",
        "equipment_type",
        "handling_stage",
        "congestion_level",
        "pallet_stability",
        "equipment_condition",
        "slotting_risk",
    ]
    numeric_features = ["cases_handled", "weight_lbs", "cube", "cost_per_case"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    importances = model.named_steps["classifier"].feature_importances_
    feature_importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(15)
    )

    return {
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
        "feature_importance": feature_importance_df,
    }


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def format_percent(value: float) -> str:
    return f"{value:.2f}%"


def recommendation_text(row: pd.Series | dict) -> str:
    actions: List[str] = []
    damage_source = row.get("damage_source", "Unknown")
    category = row["category"]
    process_area = row["process_area"]
    equipment_type = row["equipment_type"]
    congestion_level = row["congestion_level"]
    pallet_stability = row["pallet_stability"]
    equipment_condition = row["equipment_condition"]
    slotting_risk = row["slotting_risk"]
    slot_level = row.get("slot_level", "Mid")
    weight_lbs = float(row.get("weight_lbs", 0))

    source_actions = {
        "Driver backing up": "Dock safety review, backing procedure audit, and spotter communication check.",
        "Freight fell over": "Pallet stability review, load securement check, and wrap quality audit.",
        "Receiver handling error": "Receiving SOP refresher and unloading process check.",
        "Already damaged on arrival": "Vendor and carrier claim review with stronger inbound inspection documentation.",
        "Putaway impact": "Racking clearance review, slot fit check, and MHE handling review.",
        "Replenishment impact": "Replenishment path review and equipment handling coaching.",
        "Picker handling": "Pick-face review, handling standard review, and slotting check.",
        "Shipper loading issue": "Loading SOP review and pallet build quality audit.",
        "Pallet collapse": "Stacking pattern review and wrap quality audit.",
        "Product stacked incorrectly": "Pallet build standard review and heavy-bottom, light-top validation.",
    }
    if damage_source in source_actions:
        actions.append(source_actions[damage_source])
    if category == "Chemical":
        actions.append("Review packaging, segregation, and spill-control readiness.")
    if weight_lbs >= 40 and slot_level == "Top":
        actions.append("Move this heavier item to a lower slot to reduce drop and reach risk.")
    if process_area == "Yard / Dock" and equipment_type == "Truck / trailer":
        actions.append("Tighten dock movement controls and trailer communication practices.")
    if congestion_level == "High":
        actions.append("Reduce congestion with aisle flow adjustments and traffic control improvements.")
    if pallet_stability == "Unstable":
        actions.append("Review pallet stability, wrapping, and stacking pattern controls.")
    if equipment_condition in {"Needs inspection", "Maintenance concern"}:
        actions.append("Schedule equipment inspection and maintenance follow-up.")
    if slotting_risk == "High":
        actions.append("Complete a slotting review and product-location fit assessment.")
    if not actions:
        actions.append("Monitor this pattern and validate local handling, slotting, and equipment conditions.")
    return " ".join(actions)


def build_recommendations(data: pd.DataFrame) -> pd.DataFrame:
    recommendation_columns = [
        "process_area",
        "sku",
        "category",
        "aisle",
        "damage_source",
        "equipment_type",
        "congestion_level",
        "pallet_stability",
        "equipment_condition",
        "slotting_risk",
    ]
    grouped = (
        data.groupby(recommendation_columns, as_index=False)
        .agg(
            damage_rate=("damage_rate", "mean"),
            damage_cost=("damage_cost", "sum"),
            weight_lbs=("weight_lbs", "mean"),
            slot_level=("slot_level", lambda values: values.mode().iloc[0] if not values.mode().empty else "Mid"),
        )
        .sort_values(["damage_cost", "damage_rate"], ascending=[False, False])
    )

    high_cost_threshold = grouped["damage_cost"].quantile(0.75)
    high_rate_threshold = grouped["damage_rate"].quantile(0.75)

    def assign_priority(row: pd.Series) -> str:
        high_cost = row["damage_cost"] >= high_cost_threshold
        high_rate = row["damage_rate"] >= high_rate_threshold
        if high_cost and high_rate:
            return "High"
        if high_cost or high_rate:
            return "Medium"
        return "Low"

    grouped["recommended_action"] = grouped.apply(recommendation_text, axis=1)
    grouped["priority"] = grouped.apply(assign_priority, axis=1)
    grouped["damage_rate"] = grouped["damage_rate"].round(4)
    grouped["damage_cost"] = grouped["damage_cost"].round(2)
    display_columns = recommendation_columns + ["damage_rate", "damage_cost", "recommended_action", "priority"]
    return grouped[display_columns].head(20)


def metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div style="padding:1rem;border:1px solid #d9e2ec;border-radius:14px;background:#f8fbff;">
            <div style="font-size:0.9rem;color:#486581;">{label}</div>
            <div style="font-size:1.6rem;font-weight:700;color:#102a43;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_executive_summary(data: pd.DataFrame) -> None:
    st.header("1. Executive Summary")
    top_area = data.groupby("process_area")["damage_cost"].sum().idxmax()
    top_category = data.groupby("category")["damage_cost"].sum().idxmax()
    top_source = data.groupby("damage_source")["damage_cost"].sum().idxmax()

    total_cases = int(data["cases_handled"].sum())
    total_damaged = int(data["damaged_cases"].sum())
    overall_rate = (total_damaged / total_cases) * 100
    total_damage_cost = float(data["damage_cost"].sum())
    high_damage_count = int(data["high_damage"].sum())

    row1 = st.columns(4)
    row2 = st.columns(4)

    with row1[0]:
        metric_card("Total Cases Handled", f"{total_cases:,.0f}")
    with row1[1]:
        metric_card("Total Damaged Cases", f"{total_damaged:,.0f}")
    with row1[2]:
        metric_card("Overall Damage Rate", format_percent(overall_rate))
    with row1[3]:
        metric_card("Total Damage Cost", format_currency(total_damage_cost))
    with row2[0]:
        metric_card("High-Damage Records", f"{high_damage_count:,}")
    with row2[1]:
        metric_card("Highest Damage Process Area", top_area)
    with row2[2]:
        metric_card("Highest Damage Category", top_category)
    with row2[3]:
        metric_card("Highest Damage Source", top_source)


def show_damage_hotspots(data: pd.DataFrame) -> None:
    st.header("2. Damage Hotspots")
    col1, col2 = st.columns(2)

    top_skus = data.groupby("sku", as_index=False)["damage_cost"].sum().nlargest(10, "damage_cost")
    fig_skus = px.bar(top_skus, x="damage_cost", y="sku", orientation="h", title="Top 10 SKUs by Damage Cost")
    fig_skus.update_layout(yaxis={"categoryorder": "total ascending"})

    by_aisle_cost = data.groupby("aisle", as_index=False)["damage_cost"].sum()
    fig_aisle_cost = px.bar(by_aisle_cost, x="aisle", y="damage_cost", title="Damage Cost by Aisle")

    with col1:
        st.plotly_chart(fig_skus, use_container_width=True)
        st.plotly_chart(fig_aisle_cost, use_container_width=True)

    by_aisle_rate = (
        data.groupby("aisle", as_index=False)
        .agg(damaged_cases=("damaged_cases", "sum"), cases_handled=("cases_handled", "sum"))
        .assign(damage_rate_pct=lambda df: df["damaged_cases"] / df["cases_handled"] * 100)
    )
    fig_aisle_rate = px.line(by_aisle_rate, x="aisle", y="damage_rate_pct", markers=True, title="Damage Rate by Aisle")

    daily_trend = data.groupby("date", as_index=False)[["damage_cost", "damaged_cases"]].sum()
    fig_trend = px.line(daily_trend, x="date", y="damage_cost", title="Damage Trend Over Time")

    with col2:
        st.plotly_chart(fig_aisle_rate, use_container_width=True)
        st.plotly_chart(fig_trend, use_container_width=True)

    damage_type_breakdown = data.groupby("damage_type", as_index=False)["damaged_cases"].sum()
    fig_damage_type = px.pie(damage_type_breakdown, names="damage_type", values="damaged_cases", title="Damage Type Breakdown")
    st.plotly_chart(fig_damage_type, use_container_width=True)


def show_process_area_analysis(data: pd.DataFrame) -> None:
    st.header("3. Process Area Analysis")
    col1, col2 = st.columns(2)

    area_summary = (
        data.groupby("process_area", as_index=False)
        .agg(
            damage_cost=("damage_cost", "sum"),
            damaged_cases=("damaged_cases", "sum"),
            cases_handled=("cases_handled", "sum"),
        )
        .assign(damage_rate_pct=lambda df: df["damaged_cases"] / df["cases_handled"] * 100)
    )

    with col1:
        st.plotly_chart(px.bar(area_summary, x="process_area", y="damage_cost", title="Damage Cost by Process Area"), use_container_width=True)
        st.plotly_chart(px.bar(area_summary, x="process_area", y="damaged_cases", title="Damaged Cases by Process Area"), use_container_width=True)
    with col2:
        st.plotly_chart(px.bar(area_summary, x="process_area", y="damage_rate_pct", title="Damage Rate by Process Area"), use_container_width=True)
        trend = data.groupby(["date", "process_area"], as_index=False)["damage_cost"].sum()
        st.plotly_chart(px.line(trend, x="date", y="damage_cost", color="process_area", title="Damage Trend by Process Area"), use_container_width=True)

    highest_area = area_summary.sort_values("damage_cost", ascending=False).iloc[0]
    lowest_area = area_summary.sort_values("damage_rate_pct", ascending=True).iloc[0]
    st.info(
        f"{highest_area['process_area']} is the largest cost contributor, while {lowest_area['process_area']} has the lowest damage rate. "
        "This comparison helps leaders focus improvement work where process conditions create the biggest financial impact."
    )


def show_sku_category_analysis(data: pd.DataFrame) -> None:
    st.header("4. SKU and Category Analysis")
    col1, col2 = st.columns(2)

    category_summary = (
        data.groupby("category", as_index=False)
        .agg(damage_cost=("damage_cost", "sum"), damaged_cases=("damaged_cases", "sum"), cases_handled=("cases_handled", "sum"))
        .assign(damage_rate_pct=lambda df: df["damaged_cases"] / df["cases_handled"] * 100)
    )
    high_risk_skus = (
        data.groupby(["sku", "description"], as_index=False)
        .agg(damage_rate=("damage_rate", "mean"), damage_cost=("damage_cost", "sum"))
        .sort_values(["damage_rate", "damage_cost"], ascending=[False, False])
        .head(10)
    )
    slot_rate = (
        data.groupby("slot_level", as_index=False)
        .agg(damaged_cases=("damaged_cases", "sum"), cases_handled=("cases_handled", "sum"))
        .assign(damage_rate_pct=lambda df: df["damaged_cases"] / df["cases_handled"] * 100)
    )

    with col1:
        st.plotly_chart(px.bar(category_summary, x="category", y="damage_rate_pct", title="Damage Rate by Category"), use_container_width=True)
        st.plotly_chart(px.bar(category_summary, x="category", y="damage_cost", title="Damage Cost by Category"), use_container_width=True)
    with col2:
        st.plotly_chart(px.bar(high_risk_skus, x="damage_rate", y="sku", orientation="h", title="Top 10 High-Risk SKUs"), use_container_width=True)
        st.plotly_chart(px.bar(slot_rate, x="slot_level", y="damage_rate_pct", title="Slot Level vs Damage Rate"), use_container_width=True)

    heatmap_data = (
        data.groupby(["category", "slot_level"], as_index=False)
        .agg(damaged_cases=("damaged_cases", "sum"), cases_handled=("cases_handled", "sum"))
        .assign(damage_rate_pct=lambda df: df["damaged_cases"] / df["cases_handled"] * 100)
    )
    heatmap_matrix = heatmap_data.pivot(index="category", columns="slot_level", values="damage_rate_pct").fillna(0)
    heatmap = go.Figure(
        data=go.Heatmap(
            z=heatmap_matrix.values,
            x=heatmap_matrix.columns,
            y=heatmap_matrix.index,
            colorscale="YlOrRd",
        )
    )
    heatmap.update_layout(title="Category and Slot Level Heatmap")
    st.plotly_chart(heatmap, use_container_width=True)


def show_equipment_source_analysis(data: pd.DataFrame) -> None:
    st.header("5. Equipment and Damage Source Analysis")
    col1, col2 = st.columns(2)

    equipment_cost = data.groupby("equipment_type", as_index=False)["damage_cost"].sum()
    source_count = data.groupby("damage_source", as_index=False).size().rename(columns={"size": "damage_count"})
    source_area = data.groupby(["process_area", "damage_source"], as_index=False)["damage_cost"].sum()
    equipment_area = data.groupby(["process_area", "equipment_type"], as_index=False)["damage_cost"].sum()

    with col1:
        st.plotly_chart(px.bar(equipment_cost, x="equipment_type", y="damage_cost", title="Damage Cost by Equipment Type"), use_container_width=True)
        st.plotly_chart(px.bar(source_count, x="damage_source", y="damage_count", title="Damage Count by Damage Source"), use_container_width=True)
    with col2:
        st.plotly_chart(px.bar(source_area, x="process_area", y="damage_cost", color="damage_source", title="Damage Source by Process Area"), use_container_width=True)
        st.plotly_chart(px.bar(equipment_area, x="process_area", y="damage_cost", color="equipment_type", title="Equipment Type by Process Area"), use_container_width=True)


def show_operational_risk_analysis(data: pd.DataFrame) -> None:
    st.header("6. Operational Risk Analysis")
    st.caption("This section emphasizes process conditions and operating environment rather than individual blame.")

    col1, col2 = st.columns(2)

    def risk_summary(field: str) -> pd.DataFrame:
        return (
            data.groupby(field, as_index=False)
            .agg(damaged_cases=("damaged_cases", "sum"), cases_handled=("cases_handled", "sum"), damage_cost=("damage_cost", "sum"))
            .assign(damage_rate_pct=lambda df: df["damaged_cases"] / df["cases_handled"] * 100)
        )

    congestion = risk_summary("congestion_level")
    pallet = risk_summary("pallet_stability")
    equipment = risk_summary("equipment_condition")
    slotting = risk_summary("slotting_risk")

    with col1:
        st.plotly_chart(px.bar(congestion, x="congestion_level", y="damage_rate_pct", title="Damage Rate by Congestion Level"), use_container_width=True)
        st.plotly_chart(px.bar(pallet, x="pallet_stability", y="damage_rate_pct", title="Damage Rate by Pallet Stability"), use_container_width=True)
        st.plotly_chart(px.bar(equipment, x="equipment_condition", y="damage_rate_pct", title="Damage Rate by Equipment Condition"), use_container_width=True)
    with col2:
        st.plotly_chart(px.bar(slotting, x="slotting_risk", y="damage_rate_pct", title="Damage Rate by Slotting Risk"), use_container_width=True)
        st.plotly_chart(px.bar(congestion, x="congestion_level", y="damage_cost", title="Damage Cost by Congestion Level"), use_container_width=True)
        st.plotly_chart(px.bar(pallet, x="pallet_stability", y="damage_cost", title="Damage Cost by Pallet Stability"), use_container_width=True)


def show_model_section(model_artifacts: dict) -> None:
    st.header("7. Machine Learning Model")
    metrics_cols = st.columns(4)
    metrics_cols[0].metric("Accuracy", f"{model_artifacts['accuracy']:.3f}")
    metrics_cols[1].metric("Precision", f"{model_artifacts['precision']:.3f}")
    metrics_cols[2].metric("Recall", f"{model_artifacts['recall']:.3f}")
    metrics_cols[3].metric("F1 Score", f"{model_artifacts['f1']:.3f}")

    cm = model_artifacts["confusion_matrix"]
    cm_df = pd.DataFrame(cm, index=["Actual Low", "Actual High"], columns=["Predicted Low", "Predicted High"])
    cm_fig = go.Figure(
        data=go.Heatmap(
            z=cm_df.values,
            x=cm_df.columns,
            y=cm_df.index,
            colorscale="Blues",
            text=cm_df.values,
            texttemplate="%{text}",
        )
    )
    cm_fig.update_layout(title="Confusion Matrix")
    st.plotly_chart(cm_fig, use_container_width=True)

    st.subheader("Classification Report")
    st.text(model_artifacts["classification_report"])

    importance_fig = px.bar(
        model_artifacts["feature_importance"].sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        title="Top 15 Feature Importances",
    )
    st.plotly_chart(importance_fig, use_container_width=True)


def show_prediction_tool(model: Pipeline) -> None:
    st.header("8. Prediction Tool")
    st.caption(
        "Synthetic-data demonstration. Enter conditions known before handling; use planned cases handled. "
        "Recorded damage source is used only for historical analysis and corrective actions."
    )
    with st.form("prediction_form"):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            category = st.selectbox("Category", CATEGORIES)
            aisle = st.selectbox("Aisle", AISLES)
            slot_level = st.selectbox("Slot Level", SLOT_LEVELS)
            shift = st.selectbox("Shift", SHIFTS)
        with col2:
            process_area = st.selectbox("Process Area", PROCESS_AREAS)
            equipment_type = st.selectbox("Equipment Type", EQUIPMENT_TYPES)
            handling_stage = st.selectbox("Handling Stage", HANDLING_STAGES)
        with col3:
            congestion_level = st.selectbox("Congestion Level", CONGESTION_LEVELS)
            pallet_stability = st.selectbox("Pallet Stability", PALLET_STABILITY)
            equipment_condition = st.selectbox("Equipment Condition", EQUIPMENT_CONDITIONS)
            slotting_risk = st.selectbox("Slotting Risk", SLOTTING_RISK)
        with col4:
            cases_handled = st.number_input("Cases Handled", min_value=1, value=60, step=1)
            weight_lbs = st.number_input("Weight (lbs)", min_value=1.0, value=30.0, step=1.0)
            cube = st.number_input("Cube", min_value=0.1, value=1.2, step=0.1)
            cost_per_case = st.number_input("Cost per Case", min_value=1.0, value=24.0, step=1.0)

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame(
            [
                {
                    "category": category,
                    "aisle": aisle,
                    "slot_level": slot_level,
                    "shift": shift,
                    "process_area": process_area,
                    "equipment_type": equipment_type,
                    "handling_stage": handling_stage,
                    "congestion_level": congestion_level,
                    "pallet_stability": pallet_stability,
                    "equipment_condition": equipment_condition,
                    "slotting_risk": slotting_risk,
                    "cases_handled": cases_handled,
                    "weight_lbs": weight_lbs,
                    "cube": cube,
                    "cost_per_case": cost_per_case,
                }
            ]
        )
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])
        recommendation = recommendation_text(
            {
                "category": category,
                "slot_level": slot_level,
                "process_area": process_area,
                "equipment_type": equipment_type,
                "congestion_level": congestion_level,
                "pallet_stability": pallet_stability,
                "equipment_condition": equipment_condition,
                "slotting_risk": slotting_risk,
                "weight_lbs": weight_lbs,
            }
        )

        risk_label = "High Damage Risk" if prediction == 1 else "Low Damage Risk"
        st.success(f"Prediction: {risk_label}")
        st.write(f"Predicted probability of high damage: {probability:.1%}")
        st.write(f"Recommendation: {recommendation}")


def show_recommendations(data: pd.DataFrame) -> None:
    st.header("9. Recommendations")
    recommendations = build_recommendations(data)
    st.dataframe(recommendations, use_container_width=True, hide_index=True)


def show_business_impact(data: pd.DataFrame) -> None:
    st.header("10. Business Impact Summary")
    current_cost = float(data["damage_cost"].sum())
    scenarios = pd.DataFrame(
        {
            "Scenario": ["Current Damage Cost", "10% Reduction", "15% Reduction", "20% Reduction"],
            "Value": [
                current_cost,
                current_cost * 0.10,
                current_cost * 0.15,
                current_cost * 0.20,
            ],
        }
    )
    scenarios["Display"] = scenarios["Value"].map(format_currency)

    cols = st.columns(4)
    for idx, (_, row) in enumerate(scenarios.iterrows()):
        cols[idx].metric(row["Scenario"], row["Display"])

    st.write(
        "Reducing damage improves product availability, reduces claims, lowers rework, improves safety, and supports stronger warehouse productivity."
    )


def main() -> None:
    st.title("Warehouse Damage Intelligence Dashboard")
    st.write(
        "A supply chain analytics and machine learning tool for identifying warehouse damage hotspots, process risk, and prevention actions."
    )

    st.page_link("pages/1_Picking_Order_Review.py", label="Open Picking Order Review")

    if not DATA_PATH.exists():
        st.error("Dataset not found. Run `python generate_data.py` to create `data/warehouse_damage_data.csv`.")
        st.stop()

    data = load_data()
    model_artifacts = train_model(data)

    show_executive_summary(data)
    show_damage_hotspots(data)
    show_process_area_analysis(data)
    show_sku_category_analysis(data)
    show_equipment_source_analysis(data)
    show_operational_risk_analysis(data)
    show_model_section(model_artifacts)
    show_prediction_tool(model_artifacts["model"])
    show_recommendations(data)
    show_business_impact(data)


if __name__ == "__main__":
    main()
