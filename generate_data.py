from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


OUTPUT_PATH = Path("data/warehouse_damage_data.csv")
ROW_COUNT = 5000
RANDOM_SEED = 42


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

PROCESS_AREAS = [
    "Receiving",
    "Putaway",
    "Replenishment",
    "Picking",
    "Shipping",
    "Yard / Dock",
]

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

EQUIPMENT_TYPES = [
    "Clamp",
    "Power jack",
    "Reach truck",
    "Forklift",
    "Walkie",
    "Manual handling",
    "Truck / trailer",
]

HANDLING_STAGES = [
    "Unloading",
    "Receiving inspection",
    "Putaway",
    "Replenishment",
    "Picking",
    "Staging",
    "Loading",
    "Dock movement",
]

SHIFTS = ["Day", "Night", "Weekend"]
SLOT_LEVELS = ["Ground", "Mid", "Top"]
CONGESTION_LEVELS = ["Low", "Medium", "High"]
PALLET_STABILITY = ["Stable", "Slightly unstable", "Unstable"]
EQUIPMENT_CONDITIONS = ["Good", "Needs inspection", "Maintenance concern"]
SLOTTING_RISK = ["Low", "Medium", "High"]
DAMAGE_TYPES = [
    "Crushed",
    "Leaking",
    "Torn",
    "Broken",
    "Pallet collapse",
    "Dropped",
    "Trailer impact",
    "Stacking damage",
]


CATEGORY_PROFILES: Dict[str, Dict[str, float]] = {
    "Grocery": {"weight_low": 8, "weight_high": 35, "cube_low": 0.3, "cube_high": 1.8, "cost_low": 10, "cost_high": 32},
    "Chemical": {"weight_low": 12, "weight_high": 42, "cube_low": 0.4, "cube_high": 1.6, "cost_low": 18, "cost_high": 65},
    "Tissue": {"weight_low": 4, "weight_high": 16, "cube_low": 1.2, "cube_high": 3.5, "cost_low": 12, "cost_high": 28},
    "Pet Food": {"weight_low": 20, "weight_high": 65, "cube_low": 0.6, "cube_high": 2.4, "cost_low": 20, "cost_high": 48},
    "Beverage": {"weight_low": 18, "weight_high": 55, "cube_low": 0.5, "cube_high": 2.0, "cost_low": 16, "cost_high": 40},
    "Frozen": {"weight_low": 10, "weight_high": 38, "cube_low": 0.4, "cube_high": 1.7, "cost_low": 18, "cost_high": 52},
    "Produce": {"weight_low": 8, "weight_high": 28, "cube_low": 0.5, "cube_high": 2.2, "cost_low": 14, "cost_high": 36},
    "Dairy": {"weight_low": 10, "weight_high": 30, "cube_low": 0.5, "cube_high": 1.6, "cost_low": 14, "cost_high": 34},
    "Meat": {"weight_low": 12, "weight_high": 40, "cube_low": 0.4, "cube_high": 1.5, "cost_low": 24, "cost_high": 72},
    "Snacks": {"weight_low": 5, "weight_high": 18, "cube_low": 0.8, "cube_high": 2.8, "cost_low": 10, "cost_high": 26},
}


AREA_SOURCE_WEIGHTS: Dict[str, Dict[str, int]] = {
    "Receiving": {
        "Receiver handling error": 24,
        "Already damaged on arrival": 28,
        "Pallet collapse": 16,
        "Freight fell over": 18,
        "Unknown": 8,
        "Putaway impact": 3,
        "Driver backing up": 3,
    },
    "Putaway": {
        "Putaway impact": 36,
        "Product stacked incorrectly": 18,
        "Pallet collapse": 14,
        "Freight fell over": 10,
        "Unknown": 8,
        "Receiver handling error": 6,
        "Replenishment impact": 8,
    },
    "Replenishment": {
        "Replenishment impact": 34,
        "Product stacked incorrectly": 20,
        "Pallet collapse": 10,
        "Unknown": 16,
        "Putaway impact": 10,
        "Picker handling": 10,
    },
    "Picking": {
        "Picker handling": 38,
        "Product stacked incorrectly": 14,
        "Pallet collapse": 10,
        "Freight fell over": 8,
        "Unknown": 12,
        "Replenishment impact": 8,
        "Putaway impact": 10,
    },
    "Shipping": {
        "Shipper loading issue": 34,
        "Pallet collapse": 18,
        "Product stacked incorrectly": 18,
        "Freight fell over": 12,
        "Unknown": 10,
        "Driver backing up": 8,
    },
    "Yard / Dock": {
        "Driver backing up": 30,
        "Freight fell over": 20,
        "Pallet collapse": 14,
        "Already damaged on arrival": 8,
        "Shipper loading issue": 10,
        "Unknown": 10,
        "Product stacked incorrectly": 8,
    },
}


AREA_STAGE_WEIGHTS: Dict[str, Dict[str, int]] = {
    "Receiving": {"Unloading": 40, "Receiving inspection": 40, "Dock movement": 20},
    "Putaway": {"Putaway": 70, "Receiving inspection": 10, "Dock movement": 20},
    "Replenishment": {"Replenishment": 70, "Picking": 10, "Staging": 20},
    "Picking": {"Picking": 75, "Staging": 15, "Replenishment": 10},
    "Shipping": {"Loading": 50, "Staging": 30, "Dock movement": 20},
    "Yard / Dock": {"Dock movement": 50, "Loading": 25, "Unloading": 25},
}


AREA_EQUIPMENT_WEIGHTS: Dict[str, Dict[str, int]] = {
    "Receiving": {"Forklift": 24, "Power jack": 20, "Manual handling": 20, "Walkie": 16, "Truck / trailer": 20},
    "Putaway": {"Reach truck": 34, "Forklift": 30, "Walkie": 18, "Clamp": 8, "Power jack": 10},
    "Replenishment": {"Reach truck": 24, "Walkie": 24, "Power jack": 22, "Forklift": 18, "Manual handling": 12},
    "Picking": {"Manual handling": 34, "Walkie": 26, "Power jack": 18, "Reach truck": 12, "Forklift": 10},
    "Shipping": {"Forklift": 28, "Power jack": 22, "Truck / trailer": 18, "Manual handling": 18, "Walkie": 14},
    "Yard / Dock": {"Truck / trailer": 42, "Forklift": 18, "Power jack": 14, "Manual handling": 10, "Walkie": 16},
}


AISLE_RISK_MULTIPLIERS = {
    "A01": 0.95,
    "A02": 1.00,
    "A03": 1.05,
    "A04": 1.18,
    "A05": 1.08,
    "A06": 1.00,
    "A07": 1.22,
    "A08": 1.12,
    "A09": 0.96,
    "A10": 1.15,
    "A11": 1.03,
    "A12": 1.20,
}


def weighted_choice(options: Dict[str, int], rng: np.random.Generator) -> str:
    names = list(options.keys())
    weights = np.array(list(options.values()), dtype=float)
    probabilities = weights / weights.sum()
    return str(rng.choice(names, p=probabilities))


def build_sku_catalog(rng: np.random.Generator) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for category in CATEGORIES:
        for index in range(1, 11):
            sku = f"{category[:3].upper()}-{index:03d}"
            records.append(
                {
                    "sku": sku,
                    "description": f"{category} Item {index}",
                    "category": category,
                }
            )
    catalog = pd.DataFrame(records)
    catalog = catalog.sample(frac=1, random_state=int(rng.integers(1, 9999))).reset_index(drop=True)
    return catalog


def generate_record(rng: np.random.Generator, sku_catalog: pd.DataFrame) -> Dict[str, object]:
    item = sku_catalog.sample(n=1, random_state=int(rng.integers(1, 1_000_000))).iloc[0]
    category = str(item["category"])
    profile = CATEGORY_PROFILES[category]

    process_area = str(rng.choice(PROCESS_AREAS, p=[0.20, 0.18, 0.12, 0.24, 0.16, 0.10]))
    damage_source = weighted_choice(AREA_SOURCE_WEIGHTS[process_area], rng)
    handling_stage = weighted_choice(AREA_STAGE_WEIGHTS[process_area], rng)
    equipment_type = weighted_choice(AREA_EQUIPMENT_WEIGHTS[process_area], rng)

    aisle = str(rng.choice(list(AISLE_RISK_MULTIPLIERS.keys())))
    shift = str(rng.choice(SHIFTS, p=[0.55, 0.30, 0.15]))

    if process_area in {"Picking", "Putaway"}:
        slot_level = str(rng.choice(SLOT_LEVELS, p=[0.25, 0.40, 0.35]))
    else:
        slot_level = str(rng.choice(SLOT_LEVELS, p=[0.45, 0.35, 0.20]))

    if aisle in {"A04", "A07", "A12"} or process_area in {"Picking", "Shipping"}:
        congestion_level = str(rng.choice(CONGESTION_LEVELS, p=[0.20, 0.45, 0.35]))
    else:
        congestion_level = str(rng.choice(CONGESTION_LEVELS, p=[0.35, 0.45, 0.20]))

    if process_area in {"Receiving", "Shipping", "Yard / Dock"}:
        pallet_stability = str(rng.choice(PALLET_STABILITY, p=[0.48, 0.32, 0.20]))
    else:
        pallet_stability = str(rng.choice(PALLET_STABILITY, p=[0.60, 0.28, 0.12]))

    if equipment_type in {"Forklift", "Reach truck", "Truck / trailer"}:
        equipment_condition = str(rng.choice(EQUIPMENT_CONDITIONS, p=[0.68, 0.22, 0.10]))
    else:
        equipment_condition = str(rng.choice(EQUIPMENT_CONDITIONS, p=[0.76, 0.18, 0.06]))

    if slot_level == "Top" or process_area in {"Putaway", "Picking"}:
        slotting_risk = str(rng.choice(SLOTTING_RISK, p=[0.30, 0.42, 0.28]))
    else:
        slotting_risk = str(rng.choice(SLOTTING_RISK, p=[0.46, 0.38, 0.16]))

    cases_handled = int(rng.integers(12, 180))
    weight_lbs = round(float(rng.uniform(profile["weight_low"], profile["weight_high"])), 1)
    cube = round(float(rng.uniform(profile["cube_low"], profile["cube_high"])), 2)
    cost_per_case = round(float(rng.uniform(profile["cost_low"], profile["cost_high"])), 2)

    if category == "Pet Food":
        weight_lbs = round(weight_lbs + float(rng.uniform(5, 10)), 1)
    if category == "Beverage":
        weight_lbs = round(weight_lbs + float(rng.uniform(4, 8)), 1)

    damage_type_weights = {
        "Crushed": 14,
        "Leaking": 10,
        "Torn": 12,
        "Broken": 14,
        "Pallet collapse": 10,
        "Dropped": 16,
        "Trailer impact": 10,
        "Stacking damage": 14,
    }
    if category == "Chemical":
        damage_type_weights["Leaking"] += 18
    if category == "Tissue":
        damage_type_weights["Crushed"] += 18
    if process_area == "Yard / Dock":
        damage_type_weights["Trailer impact"] += 14
    if damage_source == "Pallet collapse":
        damage_type_weights["Pallet collapse"] += 18
    if damage_source == "Product stacked incorrectly":
        damage_type_weights["Stacking damage"] += 18

    damage_type = weighted_choice(damage_type_weights, rng)

    risk_score = 0.012
    risk_score += {
        "Receiving": 0.030,
        "Putaway": 0.024,
        "Replenishment": 0.012,
        "Picking": 0.034,
        "Shipping": 0.030,
        "Yard / Dock": 0.028,
    }[process_area]
    risk_score += {"Low": 0.0, "Medium": 0.010, "High": 0.025}[congestion_level]
    risk_score += {"Stable": 0.0, "Slightly unstable": 0.014, "Unstable": 0.032}[pallet_stability]
    risk_score += {"Good": 0.0, "Needs inspection": 0.015, "Maintenance concern": 0.030}[equipment_condition]
    risk_score += {"Low": 0.0, "Medium": 0.010, "High": 0.024}[slotting_risk]
    risk_score += {"Ground": 0.0, "Mid": 0.005, "Top": 0.020}[slot_level]
    risk_score += (AISLE_RISK_MULTIPLIERS[aisle] - 1.0) * 0.08
    risk_score += max(weight_lbs - 30, 0) / 1000
    risk_score += max(cube - 1.8, 0) * 0.01

    if damage_source == "Picker handling":
        risk_score += 0.015
    if damage_source == "Driver backing up":
        risk_score += 0.018
    if damage_source == "Freight fell over":
        risk_score += 0.020
    if damage_source == "Already damaged on arrival":
        risk_score += 0.024
    if category in {"Chemical", "Beverage", "Pet Food"}:
        risk_score += 0.006
    if process_area == "Replenishment":
        risk_score -= 0.008

    damage_rate = float(np.clip(rng.normal(loc=risk_score, scale=0.012), 0.003, 0.22))
    damaged_cases = max(1, int(round(cases_handled * damage_rate)))
    damaged_cases = min(damaged_cases, cases_handled)
    damage_rate = damaged_cases / cases_handled
    damage_cost = round(damaged_cases * cost_per_case, 2)
    high_damage = int(damage_rate >= 0.05)

    date_value = pd.Timestamp("2025-01-01") + pd.to_timedelta(int(rng.integers(0, 365)), unit="D")

    return {
        "date": date_value.date().isoformat(),
        "sku": item["sku"],
        "description": item["description"],
        "category": category,
        "aisle": aisle,
        "slot_level": slot_level,
        "shift": shift,
        "process_area": process_area,
        "damage_source": damage_source,
        "equipment_type": equipment_type,
        "handling_stage": handling_stage,
        "congestion_level": congestion_level,
        "pallet_stability": pallet_stability,
        "equipment_condition": equipment_condition,
        "slotting_risk": slotting_risk,
        "cases_handled": cases_handled,
        "damaged_cases": damaged_cases,
        "damage_type": damage_type,
        "weight_lbs": round(weight_lbs, 1),
        "cube": cube,
        "cost_per_case": cost_per_case,
        "damage_rate": round(damage_rate, 4),
        "damage_cost": damage_cost,
        "high_damage": high_damage,
    }


def generate_dataset(row_count: int = ROW_COUNT, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sku_catalog = build_sku_catalog(rng)
    records = [generate_record(rng, sku_catalog) for _ in range(row_count)]
    data = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    return data


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset = generate_dataset()
    dataset.to_csv(OUTPUT_PATH, index=False)

    print(f"Dataset shape: {dataset.shape}")
    print("\nFirst 5 rows:")
    print(dataset.head())
    print(f"\nDataset saved successfully to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
