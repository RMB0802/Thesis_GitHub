import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from math import radians, sin, cos, sqrt, atan2
import gurobipy as gp
from gurobipy import GRB, quicksum

# ---------- GUROBI WLS LICENSE ----------
options = {
    "WLSACCESSID": "bbc701aa-34c0-43bf-84f5-f75b92604e68",
    "WLSSECRET": "0b07466a-8c90-4365-91b5-54eb2788c6de",
    "LICENSEID": 2647048,
}

# ---------- SETUP ----------
st.set_page_config(layout="wide")
st.title("Indo-PACOM TLAMM Optimization Dashboard")

# Predefined RF bases with red pins
rf_bases = [
    {"name": "Base 61 - Huangshan", "Latitude": 29.6956, "Longitude": 118.2997},
    {"name": "Base 62 - Kunming", "Latitude": 24.9888, "Longitude": 102.8346},
    {"name": "Base 63 - Huaihua", "Latitude": 27.5747, "Longitude": 110.0250},
    {"name": "Base 64 - Lanzhou", "Latitude": 35.9387, "Longitude": 104.0159},
    {"name": "Base 65 - Shenyang", "Latitude": 41.8586, "Longitude": 123.4514},
    {"name": "Base 66 - Luoyang", "Latitude": 34.6405, "Longitude": 112.3823}
]
rf_df = pd.DataFrame(rf_bases)

# Utility Functions
def great_circle(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * atan2(sqrt(a), sqrt(1-a)) * R

def calculate_wez(row):
    max_wez = 0
    for base in rf_bases:
        distance_to_base = great_circle(row['Latitude'], row['Longitude'], base['Latitude'], base['Longitude'])
        if distance_to_base <= 2500:
            wez_value = 1
        elif distance_to_base <= 4000:
            wez_value = 0.5
        else:
            wez_value = 0
        max_wez = max(max_wez, wez_value)
    return max_wez

def threat_level_label(wez):
    return "High" if wez == 1 else "Moderate" if wez == 0.5 else "Low"

# ---------- STEP 1: Customer Locations ----------
st.subheader("Step 1: Select Supported Unit Locations")
customer_data = st.file_uploader("Upload Customer Locations CSV", type="csv")
if customer_data:
    customers_df = pd.read_csv(customer_data)
    customers_df.columns = customers_df.columns.str.strip().str.lower()
    customers_df.rename(columns={"latitude": "Latitude", "longitude": "Longitude", "name": "name"}, inplace=True)
    st.write("Customer Locations:", customers_df)
else:
    st.stop()

# ---------- STEP 2: Warehouse Locations ----------
st.subheader("Step 2: Select Potential TLAMM Locations")
warehouse_data = st.file_uploader("Upload Warehouse Locations CSV", type="csv")
if warehouse_data:
    warehouses_df = pd.read_csv(warehouse_data)
    warehouses_df.columns = warehouses_df.columns.str.strip().str.lower()
    warehouses_df.rename(columns={"latitude": "Latitude", "longitude": "Longitude", "name": "name"}, inplace=True)
    st.write("Warehouse Locations:", warehouses_df)
else:
    st.stop()

# ---------- STEP 3: Port/Airstrip Input ----------
st.subheader("Step 3: Mark Port or Airstrip Presence")
warehouses_df["Airstrip_or_Port"] = warehouses_df["name"].apply(lambda x: 1 if st.checkbox(f"Does {x} have a port/airstrip?", key=x) else 0)

# ---------- STEP 4: Ensure Table is correct for TLAMM attributes ----------
st.subheader("Step 4: Ensure Table is correct for TLAMM attributes")
warehouses_df["WEZ"] = warehouses_df.apply(calculate_wez, axis=1)
display_df = warehouses_df.copy()
display_df["Port/Airstrip"] = display_df["Airstrip_or_Port"].apply(lambda x: "Yes" if x == 1 else "No")
display_df["Threat Level"] = display_df["WEZ"].apply(threat_level_label)
cols_to_show = ["name", "Latitude", "Longitude", "Port/Airstrip", "Threat Level"]
st.write("TLAMM attribute table:", display_df[cols_to_show])

# ---------- STEP 5: Run Optimization ----------
st.subheader("Step 5: Run Optimization Model")
if st.button("Run Optimization"):
    I = list(range(len(warehouses_df)))
    J = list(range(len(customers_df)))

    lat_w, lon_w = warehouses_df["Latitude"].values, warehouses_df["Longitude"].values
    lat_c, lon_c = customers_df["Latitude"].values, customers_df["Longitude"].values

    distances = np.zeros((len(I), len(J)))
    for i in I:
        for j in J:
            distances[i, j] = great_circle(lat_w[i], lon_w[i], lat_c[j], lon_c[j])

    fixed_cost_value = 100000
    wez_penalty_multiplier = 20000
    port_bonus_multiplier = 8000
    shipping_cost = 1.0
    P = 15000
    L = 3

    fixed_costs = np.full(len(I), fixed_cost_value)
    adjusted_distances = distances * shipping_cost
    wez_penalty = warehouses_df["WEZ"].values
    port_bonus = warehouses_df["Airstrip_or_Port"].values

    with gp.Env(params=options) as env:
        model = gp.Model("Facility Location", env=env)

        x = model.addVars(I, vtype=GRB.BINARY, name="x")
        y = model.addVars(I, J, vtype=GRB.CONTINUOUS, name="y")
        M = model.addVar(vtype=GRB.CONTINUOUS, name="M")

        model.setObjective(
            quicksum(fixed_costs[i] * x[i] for i in I) +
            quicksum(adjusted_distances[i, j] * y[i, j] for i in I for j in J) +
            quicksum(wez_penalty_multiplier * wez_penalty[i] * x[i] for i in I) -
            quicksum(port_bonus_multiplier * port_bonus[i] * x[i] for i in I) +
            P * M,
            GRB.MINIMIZE
        )

        model.addConstrs((quicksum(y[i, j] for i in I) == 1 for j in J), name="CustomerDemand")
        model.addConstrs((y[i, j] <= x[i] for i in I for j in J), name="Assignment")
        model.addConstr(quicksum(x[i] for i in I) <= L, name="MaxWarehouses")
        model.addConstr(M >= quicksum(x[i] for i in I) - 1, name="ExcessWarehouses")
        model.addConstr(M >= 0, name="NonNegativeExcess")

        model.optimize()

        if model.status == GRB.OPTIMAL:
            assignments = []
            for j in J:
                for i in I:
                    if y[i, j].x > 0.001:
                        customer_name = customers_df.iloc[j]["name"]
                        tlammm_name = warehouses_df.iloc[i]["name"]
                        dist = distances[i, j]
                        assignments.append({
                            "Customer": customer_name,
                            "Assigned TLAMM": tlammm_name,
                            "Distance (km)": round(dist, 2)
                        })

            st.session_state["assignments"] = assignments
            st.session_state["selected"] = [warehouses_df.iloc[i]["name"] for i in I if x[i].x > 0.5]
            st.session_state["obj_val"] = model.objVal
            st.session_state["show_results"] = True

# ---------- DISPLAY RESULTS IF OPTIMIZATION COMPLETED ----------
if st.session_state.get("show_results"):
    selected = st.session_state["selected"]
    obj_val = st.session_state["obj_val"]

    st.success("Optimization Complete!")
    st.write("Selected TLAMM:", selected)
    st.write("Total Objective Value:", obj_val)

    st.subheader("Step 6: Map of RF Bases, Supported Units, and Selected TLAMMs")

    rf_layer = pdk.Layer(
        "ScatterplotLayer",
        data=rf_df,
        get_position='[Longitude, Latitude]',
        get_color='[255, 0, 0]',
        get_radius=50000,
        pickable=True,
    )

    customer_layer = pdk.Layer(
        "ScatterplotLayer",
        data=customers_df,
        get_position='[Longitude, Latitude]',
        get_color='[0, 0, 255]',
        get_radius=30000,
        pickable=True,
    )

    selected_df = warehouses_df[warehouses_df['name'].isin(selected)]
    tlamms_layer = pdk.Layer(
        "ScatterplotLayer",
        data=selected_df,
        get_position='[Longitude, Latitude]',
        get_color='[0, 255, 0]',
        get_radius=60000,
        pickable=True,
    )

    tooltip = {
        "html": "<b>{name}</b>",
        "style": {"backgroundColor": "white", "color": "black"}
    }

    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(latitude=20, longitude=130, zoom=2),
        layers=[rf_layer, customer_layer, tlamms_layer],
        tooltip=tooltip
    ))
        # Custom Legend (Key)
    st.markdown("### Map Legend")
    legend_cols = st.columns(3)

    with legend_cols[0]:
        st.markdown("🟥 **Red**: RF Threat Bases")

    with legend_cols[1]:
        st.markdown("🟦 **Blue**: Supported Units (Customers)")

    with legend_cols[2]:
        st.markdown("🟩 **Green**: Selected TLAMMs")

    if len(selected) > 1:
        if st.button("Show Customer-to-TLAMM Assignments"):
            assignment_df = pd.DataFrame(st.session_state["assignments"])
            st.subheader("Customer Assignments to TLAMMs")
            st.dataframe(assignment_df)
    else:
        st.info("Only one TLAMM selected — all units are served by the same site.")

elif "show_results" in st.session_state:
    st.error("Optimization failed or hasn't been run yet. Upload data and try again.")
