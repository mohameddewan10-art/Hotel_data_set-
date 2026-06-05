import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Hotel Dashboard", layout="wide")
st.title("Hotel Booking Dashboard")

sns.set_theme(style="whitegrid")

# load data
st.cache_data
def load_data():
    df = pd.read_csv("hotels_cleaned.csv")
    df["children"] = df["children"].fillna(0)
    df["country"] = df["country"].fillna("Unknown")
    return df

df = load_data()

# dataset overview
st.header("Dataset Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])
col3.metric("Cancellation Rate", f"{df['is_canceled'].mean()*100:.1f}%")

st.subheader("Sample Data")
st.dataframe(df.head(), use_container_width=True)

st.subheader("Summary Statistics")
st.write(df.describe())

# filter
st.header("Filters")

hotel_type = st.selectbox(
    "Select Hotel Type",
    ["All"] + list(df["hotel"].unique())
)

if hotel_type != "All":
    df = df[df["hotel"] == hotel_type]

# inter active chart
st.header("Interactive Charts")

# split into 2 colomn cuz viz size 
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("ADR by Hotel Type")
    fig1, ax1 = plt.subplots(figsize=(4, 2.5))  
    sns.boxplot(x="hotel", y="adr", data=df, ax=ax1)
    st.pyplot(fig1)

with chart_col2:
    st.subheader("Cancellation by Market Segment")
    fig2, ax2 = plt.subplots(figsize=(4, 2.5))  
    sns.boxplot(x="market_segment", y="is_canceled", data=df, ax=ax2)
    plt.xticks(rotation=45, fontsize=8)        
    st.pyplot(fig2)


st.subheader("Scatter Plot: Lead Time vs ADR")


df_filtered_adr = df[df['adr'] < 500]
sample_size = min(5000, len(df_filtered_adr))
sample = df_filtered_adr.sample(sample_size, random_state=42)

fig3, ax3 = plt.subplots(figsize=(8, 3.5))  
scatter = ax3.scatter(
    sample['lead_time'], sample['adr'],
    c=sample['is_canceled'], cmap='coolwarm',
    alpha=0.4, s=15
)
ax3.set_xlabel("Lead Time")
ax3.set_ylabel("ADR")


cbar = fig3.colorbar(scatter, ax=ax3)
cbar.set_label("Is Canceled")

st.pyplot(fig3)


st.header("Cancellation Prediction Model")

df["lead_time"] = df["lead_time"].fillna(df["lead_time"].median())

X = df[["lead_time"]]
y = df["is_canceled"]

model = LogisticRegression()
model.fit(X, y)

lead_time = st.slider("Select Lead Time (days)", 0, 500, 50)

prediction = model.predict([[lead_time]])[0]

if prediction == 1:
    st.error("High Risk: Booking will be Canceled")
else:
    st.success("Low Risk: Booking will NOT be Canceled")