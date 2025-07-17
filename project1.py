import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import matplotlib.pyplot as plt

# === Load Excel Data ===
file_path = r"D:\python\join\pro.xlsx"

# Read sheets
suppliers = pd.read_excel(file_path, sheet_name="supplier")
sales = pd.read_excel(file_path, sheet_name="Sales")
customers = pd.read_excel(file_path, sheet_name="Customers")
products = pd.read_excel(file_path, sheet_name="Product")

# === Merge and Clean ===
sales_customers = pd.merge(sales, customers, on="Customer_ID", how="left")
sales_products = pd.merge(sales_customers, products, on="Product_ID", how="left")
full_data = pd.merge(sales_products, suppliers, on="Supplier_ID", how="left")

clean_data = full_data.dropna(subset=["Name", "Product_Name", "Supplier_Name"])
clean_data["Net_Spend"] = clean_data["Quantity"] * clean_data["Price"] * (1 - clean_data["Discount"])

# === Customer Clustering ===
customer_summary = clean_data.groupby("Customer_ID").agg(
    Total_Spent=("Net_Spend", "sum"),
    Total_Orders=("Sale_ID", "count"),
    Avg_Spend=("Net_Spend", "mean"),
    Last_Date=("Date", "max")
).reset_index()

features = ["Total_Spent", "Total_Orders", "Avg_Spend"]
X = customer_summary[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
customer_summary["Cluster"] = kmeans.fit_predict(X_scaled)

cluster_labels = {
    0: "Mid-Level Customer",
    1: "VIP Customer",
    2: "Inactive Customer"
}
customer_summary["Cluster_Label"] = customer_summary["Cluster"].map(cluster_labels)

# === Forecast Total Sales ===
daily_sales = clean_data.groupby("Date").agg(
    total_sales=("Net_Spend", "sum")
).reset_index()
df_prophet = daily_sales.rename(columns={"Date": "ds", "total_sales": "y"})
model_total = Prophet()
model_total.fit(df_prophet)
future_total = model_total.make_future_dataframe(periods=90)
forecast_total = model_total.predict(future_total)

# === Forecast Per Product ===
xls = pd.ExcelFile(file_path)
product_sheet_names = []
forecast_dict = {}
for product in clean_data["Product_Name"].unique():
    product_data = clean_data[clean_data["Product_Name"] == product]
    daily_product_sales = product_data.groupby("Date").agg(total_sales=("Net_Spend", "sum")).reset_index()
    if len(daily_product_sales) < 10:
        continue
    df_prod = daily_product_sales.rename(columns={"Date": "ds", "total_sales": "y"})
    model = Prophet()
    model.fit(df_prod)
    future = model.make_future_dataframe(periods=90)
    forecast = model.predict(future)
    forecast["Product_Name"] = product
    sheet_name = product[:31].replace('/', '_').replace('\\', '_')
    forecast_dict[sheet_name] = forecast
    product_sheet_names.append(sheet_name)

# === Dash App ===
app = Dash(__name__)
app.title = "Sales & Customer Dashboard"

app.layout = html.Div([
    html.H1("\ud83d\udcca Sales & Customer Dashboard", style={"textAlign": "center"}),

    dcc.Tabs([
        dcc.Tab(label="Customer Segmentation", children=[
            html.H3("Customer Cluster Distribution"),
            dcc.Graph(
                figure=px.pie(customer_summary, names="Cluster_Label", title="Customer Segments")
            ),
            html.H3("Top VIP Customers"),
            dcc.Graph(
                figure=px.bar(
                    customer_summary[customer_summary["Cluster_Label"] == "VIP Customer"].nlargest(10, "Total_Spent"),
                    x="Customer_ID", y="Total_Spent", title="Top 10 VIP Customers"
                )
            )
        ]),

        dcc.Tab(label="Total Sales Forecast", children=[
            html.H3("3-Month Sales Forecast"),
            dcc.Graph(
                figure=px.line(forecast_total, x="ds", y="yhat", title="Total Sales Forecast")
            )
        ]),

        dcc.Tab(label="Product Forecast", children=[
            html.Label("Select a Product:"),
            dcc.Dropdown(product_sheet_names, value=product_sheet_names[0], id='product-dropdown'),

            dcc.Graph(id="product-forecast-graph")
        ])
    ])
])

# === Callbacks ===
@app.callback(
    Output("product-forecast-graph", "figure"),
    Input("product-dropdown", "value")
)
def update_product_forecast(product_sheet):
    df = forecast_dict[product_sheet]
    fig = px.line(df, x="ds", y="yhat", title=f"Forecast for {product_sheet}")
    return fig

# === Run App ===
if __name__ == '__main__':
    app.run(debug=True)
