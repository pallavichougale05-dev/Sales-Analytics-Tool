import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np

# Optional libraries
# If these are not installed, code will still run
try:
    from sklearn.linear_model import LinearRegression
    sklearn_available = True
except ImportError:
    sklearn_available = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    reportlab_available = True
except ImportError:
    reportlab_available = False

OUTPUT_DIR = "outputs"

def create_output_folder():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

# Print section title in terminal Makes output look cleaner
def print_section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError("File not found. Please check the path.")

    return pd.read_csv(file_path)


# Check which useful columns are present
def check_columns(df):
    expected = ['Date', 'Product', 'Category', 'Units Sold', 'Unit Price', 'Cost Price', 'Region']

    print_section("COLUMN CHECK")
    print("Available Columns:", list(df.columns))

    missing = [col for col in expected if col not in df.columns]
    if missing:
        print("⚠ Missing Columns:", missing)
    else:
        print("✅ All expected columns are present.")


def clean_data(df):             # Clean the data safely
    df = df.copy()

    if 'Date' in df.columns:        # Convert Date column if available
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    df.drop_duplicates(inplace=True)        # Remove duplicate rows

    # Remove missing values from important available columns
    important_cols = [col for col in ['Date', 'Units Sold', 'Unit Price', 'Cost Price'] if col in df.columns]
    if important_cols:
        df.dropna(subset=important_cols, inplace=True)

    # Remove invalid numeric values
    if 'Units Sold' in df.columns:
        df = df[df['Units Sold'] > 0]

    if 'Unit Price' in df.columns:
        df = df[df['Unit Price'] > 0]

    if 'Cost Price' in df.columns:
        df = df[df['Cost Price'] > 0]

    if 'Units Sold' in df.columns and 'Unit Price' in df.columns:       # Create Total Sales if possible
        df['Total Sales'] = df['Units Sold'] * df['Unit Price']

    # Create Profit if possible
    if 'Units Sold' in df.columns and 'Unit Price' in df.columns and 'Cost Price' in df.columns:
        df['Profit'] = (df['Unit Price'] - df['Cost Price']) * df['Units Sold']

    if 'Date' in df.columns:            # Extract Month from Date for monthly trend
        df['Month'] = df['Date'].dt.to_period('M').astype(str)

    return df


def analyze_data(df):           # Perform all possible analysis
    analysis = {}

    if 'Total Sales' in df.columns:         # Total revenue
        analysis['total_revenue'] = df['Total Sales'].sum()

    if 'Profit' in df.columns:          # Total profit
        analysis['total_profit'] = df['Profit'].sum()

    # Category-wise revenue and profit
    if 'Category' in df.columns and 'Total Sales' in df.columns:
        analysis['rev_by_cat'] = df.groupby('Category')['Total Sales'].sum().sort_values(ascending=False)

    if 'Category' in df.columns and 'Profit' in df.columns:
        analysis['profit_by_cat'] = df.groupby('Category')['Profit'].sum().sort_values(ascending=False)

    # Product-wise sales
    if 'Product' in df.columns and 'Total Sales' in df.columns:
        product_sales = df.groupby('Product')['Total Sales'].sum().sort_values(ascending=False)
        analysis['top_products'] = product_sales
        analysis['bottom_products'] = product_sales.sort_values().head(5)

    # Monthly trends
    if 'Month' in df.columns and 'Total Sales' in df.columns:
        analysis['monthly_sales'] = df.groupby('Month')['Total Sales'].sum()

    if 'Month' in df.columns and 'Profit' in df.columns:
        analysis['monthly_profit'] = df.groupby('Month')['Profit'].sum()

    # Region-wise sales
    if 'Region' in df.columns and 'Total Sales' in df.columns:
        analysis['region_revenue'] = df.groupby('Region')['Total Sales'].sum().sort_values(ascending=False)

    return analysis

def print_results(analysis):        # Print analysis in terminal
    print_section("SALES ANALYSIS SUMMARY")

    if 'total_revenue' in analysis:
        print(f"📌 Total Revenue : ₹{analysis['total_revenue']:,.2f}")

    if 'total_profit' in analysis:
        print(f"📌 Total Profit  : ₹{analysis['total_profit']:,.2f}")

    if 'rev_by_cat' in analysis:
        print_section("REVENUE BY CATEGORY")
        print(analysis['rev_by_cat'])

    if 'profit_by_cat' in analysis:
        print_section("PROFIT BY CATEGORY")
        print(analysis['profit_by_cat'])

    if 'top_products' in analysis:
        print_section("TOP 5 PRODUCTS")
        print(analysis['top_products'].head(5))

    if 'bottom_products' in analysis:
        print_section("BOTTOM 5 PRODUCTS")
        print(analysis['bottom_products'])

    if 'monthly_sales' in analysis:
        print_section("MONTHLY SALES")
        print(analysis['monthly_sales'])

    if 'region_revenue' in analysis:
        print_section("REGION REVENUE")
        print(analysis['region_revenue'])


def generate_insights(analysis):        # Generate simple business insights
    insights = []

    if 'rev_by_cat' in analysis:
        insights.append(f"Highest revenue category: {analysis['rev_by_cat'].idxmax()}")

    if 'profit_by_cat' in analysis:
        insights.append(f"Most profitable category: {analysis['profit_by_cat'].idxmax()}")

    if 'top_products' in analysis:
        insights.append(f"Top selling product: {analysis['top_products'].idxmax()}")

    if 'bottom_products' in analysis:
        insights.append(f"Lowest performing product: {analysis['bottom_products'].idxmin()}")

    if 'monthly_sales' in analysis:
        insights.append(f"Best sales month: {analysis['monthly_sales'].idxmax()}")

    if 'region_revenue' in analysis:
        insights.append(f"Top revenue region: {analysis['region_revenue'].idxmax()}")

    return insights


# Save reports as CSV files
def save_reports(df, analysis):
    # Save cleaned data
    df.to_csv(os.path.join(OUTPUT_DIR, "cleaned_sales_data.csv"), index=False)

    # Save summary report
    summary = []
    if 'total_revenue' in analysis:
        summary.append(['Total Revenue', analysis['total_revenue']])
    if 'total_profit' in analysis:
        summary.append(['Total Profit', analysis['total_profit']])

    if summary:
        pd.DataFrame(summary, columns=['Metric', 'Value']).to_csv(
            os.path.join(OUTPUT_DIR, "summary_report.csv"), index=False
        )

    # Save detailed reports only if available
    if 'rev_by_cat' in analysis:
        analysis['rev_by_cat'].to_csv(os.path.join(OUTPUT_DIR, "category_revenue.csv"))

    if 'profit_by_cat' in analysis:
        analysis['profit_by_cat'].to_csv(os.path.join(OUTPUT_DIR, "category_profit.csv"))

    if 'monthly_sales' in analysis:
        analysis['monthly_sales'].to_csv(os.path.join(OUTPUT_DIR, "monthly_sales.csv"))

    if 'monthly_profit' in analysis:
        analysis['monthly_profit'].to_csv(os.path.join(OUTPUT_DIR, "monthly_profit.csv"))

    if 'top_products' in analysis:
        analysis['top_products'].head(10).to_csv(os.path.join(OUTPUT_DIR, "top_products.csv"))

    if 'region_revenue' in analysis:
        analysis['region_revenue'].to_csv(os.path.join(OUTPUT_DIR, "region_revenue.csv"))


# Create charts and save them as PNG
def create_visualizations(analysis):

    if 'rev_by_cat' in analysis:            # Revenue by category chart
        plt.figure(figsize=(8, 5))
        analysis['rev_by_cat'].plot(kind='bar')
        plt.title('Revenue by Category')
        plt.ylabel('Revenue (₹)')
        plt.xlabel('Category')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "category_revenue.png"))
        plt.close()

    if 'monthly_sales' in analysis:          # Monthly sales trend chart
        plt.figure(figsize=(8, 5))
        analysis['monthly_sales'].plot(kind='line', marker='o')
        plt.title('Monthly Sales Trend')
        plt.ylabel('Revenue (₹)')
        plt.xlabel('Month')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "monthly_sales.png"))
        plt.close()

    # Top 5 products chart
    if 'top_products' in analysis:
        plt.figure(figsize=(8, 5))
        analysis['top_products'].head(5).plot(kind='bar')
        plt.title('Top 5 Products by Revenue')
        plt.ylabel('Revenue (₹)')
        plt.xlabel('Product')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "top_products.png"))
        plt.close()


# Forecast next 3 months sale Works only if:
# 1. sklearn is installed
# 2. monthly sales data exists
# 3. at least 3 months of data are available
def forecast_sales(analysis):
    if not sklearn_available:
        print("\n⚠ scikit-learn not installed. Skipping forecasting.")
        return None

    if 'monthly_sales' not in analysis:
        print("\n⚠ Monthly sales data not available. Skipping forecasting.")
        return None

    monthly_sales = analysis['monthly_sales']

    if len(monthly_sales) < 3:
        print("\n⚠ Not enough monthly data for forecasting.")
        return None

    X = np.arange(len(monthly_sales)).reshape(-1, 1)    # X = month numbers (0,1,2,3...)

    y = monthly_sales.values                            # y = sales values

    model = LinearRegression()                          # Train simple linear regression model
    model.fit(X, y)

    # Predict next 3 months
    future_X = np.arange(len(monthly_sales), len(monthly_sales) + 3).reshape(-1, 1)
    predicted_values = model.predict(future_X)

    # Create future month labels
    last_month = pd.Period(monthly_sales.index[-1], freq='M')
    future_months = [(last_month + i).strftime('%Y-%m') for i in range(1, 4)]

    forecast_df = pd.DataFrame({
        'Month': future_months,
        'Predicted Sales': predicted_values
    })

    # Save forecast CSV
    forecast_df.to_csv(os.path.join(OUTPUT_DIR, "sales_forecast.csv"), index=False)

    return forecast_df

# Generate PDF report
def generate_pdf_report(analysis, insights, forecast_df):
    if not reportlab_available:
        print("\n⚠ reportlab not installed. Skipping PDF report.")
        return

    pdf_path = os.path.join(OUTPUT_DIR, "sales_report.pdf")
    pdf = canvas.Canvas(pdf_path, pagesize=A4)

    width, height = A4
    y = height - 50

    pdf.setFont("Helvetica-Bold", 16)           # Title
    pdf.drawString(50, y, "Sales Analytics Report")
    y -= 30

    pdf.setFont("Helvetica", 11)                # Date
    pdf.drawString(50, y, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 30

    pdf.setFont("Helvetica-Bold", 13)          # Summary
    pdf.drawString(50, y, "Summary")
    y -= 20

    pdf.setFont("Helvetica", 11)
    if 'total_revenue' in analysis:
        pdf.drawString(50, y, f"Total Revenue: ₹{analysis['total_revenue']:,.2f}")
        y -= 20
    if 'total_profit' in analysis:
        pdf.drawString(50, y, f"Total Profit: ₹{analysis['total_profit']:,.2f}")
        y -= 20

    y -= 10                                     # Insights
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(50, y, "Business Insights")
    y -= 20

    pdf.setFont("Helvetica", 11)
    for insight in insights:
        pdf.drawString(60, y, f"- {insight}")
        y -= 18

    if forecast_df is not None:             # Forecast section
        y -= 20
        pdf.setFont("Helvetica-Bold", 13)
        pdf.drawString(50, y, "Sales Forecast")
        y -= 20

        pdf.setFont("Helvetica", 11)
        for _, row in forecast_df.iterrows():
            pdf.drawString(60, y, f"{row['Month']} : ₹{row['Predicted Sales']:,.2f}")
            y -= 18

    pdf.save()


# MAIN FUNCTION
def main():
    print_section("FLEXIBLE SALES ANALYTICS TOOL")

    file_path = input("\nEnter the path to your sales CSV file: ").strip()

    try:
        create_output_folder()               # Step 1: Setup

        df = load_data(file_path)            # Step 2: Load and inspect data
        check_columns(df)

        df = clean_data(df)                  # Step 3: Clean data

        analysis = analyze_data(df)          # Step 4: Analyze

        print_results(analysis)              # Step 5: Print output

        insights = generate_insights(analysis)  # Step 6: Generate insights
        if insights:
            print_section("BUSINESS INSIGHTS")
            for i, insight in enumerate(insights, 1):
                print(f"{i}. {insight}")

        save_reports(df, analysis)          # Step 7: Save reports and charts
        create_visualizations(analysis)

        forecast_df = forecast_sales(analysis)    # Step 8: Forecast sales
        if forecast_df is not None:
            print_section("SALES FORECAST (NEXT 3 MONTHS)")
            print(forecast_df)

        generate_pdf_report(analysis, insights, forecast_df)    # Step 9: Generate PDF report

        print_section("OUTPUT GENERATED SUCCESSFULLY")          # Final message
        print(f"📁 All possible reports saved in '{OUTPUT_DIR}' folder.")

    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()