#!/usr/bin/env python3
"""
Marketing Campaign Automated Reporting Script
This script generates regular performance reports for the marketing campaign analysis dashboard.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from fpdf import FPDF


def load_integrated_data(file_path):
    """
    Load the integrated marketing data
    
    Args:
        file_path (str): Path to the integrated data file
        
    Returns:
        pandas.DataFrame: Loaded data
    """
    print(f"Loading integrated marketing data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} records.")
    return df


def generate_summary_statistics(df):
    """
    Generate summary statistics for the report
    
    Args:
        df (pandas.DataFrame): Integrated marketing data
        
    Returns:
        dict: Summary statistics
    """
    print("Generating summary statistics...")
    summary = {
        'Total Spend': df['spend'].sum(),
        'Total Impressions': df['impressions'].sum(),
        'Total Clicks': df['clicks'].sum(),
        'Average CTR': df['ctr'].mean(),
        'Average CPC': df['cpc'].mean(),
        'Average CPA': df['cpa'].mean(),
        'Average ROAS': df['roas'].mean()
    }
    return summary


def create_visualizations(df, output_dir):
    """
    Create visualizations for the report
    
    Args:
        df (pandas.DataFrame): Integrated marketing data
        output_dir (str): Directory to save visualizations
    """
    print("Creating visualizations...")
    sns.set(style="whitegrid")
    
    # Plot total spend over time
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='campaign_date', y='spend', estimator='sum')
    plt.title('Total Spend Over Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    spend_plot_path = os.path.join(output_dir, 'total_spend_over_time.png')
    plt.savefig(spend_plot_path)
    plt.close()
    print(f"Saved spend plot to {spend_plot_path}")
    
    # Plot total impressions over time
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='campaign_date', y='impressions', estimator='sum')
    plt.title('Total Impressions Over Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    impressions_plot_path = os.path.join(output_dir, 'total_impressions_over_time.png')
    plt.savefig(impressions_plot_path)
    plt.close()
    print(f"Saved impressions plot to {impressions_plot_path}")


def compile_report(summary, visualizations, output_path):
    """
    Compile the report into a PDF
    
    Args:
        summary (dict): Summary statistics
        visualizations (list): List of visualization file paths
        output_path (str): Path to save the compiled report
    """
    print("Compiling report...")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Add title
    pdf.cell(200, 10, txt="Marketing Campaign Performance Report", ln=True, align='C')
    pdf.ln(10)
    
    # Add summary statistics
    pdf.cell(200, 10, txt="Summary Statistics:", ln=True)
    for key, value in summary.items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)
    pdf.ln(10)
    
    # Add visualizations
    for viz_path in visualizations:
        pdf.add_page()
        pdf.image(viz_path, x=10, y=20, w=180)
    
    # Save the PDF
    pdf.output(output_path)
    print(f"Report compiled and saved to {output_path}")


def main():
    # Define file paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    integrated_file = os.path.join(base_dir, 'data', 'integrated_marketing_data.csv')
    report_dir = os.path.join(base_dir, 'reports')
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, f"marketing_report_{datetime.now().strftime('%Y%m%d')}.pdf")
    
    # Load integrated data
    df = load_integrated_data(integrated_file)
    
    # Generate summary statistics
    summary = generate_summary_statistics(df)
    
    # Create visualizations
    create_visualizations(df, report_dir)
    
    # Compile report
    visualizations = [
        os.path.join(report_dir, 'total_spend_over_time.png'),
        os.path.join(report_dir, 'total_impressions_over_time.png')
    ]
    compile_report(summary, visualizations, report_path)

    print("\nAutomated reporting completed successfully!")


if __name__ == "__main__":
    main() 