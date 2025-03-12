#!/usr/bin/env python3
"""
Marketing Campaign Data Integration Script
This script integrates the processed campaign performance data with the expanded marketing data
to create a unified dataset for comprehensive analysis.
"""

import pandas as pd
import numpy as np
import os
import re
from datetime import datetime

def load_processed_data(campaign_file, expanded_file):
    """
    Load processed campaign and expanded marketing data
    
    Args:
        campaign_file (str): Path to processed campaign data
        expanded_file (str): Path to processed expanded marketing data
        
    Returns:
        tuple: (campaign_df, expanded_df) - Loaded dataframes
    """
    print(f"Loading processed campaign data from {campaign_file}...")
    campaign_df = pd.read_csv(campaign_file)
    print(f"Loaded {len(campaign_df)} campaign records.")
    
    print(f"Loading processed expanded marketing data from {expanded_file}...")
    expanded_df = pd.read_csv(expanded_file)
    print(f"Loaded {len(expanded_df)} expanded marketing records.")
    
    return campaign_df, expanded_df

def extract_campaign_id_from_name(campaign_name):
    """
    Extract a standardized campaign ID from campaign name for matching
    
    Args:
        campaign_name (str): Campaign name
        
    Returns:
        str: Standardized campaign ID
    """
    # Remove spaces, convert to lowercase
    clean_name = campaign_name.lower().replace(' ', '')
    # Extract year if present
    year_match = re.search(r'20\d{2}', clean_name)
    year = year_match.group(0) if year_match else ""
    
    # Create a standardized ID
    return f"CAMP-{clean_name}-{year}"

def create_matching_keys(campaign_df, expanded_df):
    """
    Create matching keys in both datasets to facilitate integration
    
    Args:
        campaign_df (pandas.DataFrame): Campaign performance data
        expanded_df (pandas.DataFrame): Expanded marketing data
        
    Returns:
        tuple: (campaign_df, expanded_df) - Dataframes with matching keys
    """
    print("Creating matching keys for data integration...")
    
    # Create copies to avoid modifying originals
    campaign_df = campaign_df.copy()
    expanded_df = expanded_df.copy()
    
    # Create standardized campaign IDs in campaign data
    campaign_df['matching_id'] = campaign_df['campaign_name'].apply(extract_campaign_id_from_name)
    
    # Extract campaign name from expanded data campaign_id
    expanded_df['campaign_name_derived'] = expanded_df['campaign_id'].str.replace('CAMP-', '')
    
    # Create date-based matching
    # Convert campaign_date to datetime if it's not already
    expanded_df['campaign_date'] = pd.to_datetime(expanded_df['campaign_date'])
    
    # Extract year and month for matching
    expanded_df['matching_year'] = expanded_df['campaign_date'].dt.year
    expanded_df['matching_month'] = expanded_df['campaign_date'].dt.month
    
    # Create a date range match flag
    campaign_df['start_date'] = pd.to_datetime(campaign_df['start_date'])
    campaign_df['end_date'] = pd.to_datetime(campaign_df['end_date'])
    
    return campaign_df, expanded_df

def match_by_fuzzy_date(campaign_df, expanded_df):
    """
    Match records based on fuzzy date matching
    
    Args:
        campaign_df (pandas.DataFrame): Campaign performance data with matching keys
        expanded_df (pandas.DataFrame): Expanded marketing data with matching keys
        
    Returns:
        pandas.DataFrame: Matched records with campaign_id mapping
    """
    print("Performing fuzzy date matching...")
    
    # Create a mapping dictionary
    campaign_id_mapping = {}
    
    # For each expanded data record
    for _, exp_row in expanded_df.iterrows():
        exp_date = exp_row['campaign_date']
        exp_id = exp_row['campaign_id']
        
        # Find campaigns that overlap with this date
        matching_campaigns = campaign_df[
            (campaign_df['start_date'] <= exp_date) & 
            (campaign_df['end_date'] >= exp_date)
        ]
        
        if not matching_campaigns.empty:
            # If multiple matches, take the one with the closest start date
            if len(matching_campaigns) > 1:
                matching_campaigns['date_diff'] = abs((matching_campaigns['start_date'] - exp_date).dt.days)
                best_match = matching_campaigns.sort_values('date_diff').iloc[0]
                campaign_id_mapping[exp_id] = best_match['campaign_id']
            else:
                campaign_id_mapping[exp_id] = matching_campaigns.iloc[0]['campaign_id']
    
    # Create a mapping dataframe
    mapping_df = pd.DataFrame({
        'expanded_campaign_id': list(campaign_id_mapping.keys()),
        'campaign_id': list(campaign_id_mapping.values())
    })
    
    print(f"Created {len(mapping_df)} campaign ID mappings.")
    return mapping_df

def match_by_channel(campaign_df, expanded_df, mapping_df):
    """
    Refine matches based on channel/platform information
    
    Args:
        campaign_df (pandas.DataFrame): Campaign performance data
        expanded_df (pandas.DataFrame): Expanded marketing data
        mapping_df (pandas.DataFrame): Initial ID mapping
        
    Returns:
        pandas.DataFrame: Refined mapping dataframe
    """
    print("Refining matches based on channel/platform information...")
    
    # Create a copy of the mapping
    refined_mapping = mapping_df.copy()
    
    # Create channel mapping
    channel_platform_map = {
        'Facebook': 'Facebook Ads',
        'Instagram': 'Instagram Ads',
        'Google Ads': 'Google Ads',
        'LinkedIn': 'LinkedIn Ads',
        'Twitter': 'Twitter Ads',
        'TikTok': 'TikTok Ads'
    }
    
    # For each mapping
    for i, row in refined_mapping.iterrows():
        exp_id = row['expanded_campaign_id']
        camp_id = row['campaign_id']
        
        # Get the expanded data record
        exp_record = expanded_df[expanded_df['campaign_id'] == exp_id].iloc[0]
        
        # Get the campaign record
        camp_record = campaign_df[campaign_df['campaign_id'] == camp_id].iloc[0]
        
        # Check if the channels match
        exp_platform = exp_record['ad_platform']
        camp_channel = camp_record['channel']
        
        # If we have a mismatch, try to find a better match
        if channel_platform_map.get(camp_channel, '') != exp_platform:
            # Find campaigns with matching dates and matching channel
            matching_campaigns = campaign_df[
                (campaign_df['start_date'] <= exp_record['campaign_date']) & 
                (campaign_df['end_date'] >= exp_record['campaign_date']) &
                (campaign_df['channel'].apply(lambda x: channel_platform_map.get(x, '')) == exp_platform)
            ]
            
            if not matching_campaigns.empty:
                # Update the mapping
                refined_mapping.at[i, 'campaign_id'] = matching_campaigns.iloc[0]['campaign_id']
                refined_mapping.at[i, 'channel_match'] = True
            else:
                refined_mapping.at[i, 'channel_match'] = False
        else:
            refined_mapping.at[i, 'channel_match'] = True
    
    print(f"Refined {len(refined_mapping[refined_mapping['channel_match']])} mappings with channel matching.")
    return refined_mapping

def integrate_datasets(campaign_df, expanded_df, mapping_df):
    """
    Integrate the datasets based on the mapping
    
    Args:
        campaign_df (pandas.DataFrame): Campaign performance data
        expanded_df (pandas.DataFrame): Expanded marketing data
        mapping_df (pandas.DataFrame): ID mapping
        
    Returns:
        pandas.DataFrame: Integrated dataset
    """
    print("Integrating datasets...")
    
    # Create a copy of expanded data
    integrated_df = expanded_df.copy()
    
    # Add the campaign_id from mapping
    integrated_df = integrated_df.merge(
        mapping_df[['expanded_campaign_id', 'campaign_id']], 
        left_on='campaign_id', 
        right_on='expanded_campaign_id', 
        how='left'
    )
    
    # Rename columns to avoid confusion
    integrated_df = integrated_df.rename(columns={
        'campaign_id_x': 'expanded_campaign_id',
        'campaign_id_y': 'campaign_id'
    })
    
    # For records with a valid campaign_id mapping, merge campaign data
    campaign_columns = [
        'campaign_id', 'campaign_name', 'channel', 'start_date', 'end_date',
        'budget', 'spend', 'impressions', 'clicks', 'conversions', 'revenue',
        'ctr', 'cpc', 'cpa', 'roas', 'campaign_duration', 'budget_utilization'
    ]
    
    # Merge with campaign data
    integrated_df = integrated_df.merge(
        campaign_df[campaign_columns],
        on='campaign_id',
        how='left'
    )
    
    # Calculate integrated metrics
    integrated_df['integrated_ctr'] = np.where(
        integrated_df['impressions'].notna(),
        integrated_df['clicks'] / integrated_df['impressions'],
        integrated_df['ad_CTR']
    )
    
    integrated_df['integrated_cpc'] = np.where(
        integrated_df['clicks'].notna() & (integrated_df['clicks'] > 0),
        integrated_df['spend'] / integrated_df['clicks'],
        integrated_df['ad_CPC']
    )
    
    # Create a match quality indicator
    integrated_df['match_quality'] = np.where(
        integrated_df['campaign_id'].isna(),
        'No Match',
        'Matched'
    )
    
    print(f"Created integrated dataset with {len(integrated_df)} records.")
    print(f"Successfully matched {len(integrated_df[integrated_df['match_quality'] == 'Matched'])} records.")
    
    return integrated_df

def save_integrated_data(df, output_path):
    """
    Save the integrated data to a CSV file
    
    Args:
        df (pandas.DataFrame): Integrated data
        output_path (str): Path to save the integrated data
    """
    print(f"Saving integrated data to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Integrated data saved successfully.")

def main():
    # Define file paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    campaign_file = os.path.join(base_dir, 'data', 'processed_campaign_data.csv')
    expanded_file = os.path.join(base_dir, 'data', 'processed_expanded_data.csv')
    mapping_file = os.path.join(base_dir, 'data', 'campaign_id_mapping.csv')
    integrated_file = os.path.join(base_dir, 'data', 'integrated_marketing_data.csv')
    
    # Load processed data
    campaign_df, expanded_df = load_processed_data(campaign_file, expanded_file)
    
    # Create matching keys
    campaign_df, expanded_df = create_matching_keys(campaign_df, expanded_df)
    
    # Match by fuzzy date
    mapping_df = match_by_fuzzy_date(campaign_df, expanded_df)
    
    # Refine matches by channel
    refined_mapping = match_by_channel(campaign_df, expanded_df, mapping_df)
    
    # Save the mapping for reference
    refined_mapping.to_csv(mapping_file, index=False)
    print(f"Saved campaign ID mapping to {mapping_file}")
    
    # Integrate the datasets
    integrated_df = integrate_datasets(campaign_df, expanded_df, refined_mapping)
    
    # Save the integrated data
    save_integrated_data(integrated_df, integrated_file)
    
    print("\nData integration completed successfully!")
    print(f"Integrated data saved to {integrated_file}")

if __name__ == "__main__":
    main() 