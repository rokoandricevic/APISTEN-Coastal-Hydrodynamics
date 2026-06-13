import os
import glob
import re
import pandas as pd
import numpy as np

def main():
    # --- Configuration ---
    # Path to the master meteorological directory provided by Saša
    base_meteo_dir = "output_meteo"
    
    # Hydrodynamic threshold filters defined by the physics of the basin
    WIND_DIR_MIN = 130.0   # Minimum SE Jugo angle (degrees)
    WIND_DIR_MAX = 170.0   # Maximum SE Jugo angle (degrees)
    WIND_SPEED_MIN = 10.0  # Kinetic forcing threshold (m/s)
    PRESSURE_MAX = 1005.0  # Cyclonic barometric low threshold (hPa)

    if not os.path.exists(base_meteo_dir):
        # Fallback check: see if the user is executing directly inside output_meteo
        if os.path.exists(".") and any(re.match(r"\d{4}-\d{2}-\d{2}", f) for f in os.listdir(".")):
            base_meteo_dir = "."
        else:
            raise FileNotFoundError(f"Cannot locate parent meteorological directory: '{base_meteo_dir}'")

    print("=======================================================================")
    print(f"Commencing Global ERA5 Screen Matrix Inside: '{base_meteo_dir}'")
    print(f"Constraints: Dir [{WIND_DIR_MIN}°-{WIND_DIR_MAX}°] | Speed > {WIND_SPEED_MIN} m/s | Pressure < {PRESSURE_MAX} hPa")
    print("=======================================================================\n")

    # Search pattern to identify all date-formatted subdirectories
    search_path = os.path.join(base_meteo_dir, "????-??-??")
    date_folders = sorted(glob.glob(search_path))
    
    discovered_events = []

    for folder in date_folders:
        date_str = os.path.basename(folder)
        
        # Construct the target file name matching Saša's CSV structure
        filename = f"{date_str}_meteo.csv"
        filepath = os.path.join(folder, filename)
        
        if not os.path.exists(filepath):
            continue
            
        try:
            # Read hourly weather data points
            df = pd.read_csv(filepath)
            
            # Enforce clean lowercase naming convention to avoid parsing errors
            df.columns = [c.strip().lower() for c in df.columns]
            
            # Apply your physical Boolean constraints simultaneously across the rows
            match_mask = (
                (df['wind_direction_10m'] >= WIND_DIR_MIN) & 
                (df['wind_direction_10m'] <= WIND_DIR_MAX) & 
                (df['wind_speed_10m'] > WIND_SPEED_MIN) & 
                (df['surface_pressure'] < PRESSURE_MAX)
            )
            
            df_matching = df[match_mask]
            
            if len(df_matching) > 0:
                # Group matches geographically to see if the storm hit the West or East grid cell
                # Longitude < 16.3 maps to the Western Bay sector, otherwise Eastern Canal sector
                hours_west = np.sum(df_matching['lon'] < 16.3)
                hours_east = np.sum(df_matching['lon'] >= 16.3)
                
                # Extract representative peak metrics observed during the storm window
                peak_speed = df_matching['wind_speed_10m'].max()
                min_pressure = df_matching['surface_pressure'].min()
                
                discovered_events.append({
                    "Date": date_str,
                    "Total_Storm_Hours": len(df_matching),
                    "Hours_West_Bay": hours_west,
                    "Hours_East_Canal": hours_east,
                    "Peak_Wind_Speed_ms": peak_speed,
                    "Min_Pressure_hPa": min_pressure,
                    "File_Path": filepath
                })
        except Exception as e:
            print(f"Skipping file {filename} due to reading conflict: {str(e)}")
            continue

    if not discovered_events:
        print("Screening Complete: No historical dates found matching your exact threshold constraints.")
        return

    # Convert the list of discovered storm events into a structured DataFrame
    df_results = pd.DataFrame(discovered_events)
    
    # Sort the dates by total storm hours to highlight the most intense events
    df_results = df_results.sort_values(by="Total_Storm_Hours", ascending=False).reset_index(drop=True)
    
    # Save the output index to a master spreadsheet for your records
    output_summary = "APISTEN_Identified_Jugo_Storm_Dates.csv"
    df_results.to_csv(output_summary, index=False)
    
    print("=== SCREENING ANALYSIS COMPLETE ===")
    print(f"Discovered {len(df_results)} dates matching all hydrodynamic forcing criteria.")
    print(f"Master summary file saved to: '{output_summary}'\n")
    print(df_results.drop(columns=["File_Path"]).to_string(index=False))

if __name__ == "__main__":
    main()