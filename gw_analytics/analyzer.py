import pandas as pd
import numpy as np
from gwpy.table import EventTable

class GWCatalogAnalyzer:
    def __init__(self, catalog_name="GWTC"):
        """
        Statistical orchestration suite to analyze public Gravitational Wave Transient Catalogs.
        
        Parameters:
        -----------
        catalog_name : str
            The target catalog identifier from GWOSC (e.g., 'GWTC', 'GWTC-1-confident', 'GWTC-3-confident').
            Default 'GWTC' pulls the complete, unified catalog of confirmed mergers.
        """
        self.catalog_name = catalog_name
        self.df = None

    def extract_all_properties(self) -> pd.DataFrame:
        """
        Queries the remote GWOSC catalog registry, fetches every single documented 
        property for all events, and flattens the records into a clean Pandas DataFrame.
        """
        print(f"🤖 Connecting to GWOSC API... Extracting full '{self.catalog_name}' parameter space.")
        try:
            # Fetching without specifying a column matrix forces GWpy to pull ALL columns.
            table = EventTable.fetch_open_data(self.catalog_name)
            self.df = table.to_pandas()
            
            # Use the physical event name (e.g., GW150914) as our primary relational row index
            if 'name' in self.df.columns:
                self.df.set_index('name', inplace=True)
                
            print(f"✅ Success: Extracted {self.df.shape[1]} unique properties across {self.df.shape[0]} documented events.")
            return self.df
        except Exception as e:
            print(f"❌ Failed to extract catalog data: {e}")
            raise

    def generate_statistical_profile(self) -> pd.DataFrame:
        """
        Computes descriptive statistical matrices (mean, median, standard deviation, percentiles) 
        across the entire population for every numerical property parsed.
        """
        if self.df is None:
            self.extract_all_properties()
            
        # Isolate numerical measurements to avoid statistical profiling on metadata strings
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        print("📊 Calculating population distribution properties...")
        # Transpose (.T) the description block for cleaner reading down the rows
        return numeric_df.describe().T

    def calculate_astrophysical_correlations(self, feature_keywords=None) -> pd.DataFrame:
        """
        Computes a correlation matrix across core physical parameters to uncover 
        underlying cosmic structural relationships.
        """
        if self.df is None:
            self.extract_all_properties()
            
        if feature_keywords is None:
            feature_keywords = ['mass', 'spin', 'distance', 'snr']
            
        # Target specific key parameter spaces using string matching on column tokens
        matched_cols = [
            col for col in self.df.columns 
            if any(key in col.lower() for key in feature_keywords)
        ]
        
        numeric_targets = self.df[matched_cols].select_dtypes(include=[np.number])
        return numeric_targets.corr()

    def isolate_outliers(self, column_name: str, top_n: int = 5) -> pd.DataFrame:
        """
        Extracts upper bound outliers for an arbitrary parameter.
        Useful for tracking down things like the highest total mass, closest mergers, or highest SNR.
        """
        if self.df is None:
            self.extract_all_properties()
            
        if column_name not in self.df.columns:
            raise ValueError(f"Column '{column_name}' is missing from this dataset release.")
            
        return self.df.sort_values(by=column_name, ascending=False).head(top_n)