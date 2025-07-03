"""Data loading utilities for OECT transfer analysis."""

import os
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

try:
    from oect_transfer import Transfer
except ImportError:
    raise ImportError(
        "oect-transfer package is required. Install with: pip install oect-transfer"
    )


class DataLoader:
    """Enhanced data loader for OECT transfer curve files."""
    
    def __init__(self, folder_path: Union[str, Path]):
        """
        Initialize the data loader.
        
        Parameters
        ----------
        folder_path : str or Path
            Path to the folder containing CSV files
        """
        self.folder_path = Path(folder_path)
        if not self.folder_path.exists():
            raise FileNotFoundError(f"Folder does not exist: {folder_path}")
        
        self.transfer_objects = []
        self.metadata = {}
    
    def load_all_files(
        self,
        device_type: str = "N",
        file_pattern: str = "transfer",
        sort_numerically: bool = True,
        vg_column: Optional[str] = None,
        id_column: Optional[str] = None,
        verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Load all transfer CSV files and create Transfer objects.
        
        Parameters
        ----------
        device_type : str, default "N"
            Device type: "N" for N-type, "P" for P-type
        file_pattern : str, default "transfer"
            File name pattern to match
        sort_numerically : bool, default True
            Whether to sort files numerically instead of alphabetically
        vg_column : str, optional
            Specific column name for gate voltage. If None, auto-detects
        id_column : str, optional
            Specific column name for drain current. If None, auto-detects
        verbose : bool, default True
            Whether to print progress information
            
        Returns
        -------
        List[Dict[str, Any]]
            List of dictionaries containing Transfer objects and metadata
        """
        if verbose:
            print(f"Loading files from: {self.folder_path}")
        
        # Find matching files
        files = list(self.folder_path.glob("*.csv"))
        transfer_files = [
            f for f in files 
            if file_pattern.lower() in f.name.lower()
        ]
        
        if not transfer_files:
            raise ValueError(
                f"No CSV files found matching pattern '{file_pattern}' "
                f"in {self.folder_path}"
            )
        
        # Sort files
        if sort_numerically:
            transfer_files.sort(key=self._extract_number_from_filename)
        else:
            transfer_files.sort()
        
        if verbose:
            print(f"Found {len(transfer_files)} transfer files")
        
        self.transfer_objects = []
        successful_loads = 0
        
        for file_path in transfer_files:
            try:
                transfer_obj = self._load_single_file(
                    file_path, device_type, vg_column, id_column, verbose
                )
                if transfer_obj:
                    self.transfer_objects.append(transfer_obj)
                    successful_loads += 1
                    
            except Exception as e:
                if verbose:
                    print(f"  ✗ Failed to load {file_path.name}: {e}")
                continue
        
        if verbose:
            print(f"\nSuccessfully loaded {successful_loads} files")
        
        # Store metadata
        self.metadata = {
            "total_files": len(transfer_files),
            "successful_loads": successful_loads, 
            "device_type": device_type,
            "folder_path": str(self.folder_path)
        }
        
        return self.transfer_objects
    
    def _load_single_file(
        self,
        file_path: Path,
        device_type: str,
        vg_column: Optional[str] = None,
        id_column: Optional[str] = None,
        verbose: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Load a single CSV file and create Transfer object."""
        
        if verbose:
            print(f"  Processing: {file_path.name}")
        
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Clean column names
            df.columns = df.columns.str.strip().str.lower()
            
            # Find voltage and current columns
            vg_col = vg_column.lower() if vg_column else self._find_vg_column(df)
            id_col = id_column.lower() if id_column else self._find_id_column(df)
            
            if vg_col is None or id_col is None:
                if verbose:
                    print(f"    ✗ Could not find Vg or Id columns")
                    print(f"    Available columns: {list(df.columns)}")
                return None
            
            # Extract and validate data
            vg_data = df[vg_col].dropna().values
            id_data = df[id_col].dropna().values
            
            # Ensure equal length
            min_length = min(len(vg_data), len(id_data))
            if min_length < 2:
                if verbose:
                    print(f"    ✗ Insufficient data points: {min_length}")
                return None
            
            vg_data = vg_data[:min_length]
            id_data = id_data[:min_length]
            
            # Create Transfer object
            transfer = Transfer(vg_data, id_data, device_type)
            
            # Create result dictionary
            result = {
                "filename": file_path.name,
                "transfer": transfer,
                "data_points": len(vg_data),
                "file_path": str(file_path),
                "vg_column": vg_col,
                "id_column": id_col,
                "device_type": device_type
            }
            
            if verbose:
                print(f"    ✓ Success - {len(vg_data)} data points")
            
            return result
            
        except Exception as e:
            if verbose:
                print(f"    ✗ Error: {e}")
            return None
    
    def _find_vg_column(self, df: pd.DataFrame) -> Optional[str]:
        """Auto-detect gate voltage column."""
        for col in df.columns:
            if any(pattern in col for pattern in ['vg', 'v_g', 'gate', 'vgs', 'v_gs']):
                return col
        return None
    
    def _find_id_column(self, df: pd.DataFrame) -> Optional[str]:
        """Auto-detect drain current column.""" 
        for col in df.columns:
            if any(pattern in col for pattern in ['id', 'i_d', 'drain', 'ids', 'i_ds', 'current']):
                return col
        return None
    
    @staticmethod
    def _extract_number_from_filename(file_path: Path) -> int:
        """Extract number from filename for numerical sorting."""
        match = re.search(r'(\d+)', file_path.name)
        return int(match.group(1)) if match else 0
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get loader metadata."""
        return self.metadata.copy()
    
    def get_filenames(self) -> List[str]:
        """Get list of loaded filenames."""
        return [obj["filename"] for obj in self.transfer_objects]
    
    def analyze_batch(self, show_details: bool = True) -> pd.DataFrame:
        """
        Analyze loaded transfer objects and return summary.
        
        Parameters
        ----------
        show_details : bool, default True
            Whether to print detailed analysis
            
        Returns
        -------
        pd.DataFrame
            Summary statistics for all loaded files
        """
        if not self.transfer_objects:
            print("No transfer objects loaded. Call load_all_files() first.")
            return pd.DataFrame()
        
        # Create summary data
        summary_data = []
        
        for i, item in enumerate(self.transfer_objects):
            filename = item["filename"]
            transfer = item["transfer"]
            
            summary_data.append({
                "index": i,
                "filename": filename,
                "data_points": item["data_points"],
                "gm_max_raw": transfer.gm_max.raw,
                "gm_max_where": transfer.gm_max.where,
                "I_max_raw": transfer.I_max.raw,
                "I_max_where": transfer.I_max.where,
                "I_min_raw": transfer.I_min.raw,
                "I_min_where": transfer.I_min.where,
                "Von_raw": transfer.Von.raw,
                "Von_where": transfer.Von.where,
                "absgm_max_raw": transfer.absgm_max.raw,
                "absI_max_raw": transfer.absI_max.raw,
                "absI_min_raw": transfer.absI_min.raw,
            })
            
            if show_details:
                print(f"\nFile {i+1}: {filename}")
                print(f"  Data points: {item['data_points']}")
                print(f"  gm_max: {transfer.gm_max.raw:.2e} (location: {transfer.gm_max.where})")
                print(f"  I_max: {transfer.I_max.raw:.2e} (location: {transfer.I_max.where})")
                print(f"  I_min: {transfer.I_min.raw:.2e} (location: {transfer.I_min.where})")
                print(f"  Von: {transfer.Von.raw:.3f} V (location: {transfer.Von.where})")
        
        return pd.DataFrame(summary_data)
    
    def show_file_sorting_demo(self) -> None:
        """Demonstrate different file sorting methods."""
        files = list(self.folder_path.glob("*.csv"))
        transfer_files = [f for f in files if 'transfer' in f.name.lower()]
        
        if not transfer_files:
            print("No transfer files found for sorting demo")
            return
        
        print("=== File Sorting Demo ===")
        
        # Alphabetical sorting
        alpha_sorted = sorted(transfer_files)
        print("\nAlphabetical sorting:")
        for i, f in enumerate(alpha_sorted):
            print(f"  {i+1:2d}. {f.name}")
        
        # Numerical sorting  
        num_sorted = sorted(transfer_files, key=self._extract_number_from_filename)
        print("\nNumerical sorting:")
        for i, f in enumerate(num_sorted):
            print(f"  {i+1:2d}. {f.name}")


# Convenience function for direct use
def load_transfer_files(
    folder_path: Union[str, Path],
    device_type: str = "N", 
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Convenience function to load transfer files directly.
    
    Parameters
    ----------
    folder_path : str or Path
        Path to folder containing CSV files
    device_type : str, default "N"
        Device type: "N" or "P"
    **kwargs
        Additional arguments passed to DataLoader.load_all_files()
        
    Returns
    -------
    List[Dict[str, Any]]
        List of transfer objects and metadata
    """
    loader = DataLoader(folder_path)
    return loader.load_all_files(device_type=device_type, **kwargs)