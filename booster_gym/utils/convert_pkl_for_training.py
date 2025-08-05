#!/usr/bin/env python3
"""
PKL Conversion Utility for Motion Imitation Training

Converts PKL files generated in general-motion-retargeting environment (Python 3.10)
to be compatible with isaac-gym-training environment (Python 3.8).

Usage:
    # From general-motion-retargeting environment (Python 3.10):
    python booster_gym/utils/convert_pkl_for_training.py input.pkl output.pkl
    
    # Or from booster_gym directory:
    python utils/convert_pkl_for_training.py ../../input.pkl motion_data/t1_4dof/output.pkl
"""

import os
import sys
import pickle
import argparse
import numpy as np
from pathlib import Path


def convert_pkl_for_training(input_path: str, output_path: str) -> bool:
    """
    Convert a PKL file from Python 3.10 (GMR) to Python 3.8 (training) compatibility.
    
    Args:
        input_path: Path to the input PKL file (from GMR environment)
        output_path: Path to save the compatible PKL file
        
    Returns:
        True if conversion successful, False otherwise
    """
    try:
        # Load the original pickle file
        print(f"Loading PKL file: {input_path}")
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Successfully loaded PKL file")
        print(f"Data keys: {list(data.keys())}")
        
        # Convert numpy arrays to basic Python data types for maximum compatibility
        converted_data = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                # Convert numpy arrays to Python lists
                converted_data[key] = value.tolist()
                print(f"Converted {key}: shape {value.shape}, dtype {value.dtype} -> list")
            elif value is None:
                converted_data[key] = None
                print(f"Kept {key}: None")
            elif isinstance(value, (int, float, str, bool)):
                converted_data[key] = value
                print(f"Kept {key}: {type(value).__name__}")
            elif isinstance(value, list):
                converted_data[key] = value
                print(f"Kept {key}: list with {len(value)} items")
            else:
                # Try to convert other types as-is
                converted_data[key] = value
                print(f"Kept {key}: {type(value).__name__} (as-is)")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
        
        # Save with protocol 2 for maximum compatibility across Python versions
        print(f"Saving compatible PKL file: {output_path}")
        with open(output_path, 'wb') as f:
            pickle.dump(converted_data, f, protocol=2)
            
        print(f"✅ Successfully converted PKL file!")
        print(f"Input:  {input_path}")
        print(f"Output: {output_path}")
        
        # Verify the converted file can be loaded
        print("Verifying converted file...")
        with open(output_path, 'rb') as f:
            test_data = pickle.load(f)
        print("✅ Converted file loads successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert PKL files from GMR env (Python 3.10) to training env (Python 3.8)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert a PKL file from project root
    python booster_gym/utils/convert_pkl_for_training.py t1_dance.pkl booster_gym/motion_data/t1_4dof/t1_dance.pkl
    
    # Convert from booster_gym directory
    python utils/convert_pkl_for_training.py ../../new_motion.pkl motion_data/t1_4dof/new_motion.pkl
    
    # Batch convert multiple files
    for file in *.pkl; do
        python utils/convert_pkl_for_training.py "$file" "motion_data/t1_4dof/$file"
    done
        """
    )
    
    parser.add_argument('input_pkl', help='Input PKL file path (from GMR environment)')
    parser.add_argument('output_pkl', help='Output PKL file path (for training environment)')
    parser.add_argument('--verify', action='store_true', 
                       help='Verify the converted file loads properly')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_pkl):
        print(f"❌ Input file not found: {args.input_pkl}")
        sys.exit(1)
    
    # Convert the file
    success = convert_pkl_for_training(args.input_pkl, args.output_pkl)
    
    if not success:
        sys.exit(1)
    
    print("\n🎭 PKL file is now ready for motion imitation training!")
    print("You can use it in the isaac-gym-training environment.")


if __name__ == "__main__":
    main()