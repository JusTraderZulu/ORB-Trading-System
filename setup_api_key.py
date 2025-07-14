#!/usr/bin/env python3
"""
Setup script to configure Polygon API key for ORB Trading System.

Simply put your API key in api_key.txt and run this script.
"""

import os
from pathlib import Path

def setup_api_key():
    """Read API key from api_key.txt and configure .env file."""
    
    # Check if api_key.txt exists
    api_key_file = Path("api_key.txt")
    if not api_key_file.exists():
        print("❌ api_key.txt file not found!")
        print("Please create api_key.txt and put your Polygon API key in it.")
        return False
    
    # Read API key from file
    try:
        with open(api_key_file, 'r') as f:
            api_key = f.read().strip()
    except Exception as e:
        print(f"❌ Error reading api_key.txt: {e}")
        return False
    
    # Validate API key
    if not api_key or api_key == "PUT_YOUR_POLYGON_API_KEY_HERE":
        print("❌ Please replace the placeholder in api_key.txt with your actual API key!")
        print("Your Polygon API key should look something like: 'abcd1234EFGH5678ijkl9012MNOP3456'")
        return False
    
    if len(api_key) < 20:
        print("⚠️  Warning: API key seems too short. Make sure you copied the full key.")
        response = input("Continue anyway? (y/n): ").lower()
        if response != 'y':
            return False
    
    # Update .env file
    env_file = Path(".env")
    
    if env_file.exists():
        # Read existing .env file
        with open(env_file, 'r') as f:
            lines = f.readlines()
        
        # Update POLYGON_API_KEY line
        updated = False
        for i, line in enumerate(lines):
            if line.startswith('POLYGON_API_KEY='):
                lines[i] = f'POLYGON_API_KEY={api_key}\n'
                updated = True
                break
        
        # If not found, add it
        if not updated:
            lines.append(f'POLYGON_API_KEY={api_key}\n')
        
        # Write back to .env file
        with open(env_file, 'w') as f:
            f.writelines(lines)
    else:
        # Create new .env file
        with open(env_file, 'w') as f:
            f.write(f"""# ORB Trading System Environment Variables
POLYGON_API_KEY={api_key}
BARCHART_API_KEY=your_barchart_api_key_here
MLFLOW_TRACKING_URI=file:./mlruns
LOG_LEVEL=INFO
""")
    
    print("✅ API key configured successfully!")
    print(f"✅ .env file updated with API key: {api_key[:8]}...{api_key[-4:]}")
    
    # Test the configuration
    print("\n🧪 Testing API key configuration...")
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        test_key = os.environ.get('POLYGON_API_KEY')
        if test_key == api_key:
            print("✅ API key loaded successfully from .env file!")
            
            # Test import of main modules
            try:
                from orb.data.polygon_loader import download_month
                print("✅ Polygon loader imported successfully!")
                print("\n🚀 Ready to download data! Example:")
                print("   from orb.data.polygon_loader import download_month")
                print("   download_month('AAPL', 2024, 1)")
                
            except Exception as e:
                print(f"⚠️  Warning: Could not import polygon_loader: {e}")
                
        else:
            print("❌ API key test failed - environment not loaded correctly")
            return False
            
    except ImportError:
        print("⚠️  python-dotenv not installed, using system environment only")
    except Exception as e:
        print(f"⚠️  Error testing configuration: {e}")
    
    # Clean up api_key.txt for security
    print(f"\n🔒 For security, removing api_key.txt...")
    try:
        api_key_file.unlink()
        print("✅ api_key.txt removed (API key is now safely stored in .env)")
    except Exception as e:
        print(f"⚠️  Could not remove api_key.txt: {e}")
        print("You may want to delete it manually for security.")
    
    return True

if __name__ == "__main__":
    print("🔧 ORB Trading System - API Key Setup")
    print("=" * 50)
    
    success = setup_api_key()
    
    if success:
        print("\n🎉 Setup complete! Your ORB system is ready to use.")
    else:
        print("\n❌ Setup failed. Please check the errors above and try again.")
        print("\nNeed help? Make sure:")
        print("1. You have a valid Polygon API key")
        print("2. You pasted it correctly in api_key.txt")
        print("3. The key is the full string from polygon.io") 