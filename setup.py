import subprocess
import sys
import os

def install_requirements():
    print("Installing requirements...")
    try:
        # Install requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Requirements installed successfully!")
        
        # Verify TensorFlow installation
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        print("TensorFlow installation verified!")
        
        # Verify Keras installation
        import keras
        print(f"Keras version: {keras.__version__}")
        print("Keras installation verified!")
        
        # Verify ReportLab installation
        import reportlab
        print(f"ReportLab version: {reportlab.__version__}")
        print("ReportLab installation verified!")
        
        return True
    except Exception as e:
        print(f"Error during installation: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting setup...")
    if install_requirements():
        print("\nSetup completed successfully!")
        print("\nYou can now run the application with: python app.py")
    else:
        print("\nSetup failed. Please check the error messages above.") 