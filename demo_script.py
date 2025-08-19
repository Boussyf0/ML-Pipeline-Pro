"""
MLOps Pipeline Demonstration Script
==================================
This script helps demonstrate the MLOps pipeline capabilities.
"""
import webbrowser
import time
import os
from pathlib import Path

def print_banner():
    print("🎯" + "=" * 60 + "🎯")
    print("    MLOps Pipeline Demonstration")
    print("    Enterprise Customer Churn Prediction System")
    print("🎯" + "=" * 60 + "🎯")
    print()

def show_demo_menu():
    """Display demo options"""
    print("📋 Demo Menu:")
    print("1. 🌐 Open Interactive Demo UI")
    print("2. 📊 Open MLflow Experiment Tracking") 
    print("3. 🔧 Open API Documentation (Swagger)")
    print("4. ❤️  Check System Health")
    print("5. 📈 View Prometheus Metrics")
    print("6. 🔍 Show System Status")
    print("7. 📋 Display Demo Talking Points")
    print("8. 🎯 Run Live API Test")
    print("0. ❌ Exit Demo")
    print()

def open_demo_ui():
    """Open the interactive demo UI"""
    demo_file = Path("demo_ui.html").absolute()
    print(f"🌐 Opening Interactive Demo UI...")
    print(f"📁 File: {demo_file}")
    webbrowser.open(f"file://{demo_file}")
    print("✅ Demo UI opened in your default browser!")

def open_mlflow():
    """Open MLflow UI"""
    print("📊 Opening MLflow Experiment Tracking...")
    webbrowser.open("http://localhost:5001")
    print("✅ MLflow UI opened!")

def open_api_docs():
    """Open Swagger API docs"""
    print("🔧 Opening API Documentation...")
    webbrowser.open("http://localhost:8000/docs")
    print("✅ Swagger UI opened!")

def check_system_health():
    """Check system health via API"""
    import requests
    try:
        print("❤️  Checking System Health...")
        response = requests.get("http://localhost:8000/health", timeout=5)
        data = response.json()
        
        print("\n📊 System Health Report:")
        print(f"   Status: {data['status']}")
        print(f"   Database: {'✅ Healthy' if data['details']['database_healthy'] else '❌ Unhealthy'}")
        print(f"   Redis: {'✅ Healthy' if data['details']['redis_healthy'] else '❌ Unhealthy'}")
        print(f"   Loaded Models: {data['details']['loaded_models']}")
        print(f"   Uptime: {data.get('uptime_seconds', 'N/A')} seconds")
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        print("   Make sure the API server is running on localhost:8000")

def view_metrics():
    """Open Prometheus metrics"""
    print("📈 Opening Prometheus Metrics...")
    webbrowser.open("http://localhost:8000/metrics")
    print("✅ Metrics endpoint opened!")

def show_system_status():
    """Display current system status"""
    print("\n🔍 System Status Overview:")
    print("=" * 40)
    
    services = [
        ("FastAPI Server", "http://localhost:8000/health", "🚀"),
        ("MLflow Tracking", "http://localhost:5001", "📊"),
        ("PostgreSQL DB", "localhost:5432", "🐘"),
        ("Redis Cache", "localhost:6379", "📦")
    ]
    
    for service_name, endpoint, icon in services:
        try:
            if "localhost:5432" in endpoint or "localhost:6379" in endpoint:
                # These are not HTTP endpoints, just show as configured
                print(f"{icon} {service_name}: ✅ Configured")
            else:
                import requests
                response = requests.get(endpoint, timeout=2)
                status = "✅ Running" if response.status_code == 200 else f"⚠️ HTTP {response.status_code}"
                print(f"{icon} {service_name}: {status}")
        except Exception:
            print(f"{icon} {service_name}: ❌ Not accessible")
    
    print("\n📈 Model Performance Summary:")
    print("   🎯 Accuracy: 78.0%")
    print("   📊 AUC-ROC: 0.784") 
    print("   🔮 Precision: 58.5%")
    print("   📡 Recall: 47.9%")
    
    print("\n💰 Business Impact:")
    print("   📊 Dataset: 5,634 customers")
    print("   📉 Churn Rate: 26.5%")
    print("   💸 Revenue at Risk: $1.16M/year")
    print("   💰 Potential Savings: $233K/year")
    print("   📈 ROI: 4.7x")

def show_talking_points():
    """Display key talking points for demo"""
    print("\n🎤 Demo Talking Points:")
    print("=" * 50)
    
    points = [
        ("🎯 Business Problem", [
            "Customer churn is costing companies millions annually",
            "26.5% of customers are at risk of leaving",
            "$1.16M in annual revenue at risk",
            "Need proactive identification and intervention"
        ]),
        
        ("🏗️ Solution Architecture", [
            "End-to-end MLOps pipeline with production-grade components",
            "Real-time API for instant churn predictions",
            "MLflow for experiment tracking and model versioning",
            "Complete monitoring with health checks and metrics"
        ]),
        
        ("📈 Model Performance", [
            "Trained 3 models: LightGBM, XGBoost, Random Forest",
            "Best model achieves 78% accuracy with 0.784 AUC-ROC",
            "Hyperparameter optimization with 50+ trials per model",
            "Exceeds minimum performance thresholds"
        ]),
        
        ("💰 Business Value", [
            "20% churn reduction saves $233K annually",
            "4.7x ROI on MLOps platform investment",
            "Enables targeted retention campaigns",
            "Reduces customer acquisition costs"
        ]),
        
        ("🚀 Production Features", [
            "API authentication and security",
            "Health monitoring and alerting",
            "Containerized deployment with Docker",
            "Scalable architecture for high throughput"
        ])
    ]
    
    for title, items in points:
        print(f"\n{title}:")
        for item in items:
            print(f"   • {item}")

def run_api_test():
    """Run live API test demonstration"""
    import requests
    import json
    
    print("\n🎯 Live API Test Demonstration:")
    print("=" * 40)
    
    # Test data
    test_customer = {
        "customer_id": "DEMO_001",
        "monthly_charges": 85.50,
        "tenure": 12,
        "total_charges": 1025.40,
        "contract": "Month-to-month",
        "internet_service": "Fiber optic",
        "payment_method": "Electronic check"
    }
    
    print("📋 Test Customer Profile:")
    for key, value in test_customer.items():
        print(f"   {key}: {value}")
    
    try:
        print("\n🔮 Making Prediction Request...")
        response = requests.post(
            "http://localhost:8000/predict",
            json=test_customer,
            timeout=5
        )
        
        if response.status_code == 401:
            print("🔐 Authentication Required (Expected in Production):")
            print("   • API correctly enforces security")
            print("   • In demo: shows endpoint is active and protected")
            print("   • In production: would provide API key for access")
            
        elif response.status_code == 200:
            result = response.json()
            print("✅ Prediction Successful:")
            print(f"   🎯 Churn Probability: {result.get('churn_probability', 'N/A')}")
            print(f"   📊 Risk Level: {result.get('risk_level', 'N/A')}")
            
        else:
            print(f"⚠️ Unexpected Response: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ API Test Failed: {e}")
        print("   Make sure the FastAPI server is running")

def main():
    """Main demo loop"""
    print_banner()
    
    while True:
        show_demo_menu()
        choice = input("Select option (0-8): ").strip()
        
        if choice == "0":
            print("\n👋 Demo completed. Thank you!")
            break
        elif choice == "1":
            open_demo_ui()
        elif choice == "2":
            open_mlflow()
        elif choice == "3":
            open_api_docs()
        elif choice == "4":
            check_system_health()
        elif choice == "5":
            view_metrics()
        elif choice == "6":
            show_system_status()
        elif choice == "7":
            show_talking_points()
        elif choice == "8":
            run_api_test()
        else:
            print("❌ Invalid option. Please try again.")
        
        input("\n⏸️  Press Enter to continue...")
        print("\n" + "─" * 60 + "\n")

if __name__ == "__main__":
    main()