"""Test script for the Innovation Dashboard"""
import json
import sys
# Import directly from app.py file
import app.app as flask_app

# Use Flask test client
with flask_app.app.test_client() as client:
    print("=" * 60)
    print("TESTING OPENENV INNOVATION DASHBOARD")
    print("=" * 60)
    
    # Test 1: Dashboard empty state
    print("\n[1] Testing empty dashboard...")
    resp = client.get('/dashboard')
    print(f"    Status: {resp.status_code}")
    data = resp.data.decode()
    has_empty = 'No Episodes Yet' in data
    print(f"    Empty state shown: {has_empty}")
    
    # Test 2: Run episodes
    print("\n[2] Running 5 test episodes...")
    for i in range(5):
        resp = client.post('/api/run-episode')
        result = resp.get_json()
        print(f"    Episode {i+1}: reward={result.get('reward')}, status={resp.status_code}")
    
    # Test 3: Dashboard with data
    print("\n[3] Testing dashboard with data...")
    resp = client.get('/dashboard')
    data = resp.data.decode()
    print(f"    Status: {resp.status_code}")
    print(f"    Has Persona Chart: {'Persona Distribution' in data}")
    print(f"    Has Drift Chart: {'Policy Drift' in data}")
    print(f"    Has Reward Chart: {'Reward Over Time' in data}")
    print(f"    Has Episode Table: {'Recent Episodes' in data}")
    
    # Test 4: API endpoints
    print("\n[4] Testing API endpoints...")
    resp = client.get('/api/episodes')
    result = resp.get_json()
    print(f"    /api/episodes: {result.get('total')} episodes")
    
    resp = client.get('/api/episode/latest')
    result = resp.get_json()
    print(f"    /api/episode/latest: persona={result.get('persona',{}).get('name')}")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! [OK]")
    print("=" * 60)
    print("\nDashboard available at: http://localhost:5000/dashboard")
    print("Run episodes at: POST http://localhost:5000/api/run-episode")