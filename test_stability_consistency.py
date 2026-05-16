#!/usr/bin/env python3
"""
Test script to validate cluster stability consistency after corrections.
"""
import sys
import os
import logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dbclust.clusterize import Clusterize
from dbclust.phase import Phase
from datetime import datetime
import numpy as np

# Configure logger
logger = logging.getLogger("test_stability")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
logger.addHandler(handler)


def create_test_phases():
    """Create test phases for clustering."""
    from obspy import UTCDateTime
    return [
        Phase(
            network="NET",
            station="STA1",
            location="00",
            channel="EHZ",
            phase="P",
            time=UTCDateTime(2023, 1, 1, 0, 0, i),
            time_uncertainty=0.1,
            proba=0.9,
            info_sta="",
            event_id=f"event_{i%3}" if i % 2 == 0 else None,
            coord={"latitude": 0.0 + i*0.01, "longitude": 0.0 + i*0.01, "elevation": 0.0}
        )
        for i in range(20)
    ]


def test_basic_clustering():
    """Test that basic clustering maintains stability consistency."""
    print("Testing basic clustering...")
    phases = create_test_phases()
    
    clust = Clusterize(
        phases=phases, 
        min_cluster_size=3, 
        average_velocity=5.0,
        min_com_phases=2
    )
    
    # Verify initial consistency
    assert len(clust.clusters) == len(clust.clusters_stability), \
        f"Initial mismatch: {len(clust.clusters)} clusters vs {len(clust.clusters_stability)} stabilities"
    
    # Verify stability values are reasonable (HDBSCAN can return values > 1.0)
    for i, stab in enumerate(clust.clusters_stability):
        assert stab >= 0, f"Invalid stability {stab} for cluster {i}"
        if stab > 1.0:
            print(f"  Note: Cluster {i} has high stability {stab} (> 1.0)")
    
    print(f"✓ Basic clustering: {len(clust.clusters)} clusters with valid stabilities")
    return clust


def test_event_id_merge(clust):
    """Test that event ID merge preserves stability consistency."""
    print("Testing event ID merge...")
    initial_count = len(clust.clusters)
    
    clust.cluster_merge_based_on_eventid()
    
    # Verify consistency after merge (should be maintained or recovered)
    if len(clust.clusters) != len(clust.clusters_stability):
        print(f"✗ Consistency check failed: {len(clust.clusters)} clusters vs {len(clust.clusters_stability)} stabilities")
        return False
    
    # Verify stability values are reasonable (HDBSCAN can return values > 1.0)
    for i, stab in enumerate(clust.clusters_stability):
        assert stab >= 0, f"Invalid stability {stab} after merge for cluster {i}"
        if stab > 1.0:
            print(f"  Note: Cluster {i} has high stability {stab} after merge (> 1.0)")
    
    print(f"✓ Event ID merge: {initial_count} -> {len(clust.clusters)} clusters, stability preserved")
    return True


def test_merge_operation():
    """Test that merge operation maintains consistency."""
    print("Testing merge operation...")
    
    # Create two clusterize objects
    phases1 = create_test_phases()
    phases2 = create_test_phases()
    
    clust1 = Clusterize(phases=phases1, min_cluster_size=3, average_velocity=5.0)
    clust2 = Clusterize(phases=phases2, min_cluster_size=3, average_velocity=5.0)
    
    initial_count1 = len(clust1.clusters)
    initial_count2 = len(clust2.clusters)
    
    # Merge clust2 into clust1
    clust1.merge(clust2)
    
    # Verify consistency
    assert len(clust1.clusters) == len(clust1.clusters_stability), \
        f"Merge operation mismatch: {len(clust1.clusters)} clusters vs {len(clust1.clusters_stability)} stabilities"
    
    expected_count = initial_count1 + initial_count2
    assert len(clust1.clusters) == expected_count, \
        f"Expected {expected_count} clusters after merge, got {len(clust1.clusters)}"
    
    print(f"✓ Merge operation: {initial_count1} + {initial_count2} = {len(clust1.clusters)} clusters")


def test_stability_values():
    """Test that stability values are reasonable."""
    print("Testing stability value ranges...")
    phases = create_test_phases()
    
    clust = Clusterize(phases=phases, min_cluster_size=3, average_velocity=5.0)
    
    # Verify stability values are non-negative (HDBSCAN can return values > 1.0)
    for i, stab in enumerate(clust.clusters_stability):
        assert stab >= 0, f"Invalid stability {stab} for cluster {i}"
    
    # Check that we have some variation in stability values
    unique_stabilities = set(clust.clusters_stability)
    if len(unique_stabilities) > 1:
        min_stab = min(clust.clusters_stability)
        max_stab = max(clust.clusters_stability)
        avg_stab = np.mean(clust.clusters_stability)
        
        print(f"✓ Stability range: min={min_stab:.3f}, max={max_stab:.3f}, avg={avg_stab:.3f}")
        
        # Note: HDBSCAN can return stability values > 1.0 for very persistent clusters
        if max_stab > 1.0:
            print(f"  Note: Some clusters have stability > 1.0 (max={max_stab:.3f}), which is valid for HDBSCAN")
    else:
        print("✓ Single stability value (expected for single cluster)")


def main():
    """Run all stability tests."""
    print("=" * 60)
    print("CLUSTER STABILITY CONSISTENCY TESTS")
    print("=" * 60)
    
    try:
        # Test 1: Basic clustering
        clust = test_basic_clustering()
        
        # Test 2: Event ID merge
        if not test_event_id_merge(clust):
            print("✗ Event ID merge test failed")
            return False
        
        # Test 3: Merge operation
        test_merge_operation()
        
        # Test 4: Stability values
        test_stability_values()
        
        print("=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        return True
        
    except AssertionError as e:
        print(f"✗ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
